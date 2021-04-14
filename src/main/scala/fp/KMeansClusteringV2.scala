package fp


import org.apache.log4j.LogManager
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel.DISK_ONLY

import scala.reflect.io.File
import scala.util.control.Breaks.{break, breakable}

object KMeansClusteringV2 {

  def runKMeans(vectorsArray: Array[(Long, SparseVector)], K: Int, I: Int, outputDir: String) : Unit = {


    val spark = SparkSession.builder.appName(s"KMeansClusteringV2-$K")
      .config("spark.driver.memoryOverhead", 1028)
      .config("spark.yarn.executor.memoryOverhead", 1028)
      .config("spark.driver.memory", "3g")
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext


    val vectors = sc.parallelize(vectorsArray)

    val kMeansOutputDir = outputDir + File.separator + s"$K-means"
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    logger.info(s"TEST: R-$K")
    var centroids = vectors
      .takeSample(false, K)
      .zipWithIndex
      .map {
        case ((_, vector), index) => (index, vector.toDense)
      }
      .toSeq


    // I keep in memory the dimension of a vector.
    val dimensions = centroids(0)._2.size

    // Initializing an RDD that will hold every vector with its cluster assignment.
    // (centroidId, (vectorId, vector))
    var labeledVectors: RDD[(Int, (Long, SparseVector))] = sc.emptyRDD[(Int, (Long, SparseVector))]

    var SSE: Double = Double.MaxValue
    val epsilon: Double = 0.001

    breakable {
      for (_ <- 0 to I) {

        // We assign a vector to the closest centroid.
        labeledVectors = vectors.map {
          case (vectorId, vector) => (Utils.getClosestCentroid(vector, centroids), (vectorId, vector))
        }

        // We group the vectors by the assignment label and we recalculate the centroids for each cluster.
        val clusters = labeledVectors
          .groupByKey()
          .map {
            case (centroidId, clusteredVectors) => {
              val newCentroid = Utils.recalculateCentroid(dimensions, clusteredVectors)

              (centroidId, (clusteredVectors, newCentroid))
            }
          }

        // This centroids are updated in the global variable.
        centroids = clusters
          .map {
            case (centroidId, (_, centroid)) => (centroidId, centroid)
          }
          .collect()
          .toSeq


        // We calculate the SSE for each centroid. Then we sum all this values across every cluster.
        val newSSE = clusters
          .mapValues {
            case (clusteredVectors, centroid) => Utils.calculateSumOfSquareErrors(clusteredVectors, centroid)
          }
          .values
          .sum()


        // We check how much the SSE changed w.r.t the previous iteration. If the change is below epsilon, we stop.
        if (SSE - newSSE <= epsilon) {
          break
        }

        // Setting the new SSE.
        SSE = newSSE
      }
    }

    //TODO: Rename file so it shows the K.
    labeledVectors.map(x => s"${x._1},${x._2._1}")
      .saveAsTextFile(kMeansOutputDir)
  }

  def run(inputDir: String, outputDir: String, K: Int, I: Int) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    val spark = SparkSession.builder.appName("KMeansClustering")
      .config("spark.driver.memoryOverhead", 1028)
      .config("spark.yarn.executor.memoryOverhead", 1028)
      .config("spark.driver.memory", "3g")
      //      .config("spark.executor.memory", "6g")
      //      .config("spark.storage.memoryFraction", 0.2)
      //      .config("spark.executor.cores", 1)

      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext

    val data = spark.read.load(inputDir).rdd

    val vectors = data.map(row => (row.getAs[Long](0), row.getAs[SparseVector](1))).collect()


    sc.parallelize(2 to K).foreach(k => {
      runKMeans(vectors, k, I, outputDir)
    })


  }
}
