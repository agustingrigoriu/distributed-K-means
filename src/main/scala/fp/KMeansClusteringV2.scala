package fp


import org.apache.log4j.LogManager
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.rdd.RDD

import scala.reflect.io.File
import scala.util.control.Breaks.{break, breakable}

object KMeansClusteringV2Main {


  def runKMeans(sparkContext: SparkContext, broadcastVectors: Array[(Long, SparseVector)], K: Int, I: Int, outputDir: String) {


    val kMeansOutputDir = outputDir + File.separator + s"$K-Means"

    val vectors = sparkContext.parallelize(broadcastVectors)

    vectors.take(5).foreach(println)

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
    var labeledVectors = sparkContext.emptyRDD[(Int, (Long, SparseVector))]

    var SSE: Double = Double.NegativeInfinity
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
    labeledVectors.map(x => (x._2._1, x._1))
      .saveAsTextFile(kMeansOutputDir)
  }

  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 4) {
      logger.error("Usage:\n")
      System.exit(1)
    }

    val inputDir: String = args(0)
    val outputDir: String = args(1)
    val K: Int = args(2).toInt
    val I: Int = args(3).toInt

    val spark = SparkSession.builder.appName("KMeansClustering")
      .config("spark.driver.memoryOverhead", 1024)
      .config("spark.yarn.executor.memoryOverhead", 1024)
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext

    val hadoopConf = new org.apache.hadoop.conf.Configuration
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    try {
      hdfs.delete(new org.apache.hadoop.fs.Path(outputDir), true)
    } catch {
      case _: Throwable => {}
    }

    val data = spark.read.load(inputDir).rdd

    data.take(5).foreach(println)
    //TODO: Rename file so it shows the K.
    val vectors: RDD[(Long, SparseVector)] = data.map(row => (row.getAs[Long](0), row.getAs[SparseVector](1)))
      .cache()

    vectors.take(5).foreach(println)

    val broadcastVectors = sc.broadcast(vectors.collect())


    for (k <- 1 to K) {
      runKMeans(sc, broadcastVectors.value, k, I, outputDir)
    }

  }
}
