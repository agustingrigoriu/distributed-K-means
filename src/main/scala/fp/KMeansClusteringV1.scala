package fp


import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

import scala.util.control.Breaks.{break, breakable}
import breeze.linalg.min

import scala.collection.mutable.ArrayBuffer
import scala.reflect.io.File

object KMeansClusteringV1 {

  def run(inputDir: String, outputDir: String, maxK: Int, I: Int): ArrayBuffer[(Int, Double, String)] = {

    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    val spark = SparkSession.builder.appName("KMeansClustering")
      .config("spark.driver.memoryOverhead", 1024)
      .config("spark.yarn.executor.memoryOverhead", 1024)
      // .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext

    val data = spark.read.load(inputDir).rdd

    // K-MEANS ALGORITHM VERSION 1.
    var SSEList = new ArrayBuffer[(Int, Double, String)]()

    val vectors: RDD[(Long, SparseVector)] = data.map(row => (row.getAs[Long](0), row.getAs[SparseVector](1)))
      .cache()

    // We keep one RDD in memory with (ID, VECTOR).
    for (k <- 2 to maxK) {

      // Getting K random vectors to use as centroids. And parsing them to DenseVectors.
      // Resulting tuple is (centroidId, centroid)
      var centroids = vectors
        .takeSample(false, k)
        .zipWithIndex
        .map {
          case ((_, vector), index) => (index, vector.toDense)
        }
        .toSeq


      // I keep in memory the dimension of a vector.
      val dimensions = centroids(0)._2.size

      // Initializing an RDD that will hold every vector with its cluster assignment.
      // (centroidId, (vectorId, vector))
      var labeledVectors = sc.emptyRDD[(Int, (Long, SparseVector))]

      var SSE: Double = Double.MaxValue
      val epsilon: Double = Utils.Epsilon

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
          if ((SSE - newSSE).abs <= epsilon) {
            SSE = newSSE
          }

          // Setting the new SSE.
          SSE = newSSE
        }
      }


      // We save the output for this K to disk.
      val kMeansOutputDir = outputDir + File.separator + s"$k-means"
      labeledVectors.map(x => s"${x._1},${x._2._1}")
      .saveAsTextFile(kMeansOutputDir)

      // Adding a tuple (K, SSE) to the result array.
      val resultTuple = (k, SSE, kMeansOutputDir)
      SSEList += resultTuple
    }

    SSEList
  }
}
