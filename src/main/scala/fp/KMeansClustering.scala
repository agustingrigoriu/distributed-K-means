package fp


import breeze.linalg.Axis._1
import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, VectorUDT}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types.{LongType, StringType, StructType, StructField}
import org.apache.spark.ml.feature.{CountVectorizer, Normalizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.rdd.RDD

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, Map}
import scala.util.control.Breaks.{break, breakable}

object KMeansClusteringMain {

  // Since we receive normalized vectors, we just need to compute the dot product.
  def cosineSimilarity(vectorA: SparseVector, vectorB: DenseVector) = {

    // Calculating the sum of Xi * Yi.
    val productsSum = vectorA.indices.map(x => vectorA(x) * vectorB(x)).reduceOption(_ + _)

    productsSum.getOrElse(0.0)
  }


  def recalculateCentroid(dimensionality: Int, clusteredVectors: Iterable[(Long, SparseVector)]): DenseVector = {

    val values: ArrayBuffer[Double] = ArrayBuffer.fill(dimensionality) {
      0.0
    }
    val clusterSize: Int = clusteredVectors.size
    clusteredVectors.foreach(vectorTuple => {
      val vector = vectorTuple._2
      for (idx <- vector.indices) {
        values(idx) = values(idx) + vector(idx)
      }
    })

    for (i <- 0 to dimensionality - 1) {
      values(i) = values(i) / clusterSize
    }

    new DenseVector(values = values.toArray)

  }


  def getClosestCentroid(vector: SparseVector, centroids: Seq[(Int, DenseVector)]): Int = {

    var maxDistance: Double = Double.MinValue
    var closestCentroidId: Int = -1
    for (centroid <- centroids) {

      val centroidId = centroid._1
      val centroidVector = centroid._2

      val distance = cosineSimilarity(vector, centroidVector)

      if (distance > maxDistance) {
        maxDistance = distance
        closestCentroidId = centroidId
      }
    }

    closestCentroidId
  }

  def calculateSumOfSquareErrors(clusteredVectors: Iterable[(Long, SparseVector)], centroid: DenseVector): Double = {

    var sse: Double = 0.0
    clusteredVectors.foreach(vectorTuple => {

      val vector = vectorTuple._2

      for (idx <- vector.indices) {
        val difference = vector(idx) - centroid(idx)
        val squareError = math.pow(difference, 2)
        sse = sse + squareError
      }
    })

    sse
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

    // K-MEANS ALGORITHM VERSION 1.

    // We keep one RDD in memory with (ID, VECTOR).
    val vectors: RDD[(Long, SparseVector)] = data.map(row => (row.getAs[Long](0), row.getAs[SparseVector](1)))
      .cache()


    // Getting K random vectors to use as centroids. And parsing them to DenseVectors.
    // Resulting tuple is (centroidId, centroid)
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
    var labeledVectors = sc.emptyRDD[(Int, (Long, SparseVector))]

    var SSE: Double = Double.NegativeInfinity
    val epsilon: Double = 0.001

    breakable {
      for (_ <- 0 to I) {

        // We assign a vector to the closest centroid.
        labeledVectors = vectors.map {
          case (vectorId, vector) => (getClosestCentroid(vector, centroids), (vectorId, vector))
        }

        // We group the vectors by the assignment label and we recalculate the centroids for each cluster.
        val clusters = labeledVectors
          .groupByKey()
          .map {
            case (centroidId, clusteredVectors) => {
              val newCentroid = recalculateCentroid(dimensions, clusteredVectors)

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
            case (clusteredVectors, centroid) => calculateSumOfSquareErrors(clusteredVectors, centroid)
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

    labeledVectors.map(x=>(x._2._1, x._1))
    .saveAsTextFile(outputDir)
  }
}
