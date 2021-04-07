package fp

import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import scala.collection.immutable.ListMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import org.apache.spark.ml.feature.{StopWordsRemover}
import scala.util.control.Breaks.{break, breakable}

object Utils {

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

  def getTopKWords(words: Iterable[String], K: Int): ListMap[String, Int] = {
    val hi = new StopWordsRemover()
    val testWords = hi.getStopWords
    var textDict = new HashMap[String, Int].withDefaultValue(0)
    for (word <- words) {
      if (!testWords.contains(word)) {
        textDict(word) += 1
      }
    }
    val res = ListMap(textDict.toSeq.sortBy(_._2):_*).take(K)
    res
  }

    def getTopKWords1(words: Iterable[(String, Int)], K: Int): List[(String, Int)] = {
    val res = new ArrayBuffer[(String, Int)]()
    var count = 0
    breakable{
      for (word <- words) {
        res += (word)
      }
      count += 1
      if (count > K) {
        break
      }
    }
    res.toList
  }

}
