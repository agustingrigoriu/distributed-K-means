package fp


import breeze.linalg.Axis._1
import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StringType, StructType}
import org.apache.spark.ml.feature.{CountVectorizer, Normalizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
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

    var maxDistance: Double = Double.MinPositiveValue
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
    if (args.length != 2) {
      logger.error("Usage:\n")
      System.exit(1)
    }

    val inputDir: String = args(0)
    val outputDir: String = args(1)

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

    val schema = new StructType()
      .add("index", LongType, nullable = false)
      .add("tweet", StringType, true)


    //Reading all CSV from input dir. Notice that we are removing NA rows.
    val inputDF = spark.read.schema(schema).csv(inputDir).na.drop()

    inputDF.show(true)


    //Tokenizing the text in the title column.
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("tweet")
      .setOutputCol("tokens")
      .setPattern("\\W")

    val tokenizedDF = regexTokenizer.transform(inputDF)

    tokenizedDF.show(true)

    //Removing stop words from the tokenized arrays of words.
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filteredTokens")

    val removedStopWords = stopWordsRemover.transform(tokenizedDF)

    removedStopWords.show(true)


    //Parsing each list of words to a CountVectorizer object or "Bag of Words".
    val countVectorizer = new CountVectorizer()
      .setInputCol("filteredTokens")
      .setOutputCol("bagOfWords")
      .fit(removedStopWords)

    val bagOfWords = countVectorizer.transform(removedStopWords)

    bagOfWords.show(true)

    val normalizer = new Normalizer()
      .setInputCol("bagOfWords")
      .setOutputCol("normalizedBagOfWords")

    val normalizedBagOfWords = normalizer.transform(bagOfWords)

    val data = normalizedBagOfWords
      .select("index", "tweet", "normalizedBagOfWords").rdd

    val tweets = data.map(row => (row.getAs[Long](0), row.getAs[String](1)))
    val vectors: RDD[(Long, SparseVector)] = data.map(row => (row.getAs[Long](0), row.getAs[SparseVector](2)))
      .cache()


    // Getting K random vectors to use as centroids.
    var centroids = vectors
      .takeSample(false, 20)
      .zipWithIndex
      .map {
        case ((_, vector), index) => (index, vector.toDense)
      }
      .toSeq

    val dimensions = centroids(0)._2.size

    var labeledVectors = sc.emptyRDD[(Int, (Long, SparseVector))]

    var SSE = -1
    val epsilon: Double = 0.001


    breakable {
      for (i <- 0 to 20) {

        labeledVectors = vectors.map(vector => (getClosestCentroid(vector._2, centroids), vector))

        val groupedVectors = vectors.map(vector => (getClosestCentroid(vector._2, centroids), vector)).groupByKey()

        val centroidsRDD = groupedVectors.mapValues(groupedVectors => recalculateCentroid(dimensions, groupedVectors))

        centroids = centroidsRDD
          .collect()
          .toSeq

        val newSSE = groupedVectors.join(centroidsRDD)
          .mapValues {
            case (clusteredVectors, centroid) => calculateSumOfSquareErrors(clusteredVectors, centroid)
          }
          .values
          .sum()

        if (SSE - newSSE <= epsilon) {
          break
        }

      }
    }

    val labeledTweets = labeledVectors.map {
      case (label, (id, vector)) => (id, label)
    }.join(tweets)
    labeledTweets.saveAsTextFile(outputDir)
    // newCentroids.saveAsTextFile(outputDir)
    // Assigning vectors to clusters.
    //    // Printing test example.
    //    testJoin.take(40).foreach(println)
    //
    //    testRDD.saveAsTextFile(outputDir)


  }
}
