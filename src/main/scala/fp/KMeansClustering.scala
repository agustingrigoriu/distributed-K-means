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

object KMeansClusteringMain {

  // this is super ugly, just testing idea
  var notConverge: Boolean = true
  var test: Int = 0

  // Since we receive normalized vectors, we just need to compute the dot product.
  def cosineSimilarity(vectorA: SparseVector, vectorB: DenseVector) = {

    // Calculating the sum of Xi * Yi.
    val productsSum = vectorA.indices.map(x => vectorA(x) * vectorB(x)).reduceOption(_ + _)

    productsSum.getOrElse(0.0)
  }


  def recalculateCentroid(dimensionality: Int, oldVector: Seq[DenseVector], clusteredVectors: (Long, Iterable[SparseVector])): DenseVector = {

    val clustered = clusteredVectors._2
    val values: ArrayBuffer[Double] = ArrayBuffer.fill(dimensionality) {
      0.0
    }
    val clusterSize: Int = clustered.size
    clustered.foreach(vector => {
      for (idx <- vector.indices) {
        values(idx) = values(idx) + vector(idx)
      }
    })

    for (i <- 0 to dimensionality - 1) {
      values(i) = values(i) / clusterSize
      if (values(i) != oldVector(test)(i)) {
        notConverge = true
      }
    }

    test = test + 1
    new DenseVector(values = values.toArray)

  }


  def getClosestCentroid(vector: SparseVector, centroids: Seq[DenseVector]): Long = {

    var maxDistance: Double = Double.MinPositiveValue
    var closestCentroidIdx: Int = -1
    for (i <- 0 to centroids.length - 1) {
      val distance = cosineSimilarity(vector, centroids(i))

      if (distance > maxDistance) {
        maxDistance = distance
        closestCentroidIdx = i
      }
    }

    closestCentroidIdx
  }

  def calculateSumOfSquareErrors(clusteredVectors: Iterable[SparseVector], centroid: DenseVector): Double = {

    var sse : Double = 0.0
    clusteredVectors.foreach(vector => {
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
      .add("title", StringType, true)


    //Reading all CSV from input dir. Notice that we are removing NA rows.
    val inputDF = spark.read.schema(schema).csv(inputDir).na.drop()

    inputDF.show(true)


    //Tokenizing the text in the title column.
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("title")
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

    // Now we test that the similarity functions are working by self joining the table. Each row should have a 1.0 (approx) score since the are identical.
    val vectors: RDD[(Long, SparseVector)] = normalizedBagOfWords
      .select("index", "normalizedBagOfWords").rdd
      .map(row => (row.getAs[Long](0), row.getAs[SparseVector](1))).cache()
    // Probably this should be cached.


    // Getting K random vectors to use as centroids.
    var centroids = vectors.takeSample(false, 3).map(tuple => tuple._2.toDense).toSeq

    var dimensionality = centroids(0).size
    var assignedVectors = vectors.map(vector => (getClosestCentroid(vector._2, centroids), vector._2))

    while (notConverge) {
      notConverge = false
      assignedVectors = vectors.map(vector => (getClosestCentroid(vector._2, centroids), vector._2))

      centroids = assignedVectors.groupByKey().map(groupedVectors => recalculateCentroid(dimensionality, centroids, groupedVectors)).collect().toSeq
      test = 0
    }
    assignedVectors.saveAsTextFile(outputDir)
    // newCentroids.saveAsTextFile(outputDir)
    // Assigning vectors to clusters.
    //    // Printing test example.
    //    testJoin.take(40).foreach(println)
    //
    //    testRDD.saveAsTextFile(outputDir)


  }
}
