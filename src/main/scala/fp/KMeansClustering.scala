package fp


import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StringType, StructType}
import org.apache.spark.ml.feature.{CountVectorizer, Normalizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.rdd.RDD

object KMeansClusteringMain {

  // Since we receive normalized vectors, we just need to compute the dot product.
  def cosineSimilarity(vectorA: SparseVector, vectorB: SparseVector) = {
    val commonIndices = vectorA.indices intersect vectorB.indices

    // Calculating the sum of Xi * Yi.
    val productsSum = commonIndices.map(x => vectorA(x) * vectorB(x)).reduceOption(_ + _)

    productsSum.getOrElse(0.0)
  }

  def recalculateCentroid(clusteredVectors: Iterable[SparseVector]): SparseVector = {

    var avgVector: SparseVector = null
    val vectorSize: Double = if (clusteredVectors.size == 0) 1 else clusteredVectors.size.toDouble

    clusteredVectors.foreach(vector => {

      if (avgVector == null) {
        avgVector = vector
      } else {
        val newIndices = (avgVector.indices union vector.indices).distinct
        scala.util.Sorting.quickSort(newIndices)
        val newValues = newIndices.map(i => {

          val value1 = if (avgVector.indices.contains(i)) avgVector(i) else 0.0
          val value2 = if (vector.indices.contains(i)) vector(i) else 0.0

          value1 + value2
        })

        avgVector = new SparseVector(vector.size, newIndices, newValues)
      }

    })

    val averagedValues = avgVector.values.map(value => value / vectorSize)
    new SparseVector(avgVector.size, avgVector.indices, averagedValues)

  }


  def getClosestCentroid(vector: SparseVector, centroids: Seq[SparseVector]): Long = {

    var minDistance: Double = Double.MaxValue
    var closestCentroidIdx: Int = -1
    for (i <- 0 to centroids.length - 1) {
      val distance = cosineSimilarity(vector, centroids(i))

      if (distance < minDistance) {
        minDistance = distance
        closestCentroidIdx = i
      }
    }

    closestCentroidIdx
  }

  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 2) {
      logger.error("Usage:\n")
      System.exit(1)
    }

    val spark = SparkSession.builder.appName("KMeansClustering")
      .config("spark.driver.memoryOverhead", 1024)
      .config("spark.yarn.executor.memoryOverhead", 1024)
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext

    val inputDir: String = args(0)
    val outputDir: String = args(1)

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
      .map(row => (row.getAs[Long](0), row.getAs[SparseVector](1)))
    // Probably this should be cached.


    // Getting K random vectors to use as centroids.
    val centroids = vectors.takeSample(false, 10).map(tuple => tuple._2).toSeq


    // Assigning vectors to clusters.
    val assignedVectors = vectors.map(vector => (getClosestCentroid(vector._2, centroids), vector._2))
    val newCentroids = assignedVectors.groupByKey().map(groupedVectors => recalculateCentroid(groupedVectors._2))

    newCentroids.saveAsTextFile(outputDir)
    //    // Printing test example.
    //    testJoin.take(40).foreach(println)
    //
    //    testRDD.saveAsTextFile(outputDir)


  }
}
