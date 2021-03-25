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
    val productsSum = commonIndices.map(x => vectorA(x) * vectorB(x)).reduceOption(_+_)

    productsSum.getOrElse(0.0)
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
      .add("created_utc", LongType, nullable = false)
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
    val testRDD: RDD[(Long, SparseVector)] = normalizedBagOfWords
      .select("created_utc", "normalizedBagOfWords").rdd
      .map(row => (row.getAs[Long](0), row.getAs[SparseVector](1)))


    val testJoin = testRDD.join(testRDD).mapValues(arr => cosineSimilarity(arr._1, arr._2))

    // Printing test example.
    testJoin.take(40).foreach(println)

    testRDD.saveAsTextFile(outputDir)


  }
}