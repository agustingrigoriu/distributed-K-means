package fp

import breeze.linalg.Axis._1
import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StringType, StructType}
import org.apache.spark.ml.feature.{CountVectorizer, Normalizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.rdd.RDD

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, Map}
import scala.util.control.Breaks.{break, breakable}

object PreProcessing {

  def run(inputDir: String, outputDir: String) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    val spark = SparkSession.builder.appName("KMeansClustering-PreProcessing")
      .master("local[*]")
    .config("spark.driver.memoryOverhead", 1024)
    .config("spark.yarn.executor.memoryOverhead", 1024)

    .getOrCreate()

    val sc = spark.sparkContext

    val schema = new StructType()
    .add("index", LongType, nullable = false)
    .add("tweet", StringType, true)


    //Reading all CSV from input dir. Notice that we are removing NA rows.
    val inputDF = spark.read.schema(schema).csv(inputDir).na.drop()

    //Tokenizing the text in the title column.
    val regexTokenizer = new RegexTokenizer()
    //      .setGaps(false)
    //      .setPattern("#(\\w+)")
    .setInputCol("tweet")
    .setOutputCol("tokens")
    .setPattern(Utils.TokensRegex)

    val tokenizedDF = regexTokenizer.transform(inputDF)

    val stemmedDF = new Stemmer()
      .setInputCol("tokens")
      .setOutputCol("stemmedTokens")
      .setLanguage("English")
      .transform(tokenizedDF)

    //Removing stop words from the tokenized arrays of words.
    val stopWordsRemover = new StopWordsRemover()
    .setInputCol("stemmedTokens")
    .setOutputCol("filteredTokens")

    val removedStopWords = stopWordsRemover.transform(stemmedDF)

    //Parsing each list of words to a CountVectorizer object or "Bag of Words".
    val countVectorizer = new CountVectorizer()
    .setInputCol("filteredTokens")
    .setOutputCol("bagOfWords")
    .fit(removedStopWords)

    val bagOfWords = countVectorizer.transform(removedStopWords)

    // Normalizing our vectors so we can make the cosine similarity calculation more straightforward.
    val normalizer = new Normalizer()
    .setInputCol("bagOfWords")
    .setOutputCol("normalizedBagOfWords")

    val normalizedBagOfWords = normalizer.transform(bagOfWords)

    val data = normalizedBagOfWords
    .select("index", "normalizedBagOfWords")
    .write.mode("append").format("parquet").save(outputDir)
  }
}