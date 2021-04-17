package fp

import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StringType, StructType}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.mllib.rdd.MLPairRDDFunctions.fromPairRDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.io.File
import scala.collection.mutable.WrappedArray

object PostProcessing {

  def run(inputDir: String, outputDir: String, documentsDir: String, kWords: Int) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    val spark = SparkSession.builder.appName("KMeansClustering-PostProcessing")
      .config("spark.driver.memoryOverhead", 1024)
      .config("spark.yarn.executor.memoryOverhead", 1024)
      // .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext

    // Read output of KMeans job.
    // Input format: (clusterId, documentId).
    val clusteringSchema = new StructType()
      .add("clusterId", LongType, nullable = false)
      .add("documentId", LongType, true)

    val labeledDocumentsIds = spark.read.schema(clusteringSchema).csv(inputDir).na.drop()

    // Read documents input.
    // Format: (documentId, document)
    val documentsSchema = new StructType()
      .add("documentId", LongType, nullable = false)
      .add("document", StringType, true)

    val documents = spark.read.schema(documentsSchema).csv(documentsDir).na.drop()

    //Tokenizing the text in the title column.
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("document")
      .setOutputCol("tokens")
      .setPattern(Utils.TokensRegex)

    val tokenizedDF = regexTokenizer.transform(documents)

    val stemmedDF = new Stemmer()
      .setInputCol("tokens")
      .setOutputCol("stemmedTokens")
      .setLanguage("English")
      .transform(tokenizedDF)

    //Removing stop words from the tokenized arrays of words.
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("stemmedTokens")
      .setOutputCol("filteredTokens")

    val originalStopWords = stopWordsRemover.getStopWords
    val additionalStopWords = Utils.stopWords
    val alStopWords = originalStopWords ++ additionalStopWords
    stopWordsRemover.setStopWords(alStopWords)

    val removedStopWords = stopWordsRemover.transform(stemmedDF)

    val joinedClustersXDocuments = labeledDocumentsIds
      .join(removedStopWords, "documentId")
      .drop("document", "tokens", "stemmedTokens")
      .rdd

    val clusteredDocuments = joinedClustersXDocuments
      .map(row => {
        val clusterId: Long = row.getAs[Long](1)
        val tokens = row.getAs[WrappedArray[String]](2)

        (clusterId, tokens)
      })
      .groupByKey()

    val topKWordsPerCluster = clusteredDocuments
      .flatMapValues(arrays => arrays)
      .flatMapValues(tokens => tokens)
      .map {
        case (clusterId, word) => ((clusterId, word), 1)
      }
      .reduceByKey(_ + _)
      .map {
        case ((clusterId, word), freq) => (clusterId, (word, freq))
      }
      .groupByKey()
      .mapValues(
        tokens => tokens.toSeq.sortBy {
          case (_, freq) => freq
        }.reverse.take(kWords)
      )

    topKWordsPerCluster.saveAsTextFile(outputDir)

  }

}
