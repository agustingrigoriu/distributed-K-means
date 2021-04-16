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

  def run(inputDir: String, outputDir: String, tweetsDir: String, kWords: Int) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    val spark = SparkSession.builder.appName("KMeansClustering-PostProcessing")
      .config("spark.driver.memoryOverhead", 1024)
      .config("spark.yarn.executor.memoryOverhead", 1024)
      // .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext


    // Read input of KMeansClustering Job.
    // Input format: (clusterId, (docId, doc)).
    val schema = new StructType()
      .add("clusterID", LongType, nullable = false)
      .add("index", LongType, true)

    val inputDF = spark.read.schema(schema).csv(inputDir).na.drop()

    val schema1 = new StructType()
      .add("index", LongType, nullable = false)
      .add("tweet", StringType, true)

    val tweets = spark.read.schema(schema1).csv(tweetsDir).na.drop()

    //Tokenizing the text in the title column.
    val regexTokenizer = new RegexTokenizer()
      //      .setGaps(false)
      //      .setPattern("#(\\w+)")
      .setInputCol("tweet")
      .setOutputCol("tokens")
      .setPattern("\\W")

    val tokenizedDF = regexTokenizer.transform(tweets)

    val stemmedDF = new Stemmer()
      .setInputCol("tokens")
      .setOutputCol("stemmedTokens")
      .setLanguage("English")
      .transform(tokenizedDF)

    //Removing stop words from the tokenized arrays of words.
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("stemmedTokens")
      .setOutputCol("filteredTokens")

    val testWords = stopWordsRemover.getStopWords
    val addWords = List(",", "", "-", "&", "rt", "amp")
    val allWords = addWords ++ testWords
    stopWordsRemover.setStopWords(allWords.toArray)

    val removedStopWords = stopWordsRemover.transform(tokenizedDF)

    val joined = inputDF.join(removedStopWords, "index").drop("tweet", "tokens").rdd

    val clusters = joined.map(row => (row.getAs[Long](1), row.getAs[WrappedArray[String]](2))).groupByKey()
    // val groupedClusters = clusters.mapValues(group => Utils.getTopKWords(group, 5))
    // groupedClusters.saveAsTextFile(outputDir)
    val filteredWords = clusters.flatMapValues(iter => iter)
      .flatMapValues(line => line)
      .map(word => (word, 1))
      .reduceByKey(_ + _)
    // topByKey not working as expected, i think maybe because values are in array, so it thinks theres only 1 value?
    val topK = filteredWords.map(input => (input._1._1, (input._2, input._1._2))).sortBy(_._2._2).topByKey(kWords)
    import spark.implicits._
    val output = topK.toDS()
    output.coalesce(1).write.json(outputDir)
    //output.saveAsTextFile(outputDir)


    // Group RDD by key => (clusterId, Iterable[(docId, doc)]

    // For each group, extract K top words from the group and map it to a list of them.
    // groupuedClusters = clusters.mapValues( group => getTopKWords(k))
    // Save the results in a file
  }

}
