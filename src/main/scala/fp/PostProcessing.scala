package fp

import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StringType, StructType}
import org.apache.spark.ml.feature.{StopWordsRemover}
import org.apache.spark.mllib.rdd.MLPairRDDFunctions.fromPairRDD
object  PostProcessingMain {

  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 3) {
      logger.error("Usage:\n")
      System.exit(1)
    }

    val inputDir: String = args(0)
    val outputDir: String = args(1)
    val tweetsDir: String = args(2)

    val spark = SparkSession.builder.appName("KMeansClustering-PostProcessing")
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

    val joined = inputDF.join(tweets, "index").rdd
    val hi = new StopWordsRemover()
    val testWords = hi.getStopWords

    val clusters = joined.map(row => (row.getAs[Long](1),row.getAs[String](2))).groupByKey()
    // val groupedClusters = clusters.mapValues(group => Utils.getTopKWords(group, 5))
    // groupedClusters.saveAsTextFile(outputDir)

    val filteredWords = clusters.flatMapValues(iter => iter)
    .flatMapValues(line => line.split(" "))
    .filter(word => !testWords.contains(word._2))
    .map(word=> (word,1))
    .reduceByKey(_+_)
    val n = 20
    // topByKey not working as expected, i think maybe because values are in array, so it thinks theres only 1 value?
    val output = filteredWords.map(input => (input._1._1, (input._1._2, input._2))).sortBy(_._2._2 * -1).groupByKey().topByKey(n)
    val test = output.mapValues(x=>x(0)).map(y=>(y._1,y._2.toList))
    test.saveAsTextFile(outputDir)

    
    // Group RDD by key => (clusterId, Iterable[(docId, doc)]

    // For each group, extract K top words from the group and map it to a list of them.
    // groupuedClusters = clusters.mapValues( group => getTopKWords(k))
    // Save the results in a file
  }

}
