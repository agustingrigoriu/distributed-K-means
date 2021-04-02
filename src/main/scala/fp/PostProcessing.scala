package fp

import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StringType, StructType}

class PostProcessing {
  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 2) {
      logger.error("Usage:\n")
      System.exit(1)
    }

    val inputDir: String = args(0)
    val outputDir: String = args(1)

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


    // Group RDD by key => (clusterId, Iterable[(docId, doc)]

    // For each group, extract K top words from the group and map it to a list of them.
    // groupuedClusters = clusters.mapValues( group => getTopKWords(k))
    // Save the results in a file
  }

}
