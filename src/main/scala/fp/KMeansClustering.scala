package fp


import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession


object KMeansClusteringMain {

  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 3) {
      logger.error("Usage:\n")
      System.exit(1)
    }

    val spark = SparkSession.builder.appName("KMeansClustering")
      .config("spark.driver.memoryOverhead", 1024)
      .config("spark.yarn.executor.memoryOverhead", 1024)
//      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext

    val inputDir: String = args(0)
    val outputDir: String = args(2)
    val k: Int = args(0).toInt;
    val iterationsNumber: Int = args(1).toInt;

    //     Delete output directory.
//    val hadoopConf = new org.apache.hadoop.conf.Configuration
//    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
//    try {
//      hdfs.delete(new org.apache.hadoop.fs.Path(outputDir), true)
//    } catch {
//      case _: Throwable => {}
//    }


  }
}