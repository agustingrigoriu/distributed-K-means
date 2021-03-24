package fp


import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StringType, StructType}


object KMeansClusteringMain {


  def parseToBOW(text: String): Array[(String, Int)] = {
    val splitText = text.split("\\s")
    val wordsCount = splitText.map(word => (word, 1))
    val groupedAndReducedCounts = wordsCount.groupBy(_._1).mapValues(_.map(_._2).sum)
    val bow = groupedAndReducedCounts.toArray

    bow
  }

  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 1) {
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
//    val outputDir: String = args(2)
//    val k: Int = args(0).toInt;
//    val iterationsNumber: Int = args(1).toInt;


    val schema = new StructType()
      .add("created_utc", LongType, nullable = false)
      .add("title",StringType,true)

    //read all files from a folder
    val df = spark.read.schema(schema).json(inputDir)
    df.show(true)

    val bagOfWords = df.rdd
      .filter( row => !row.isNullAt(1))
      .map(row => (row.getLong(0), row.getString(1)))
      .mapValues(title => parseToBOW(title))

//    .filter(x => x.Document)
//    .map( x => x.Document.replace(',',' ').replace('.',' ').replace('-',' ').lower())\
//      .flatMap(lambda x: x.split())\
//      .map(lambda x: (x, 1))
//    bow0.reduceByKey(lambda x,y:x+y).take(50)
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