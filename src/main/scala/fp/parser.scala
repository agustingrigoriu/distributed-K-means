package fp

import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StringType, StructType}
import org.apache.spark.sql.functions._ 

object parserMain {

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

    import spark.implicits._
    val sc = spark.sparkContext

    val tweetsJson: String = args(0)
    val usersJson: String = args(1)

   
    val tweets = spark.read.json(tweetsJson)
    val users = spark.read.json(usersJson)


    val df = tweets.as("S1").select("created_at", "text", "user_id")
    .join(users.select("id", "name").as("S2")).where($"S1.user_id" === $"S2.id")
    val output = df.drop("id")
    val indexOutput = output.withColumn("id",monotonicallyIncreasingId)
    val kMeansData = indexOutput.select("id","text")
    val metaData = indexOutput.select("id","created_at","user_id","name")
    output.coalesce(1).write.csv("/mnt/d/data/total.csv")
    
  }
}