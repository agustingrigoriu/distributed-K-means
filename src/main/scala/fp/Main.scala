package fp

import java.io.File
import org.apache.spark.ml.linalg.{SparseVector}
import org.apache.spark.rdd.RDD

object Main {
  def main(args: Array[String]): Unit = {

    val inputDir: String = args(0)
    val outputDir: String = args(1)
    val preProcessingOutputDir: String = outputDir + File.separator + "pre-processing"
    val kMeansOutputDir: String = outputDir + File.separator + "k-means"
    val kMeansSSEDir: String = outputDir + File.separator + "sse"
    val postProcessingOutputDir: String = outputDir + File.separator + "post-processing"
    val version: Int = args(2).toInt
    val K: String = args(3)
    val I: String = args(4)
    val topKWords: String = args(5)

    // Pre-Processing Step
    // fp.PreprocessingMain.main(Array(inputDir, preProcessingOutputDir))

    // KMeans Execution
    // if (version == 1) {
    //   val output = fp.KMeansClusteringMain.runKMeans(preProcessingOutputDir, K, I)
    //   val hadoopConf = new org.apache.hadoop.conf.Configuration
    //   val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    //   try {
    //     hdfs.delete(new org.apache.hadoop.fs.Path(kMeansOutputDir), true)
    //     hdfs.delete(new org.apache.hadoop.fs.Path(kMeansSSEDir), true)
    //   } catch {
    //     case _: Throwable => {}
    //   }
    //   output._1.coalesce(1).saveAsTextFile(kMeansSSEDir)
    //   output._2.saveAsTextFile(kMeansOutputDir)
      
    // } else {
    //   fp.KMeansClusteringV2Main.main(Array(preProcessingOutputDir, kMeansOutputDir, K, I))
    // }
    // Post-Processing Step
    fp.PostProcessingMain.main(Array(kMeansOutputDir, postProcessingOutputDir, inputDir, topKWords, K))
  }
}
