package fp

import org.apache.log4j.LogManager

import java.io.File
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.rdd.RDD

object Main {
  def main(args: Array[String]): Unit = {

    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    val inputDir: String = args(0)
    val outputDir: String = args(1)
    val preProcessingOutputDir: String = outputDir + File.separator + "pre-processing"
    val kMeansOutputDirV1: String = outputDir + File.separator + "k-means-v1"
    val kMeansOutputDirV2: String = outputDir + File.separator + "k-means-v2"
    val kMeansSSEDir: String = outputDir + File.separator + "sse"
    val postProcessingOutputDir: String = outputDir + File.separator + "post-processing"
    val version: Int = args(2).toInt
    val K: String = args(3)
    val I: String = args(4)
    val topKWords: String = args(5)


    // Remove output dir entirely.
    val hadoopConf = new org.apache.hadoop.conf.Configuration
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    try {
      hdfs.delete(new org.apache.hadoop.fs.Path(outputDir), true)
    } catch {
      case _: Throwable => {}
    }

    // Pre-Processing Step
    fp.PreprocessingMain.main(Array(inputDir, preProcessingOutputDir))

    // KMeans Execution
    if (version == 1) {
      // The result is the list of every K with the respective SSE.
      val output = fp.KMeansClusteringV1Main.run(Array(preProcessingOutputDir, kMeansOutputDirV1, K, I))
      logger.info(output)

    } else {
      fp.KMeansClusteringV2Main.main(Array(preProcessingOutputDir, kMeansOutputDirV2, K, I))
    }

    // Post-Processing Step
    //  fp.PostProcessingMain.main(Array(kMeansOutputDirV1, postProcessingOutputDir, inputDir, topKWords, K))
  }
}
