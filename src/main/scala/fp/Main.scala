package fp

import org.apache.log4j.LogManager

import java.io.File

object Main {

  def main(args: Array[String]): Unit = {

    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    val version: Int = args(2).toInt
    val inputDir: String = args(0)

    // We will have one folder per version: /output/v1 and /output/v2
    val outputDir: String = args(1) + File.separator + s"v$version"
    val preProcessingOutputDir: String = outputDir + File.separator + "pre-processing"
    val kMeansOutput: String = outputDir + File.separator + "k-means"
    val statisticsOutputDir: String = outputDir + File.separator + "statistics" // In case we want to save this info.
    val postProcessingOutputDir: String = outputDir + File.separator + "post-processing"
    val K: Int = args(3).toInt
    val I: Int = args(4).toInt
    val topKWords: Int = args(5).toInt


    // Remove output dir entirely.
    val hadoopConf = new org.apache.hadoop.conf.Configuration
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    try {
      hdfs.delete(new org.apache.hadoop.fs.Path(outputDir), true)
    } catch {
      case _: Throwable => {}
    }


    logger.info(s"Pre-processing data.")
    // Pre-Processing Step
    fp.PreprocessingMain.main(Array(inputDir, preProcessingOutputDir))
    logger.info(s"End of pre-processing data.")

    logger.info(s"Executing KMeans v$version")

    // KMeans Execution
    if (version == 1) {

      // The result is the list of every K with the respective SSE.
      val output = fp.KMeansClusteringV1.run(preProcessingOutputDir, kMeansOutput, K, I)

      logger.info("OUTPUT: " + output)

      // The purpose of the output is to plot the SSE for each K.

      // Select the K with the min SSE.
      var minSSE: Double = Double.MaxValue
      var minTuple : (Int, Double, String) = null
      for (tuple <- output) {
        val SSE = tuple._2
        if(SSE < minSSE) {
          minSSE = SSE
          minTuple = tuple
        }
      }

      // Applying post-processing to minimum K.
      val minK : String = minTuple._3
      val minKDirOutputDir : String = minTuple._3

      logger.info(s"Best K found for $minK with SSE of $minSSE.")

      logger.info(s"Post-processing data.")
      fp.PostProcessingMain.run(minKDirOutputDir, postProcessingOutputDir, inputDir, topKWords)
      logger.info(s"End of post-processing data.")


    } else {
      fp.KMeansClusteringV2.run(preProcessingOutputDir, kMeansOutput, K, I)
    }

  }
}
