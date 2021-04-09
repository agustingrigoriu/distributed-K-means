package fp

import java.io.File

object Main {
  def main(args: Array[String]): Unit = {

    val inputDir: String = args(0)
    val outputDir: String = args(1)
    val preProcessingOutputDir: String = outputDir + File.separator + "preprocessing"
    val kMeansOutputDir: String = outputDir + File.separator + "k-means"
    val version: Int = args(2).toInt
    val K: String = args(3)
    val I: String = args(4)
    val topKWords: String = args(5)

    // Pre-Processing Step
    fp.PreprocessingMain.main(Array(inputDir, preProcessingOutputDir))

    // KMeans Execution
    if (version == 1) {
      fp.KMeansClusteringMain.main(Array(preProcessingOutputDir, kMeansOutputDir, K, I))
    } else {
      fp.KMeansClusteringV2Main.main(Array(preProcessingOutputDir, kMeansOutputDir, K, I))
    }

    // Post-Processing Step
    //fp.PostProcessingMain.main(Array(kMeansOutDir, postOutputDir, prepInputDir, topKWords))
  }
}
