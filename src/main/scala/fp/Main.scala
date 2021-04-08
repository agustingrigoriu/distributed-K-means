package fp

object Main {
  def main(args: Array[String]): Unit = {
    //    Here we do pre processing. execution of cluster version, and post processing.
    val prepInputDir: String = args(0)
    val prepOutputDir: String = args(1)
    val kMeansOutDir: String = args(2)
    val K: String = args(3)
    val iters: String = args(4)
    val version: Int = args(5).toInt
    val postOutputDir: String = args(6)

    fp.PreprocessingMain.main(Array(prepInputDir, prepOutputDir))
    if (version == 1) {
      fp.KMeansClusteringMain.main(Array(prepOutputDir, kMeansOutDir, K, iters))
    } else {
      fp.KMeansClusteringV2Main.main(Array(prepOutputDir, kMeansOutDir, K, iters))
    }
    fp.PostProcessingMain.main(Array(kMeansOutDir, postOutputDir, prepInputDir))
  }
}
