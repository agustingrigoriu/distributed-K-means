### Distributed K-Means

Our project goal is to apply K-Means Clustering to a collection of tweets from U.S.politicians.  We have two versions of the algorithm, one that determines the clusters in a distributed way, while the other clusters different K values at the same time.  We decide upon the optimal K value by looking at the SSE for each run.  
# V1 

Version 1 of our K-Means algorithm runs by dividing the input (document collection) in chunks and distributing them among all machines available. Centroids are copied to each machine (not broadcasted). Therefore, each node will have all the information needed to label a document according to the closest centroid. The goal of this task is to implement this idea using Spark.

# V2

This task involves running K-Means concurrently for different K values. As we will describe on the following sections, this is not very natural to do in Spark. We will be explaining our different attempts and our final solution.

Installation
------------
These components are installed:
- JDK 1.8
- Scala 2.11.12
- Hadoop 2.9.1
- Spark 2.3.1 (without bundled Hadoop)
- Maven
- AWS CLI (for EMR execution)
