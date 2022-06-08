from math import dist
from operator import add
from pandas import DataFrame
from pyspark.sql.functions import col, countDistinct
from functools import reduce
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.mllib.clustering import PowerIterationClustering, PowerIterationClusteringModel


import pyspark.sql.functions as F
import pyspark.sql.types as T


DATASET_PATH = "data/imdbProfiles.csv"
COLUMN_NAMES = ["id", "title", "starring", "writer", "editor"]
CLUSTER_COUNT = 3


def ColumnCountHomogenity(cluster: DataFrame):
    # Count distinct values in columns
    distinct = cluster.agg(*(countDistinct(col(column)).alias(column)
                             for column in cluster.columns))
    distinct.show()
    counts = distinct.collect()[0]
    homogenity = 1 - sum(counts) / (len(counts) * cluster.count())
    return homogenity


def main():
    # starting Sprak session
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    sc = spark.sparkContext

    # Reading the data from the csv
    schema = StructType(
        StructType(
            [StructField(columnName, StringType(), True)
             for columnName in COLUMN_NAMES]
        )
    )

    df = spark.read.csv(DATASET_PATH, header=True,
                        sep='|', schema=schema).limit(100)

    # Replace null with empty string
    df = df.fillna("")

    # Print some data information.
    print("General data")
    df.printSchema()
    df.summary().show()

    # Train the model.
    print("Training the model")
    joined = df.crossJoin(
        reduce(lambda df, columnName: df.withColumnRenamed(columnName, f"{columnName}_2"), COLUMN_NAMES, df))
    joined.show()
    levenshteined = reduce(lambda df, columnName: df.withColumn(
        f"{columnName}_l", F.levenshtein(F.col(columnName), F.col(f"{columnName}_2"))), COLUMN_NAMES, joined)
    levenshteined.show()
    levenshteinedSummed = levenshteined.withColumn(
        "levenshtein", sum([F.col(f"{columnName}_l") for columnName in COLUMN_NAMES]))
    levenshteinedSummed.show()
    similarities = levenshteinedSummed.select(
        ["id", "id_2", "levenshtein"])
    similarities.show()

    similaritiesRdd = similarities.rdd.map(
        lambda row: [int(row["id"]), int(row["id_2"]), float(row["levenshtein"])])

    # Clustering
    model = PowerIterationClustering.train(
        similaritiesRdd, k=CLUSTER_COUNT, maxIterations=10)
    assignments = model.assignments().toDF()

    # Join the clusters with the initial dataframe
    df = df.join(assignments, "id")
    df.show()

    homogenity = 0
    for clusterId in range(0, CLUSTER_COUNT):
        clusterDf = df.where(df["cluster"] == clusterId)
        clusterHomo = ColumnCountHomogenity(clusterDf.drop("cluster"))
        print(f"Cluster {clusterId}: {clusterHomo}")
        homogenity += clusterHomo

    print("Clustered homogenity:")
    print(homogenity/CLUSTER_COUNT)

    print("Total homogenity:")
    print(ColumnCountHomogenity(df.drop("cluster")))

    # df.groupBy("cluster").
    # .count().show()
    # Save and load model
    # model.save(
    #     sc, "target/org/apache/spark/PythonPowerIterationClusteringExample/PICModel")
    # sameModel = PowerIterationClusteringModel\
    #     .load(sc, "target/org/apache/spark/PythonPowerIterationClusteringExample/PICModel")


if __name__ == '__main__':
    main()
