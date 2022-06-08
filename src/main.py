from math import dist
from operator import add
from pandas import DataFrame
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

# Variables
DATASET_PATH = "data/imdbProfiles.csv"
COLUMN_NAMES = ["id", "title", "starring", "writer", "editor"]
CLUSTER_COUNT = 3
CLUSTER_ITERATIONS = 10

# Constants
ID_COLUMN = "uniqueId"
LEVENSHTEIN_COLUMN = "levenshtein"
CLUSTER_COLUMN = "cluster"

LEFT_COLUMN = "_1"
RIGHT_COLUMN = "_2"
L_COLUMN = "_l"


def ColumnCountHomogenity(cluster: DataFrame):
    # Count distinct values in columns
    distinct = cluster.agg(*(F.countDistinct(F.col(column)).alias(column)
                             for column in cluster.columns))
    distinct.show()
    counts = distinct.collect()[0]
    homogenity = 1 - sum(counts) / (len(counts) * cluster.count())
    return homogenity


def main():
    # starting Sprak session
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

    # Reading the data from the csv
    schema = StructType(
        StructType(
            [StructField(columnName, StringType(), False)
             for columnName in COLUMN_NAMES]
        )
    )
    df = spark.read\
        .csv(DATASET_PATH, header=True, sep='|', schema=schema)\
        .fillna("")\
        .withColumn(ID_COLUMN, F.monotonically_increasing_id())\
        .limit(100)

    # Print some data information.
    print("General data")
    df.printSchema()
    df.summary().show()

    # Train the model.
    print("Training the model")
    joined = df.toDF(*[columnName + LEFT_COLUMN for columnName in COLUMN_NAMES], ID_COLUMN + LEFT_COLUMN).crossJoin(
        df.toDF(*[f"{columnName}{RIGHT_COLUMN}" for columnName in COLUMN_NAMES], ID_COLUMN + RIGHT_COLUMN))
    joined.show()
    levenshteined = joined.select("*", *[F.levenshtein(F.col(columnName + LEFT_COLUMN), F.col(
        columnName + RIGHT_COLUMN)).alias(columnName + L_COLUMN) for columnName in COLUMN_NAMES])
    levenshteined.show()
    levenshteinedSummed = levenshteined.withColumn(
        LEVENSHTEIN_COLUMN, sum([F.col(columnName + L_COLUMN) for columnName in COLUMN_NAMES]))
    levenshteinedSummed.show()
    similarities = levenshteinedSummed.select(
        [ID_COLUMN + LEFT_COLUMN, ID_COLUMN + RIGHT_COLUMN, LEVENSHTEIN_COLUMN])
    similarities.show()

    similaritiesRdd = similarities.rdd.map(
        lambda row: [int(row[ID_COLUMN + LEFT_COLUMN]), int(row[ID_COLUMN + RIGHT_COLUMN]), float(row[LEVENSHTEIN_COLUMN])])

    # Clustering
    model = PowerIterationClustering.train(
        similaritiesRdd, k=CLUSTER_COUNT, maxIterations=CLUSTER_ITERATIONS)
    assignments = model.assignments().toDF().toDF(ID_COLUMN, CLUSTER_COLUMN)

    # Join the clusters with the initial dataframe
    df = df.join(assignments, ID_COLUMN)
    df.show()

    homogenity = 0
    for clusterId in range(0, CLUSTER_COUNT):
        clusterDf = df.where(df[CLUSTER_COLUMN] == clusterId)
        clusterHomo = ColumnCountHomogenity(
            clusterDf.drop(ID_COLUMN, CLUSTER_COLUMN))
        print(f"Cluster {clusterId}: {clusterHomo}")
        homogenity += clusterHomo

    print("Clustered homogenity:")
    print(homogenity/CLUSTER_COUNT)

    print("Total homogenity:")
    print(ColumnCountHomogenity(df.drop(ID_COLUMN, CLUSTER_COLUMN)))

    # df.groupBy("cluster").
    # .count().show()
    # Save and load model
    # model.save(
    #     sc, "target/org/apache/spark/PythonPowerIterationClusteringExample/PICModel")
    # sameModel = PowerIterationClusteringModel\
    #     .load(sc, "target/org/apache/spark/PythonPowerIterationClusteringExample/PICModel")


if __name__ == '__main__':
    main()
