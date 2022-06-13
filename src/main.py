import pyspark.sql.functions as F
from pandas import DataFrame
from pyspark import SparkContext
from pyspark.mllib.clustering import (PowerIterationClustering,
                                      PowerIterationClusteringModel)
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from clustering.greedy import greedy, greedy_heuristic
from constants import *
from homogenity.entropy import generateEntropyColumnHomogenity

# from homogenity import generateEntropyColumnHomogenity, generateColumnCountHomogenity


def create_session():
    # starting Sprak session
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def read_data(spark):
    # Reading the data from the csv
    schema = StructType(
        StructType(
            [StructField(columnName, StringType(), False)
             for columnName in COLUMN_NAMES]
        )
    )

    df = spark.read.csv(DATASET_PATH, header=True, sep='|', schema=schema)\
        .fillna("")\
        .withColumn(ID_COLUMN, F.monotonically_increasing_id())\
        .limit(100)

    # Print some data information.
    print("General data")
    df.printSchema()
    df.summary().show()

    return df


def clusters_homogeneity(df: DataFrame, assignments, homogeneity):
    # Join the clusters with the initial dataframe
    df = df.join(assignments, ID_COLUMN)
    df.show()

    homogeneity_sum = 0

    for clusterId in range(0, CLUSTER_COUNT):
        clusterHomo = homogeneity(
            df.where(df[CLUSTER_COLUMN] == clusterId).drop(ID_COLUMN, CLUSTER_COLUMN))
        print(f"Cluster {clusterId}: {clusterHomo}")
        homogeneity_sum += clusterHomo

    print("Clustered homogenity:")
    print(homogeneity_sum / CLUSTER_COUNT)


def evaluate_partition(homogeneity, partition):
    homogeneity_sum = 0
    for i in range(len(partition)):
        homo = homogeneity(partition[i])
        print(f"Part {i}, homogenity: {homo}")
        homogeneity_sum += homo

    avg = homogeneity_sum/len(partition)
    print(f"Avg per partition:{avg}")


def main():
    spark, sc = create_session()

    df = read_data(spark)

    # train_clusterer(df, sc) # run it only once to train the model and then comment and just load the model for faster experiments

    #assignments = load_clusterer(sc).assignments().toDF().toDF(ID_COLUMN, CLUSTER_COLUMN)

    distinct = df.agg(*(F.countDistinct(F.col(column)).alias(column)
                        for column in df.columns))
    distinct.show()
    N_counts = distinct.collect()[0]

    homogeneity_func = generateEntropyColumnHomogenity(N_counts)  # HOMOGENEITY

    #clusters_homogeneity(df, assignments, homogeneity=homogeneity_func)

    partition = greedy(df, CLUSTER_COUNT, homogeneity_func)

    print("Total homogenity:")
    print(homogeneity_func(df.drop(ID_COLUMN, CLUSTER_COLUMN)))

    evaluate_partition(partition=partition, homogeneity=homogeneity_func)


if __name__ == '__main__':
    main()
