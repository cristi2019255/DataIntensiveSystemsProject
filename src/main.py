import pyspark.sql.functions as F
from pyspark import SparkContext

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from clustering.greedy import greedy_partitioning
from clustering.pic import cluster_partitioning
from constants import *
from homogenity.columnCount import generateColumnCountHomogenity
from homogenity.entropy import generateEntropyColumnHomogenity

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
    #print("General data")
    #df.printSchema()
    #df.summary().show()

    return df


def evaluate_partition(df, homogeneity, partition = None, clustering = False):
    homogeneity_sum = 0
    
    if clustering:        
        print("\nClustering approach:\n")
        df = df.join(partition, ID_COLUMN)
        for clusterId in range(0, CLUSTER_COUNT):
            homo = homogeneity(df.where(df[CLUSTER_COLUMN] == clusterId).drop(ID_COLUMN, CLUSTER_COLUMN))
            print(f"Cluster {clusterId}, homogenity: {homo}")
            homogeneity_sum += homo        
    else:
        print("\nGreedy partitioning approach:\n")
        for i in range(len(partition)):
            homo = homogeneity(partition[i].drop(ID_COLUMN))
            print(f"Cluster {i}, homogenity: {homo}")
            homogeneity_sum += homo
    
    avg = homogeneity_sum/CLUSTER_COUNT
    print(f"Avg per partition:{avg}\n\n")


def main():
    spark, sc = create_session()
    df = read_data(spark)
    
    #Running the experiments
    cluster_assignments = cluster_partitioning(df, sc, CLUSTER_COUNT)
    partition = greedy_partitioning(df.drop(ID_COLUMN), CLUSTER_COUNT)

    # Evaluating approaches
    homogeneity_func = generateColumnCountHomogenity(df)  # Generating the homogeneity function
    evaluate_partition(df, partition = cluster_assignments, homogeneity=homogeneity_func, clustering = True)
    evaluate_partition(df, partition = partition, homogeneity=homogeneity_func)    
    print("\nTotal homogenity: " + str(homogeneity_func(df.drop(ID_COLUMN, CLUSTER_COLUMN))))


if __name__ == '__main__':
    main()
