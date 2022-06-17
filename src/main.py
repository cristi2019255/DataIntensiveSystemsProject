import pyspark.sql.functions as F
from pyspark import SparkContext

from pyspark.sql import SparkSession
from clustering.frequent_sets import pattern_partitioning
from clustering.greedy import greedy_partitioning
from clustering.greedy_efficient import greedy_efficient, greedy_parameterless
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
    df = spark.read.csv(DATASET_PATH, header=True, sep='|')\
        .fillna("")\
        .withColumn(ID_COLUMN, F.monotonically_increasing_id())\
        #.limit(1000)

    # Print some data information.
    print("General data...")
    df.printSchema()
    # df.summary().show()

    return df


def evaluate_partition(df, homogeneity, partition=None, clustering=False):
    homogeneity_sum = 0

    if clustering:
        print("\nClustering approach:\n")
        df = df.join(partition, ID_COLUMN)
        for clusterId in range(0, CLUSTER_COUNT):
            homo = homogeneity(df.where(df[CLUSTER_COLUMN] == clusterId).drop(
                ID_COLUMN, CLUSTER_COLUMN))
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

    # Running the experiments
    cluster_assignments = cluster_partitioning(
        df, sc, CLUSTER_COUNT, train=False)
    
    partition = greedy_parameterless(df.drop(ID_COLUMN))

    # Evaluating approaches
    # Generating the homogeneity function
    homogeneity_func = generateEntropyColumnHomogenity(df)
    #evaluate_partition(df, partition=cluster_assignments, homogeneity=homogeneity_func, clustering=True)
    evaluate_partition(df, partition=partition, homogeneity=homogeneity_func)
    print(
        f"Total homogenity: {str(homogeneity_func(df.drop(ID_COLUMN, CLUSTER_COLUMN)))}")


def example():
    spark, _ = create_session()

    # Create dataframe
    df = spark.createDataFrame([
        ("Shrek", "2001"),
        ("Shrek", "2004"),
        ("Shrek", "2007"),
        ("Bee Movie", "2007"),
        ("Megamind", "2010")
    ], ["Title", "Release"])
    df.show()

    partition1 = df.where(df["Release"] == "2007")
    partition2 = df.where(df["Release"] != "2007")

    # Test homogenity
    homogeneity_func = generateEntropyColumnHomogenity(df)

    print(f"Total homogenity: {str(homogeneity_func(df))}")

    print(f"partition1 homogenity: {str(homogeneity_func(partition1))}")
    print(f"partition2 homogenity: {str(homogeneity_func(partition2))}")

    partitions = greedy_partitioning(df, 2)
    partitions[0].show()
    partitions[1].show()


def pattern_mining_approach():
    # worth to consider, giving much faster results than the greedy approach due to highly optimization it has for spark
    spark, _ = create_session()
    df = read_data(spark)    
    partition = pattern_partitioning(df, CLUSTER_COUNT)
    homogeneity_func = generateEntropyColumnHomogenity(df)
    evaluate_partition(df, partition=partition, homogeneity=homogeneity_func)

if __name__ == '__main__':
    main()


"""
NOTES: greedy optimized approach is taking ~50 secs to run on the complete dataset. even though for small (like limiting dataset to 100) it is taking ~30 secs,
       advantages are felt when the dataset is large.
       pattern set mining is taking ~40 secs to run on the complete dataset.
       
TODO: improve the clustering approach, infeasible for large data now
          decide if including the pattern set mining idea is good 
          experiments and results
          plots and visualizations of the results          
"""
       