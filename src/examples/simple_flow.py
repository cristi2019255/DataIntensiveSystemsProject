from pyspark.sql import DataFrame
from clustering.frequent_sets import pattern_partitioning
from clustering.greedy import greedy_partitioning
from clustering.pic import cluster_partitioning
from constants import CLUSTER_COLUMN, ID_COLUMN
from homogenity.entropy import generateEntropyColumnHomogenity

CLUSTER_COUNT = 10

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


def example1(df, sc):    
    # Running the experiments
    _, cluster_assignments = cluster_partitioning(df, sc, CLUSTER_COUNT, train=True)    
    _, partition = greedy_partitioning(df.drop(ID_COLUMN), CLUSTER_COUNT)

    # Evaluating approaches
    # Generating the homogeneity function
    homogeneity_func = generateEntropyColumnHomogenity(df)
    evaluate_partition(df, partition=cluster_assignments, homogeneity=homogeneity_func, clustering=True)
    evaluate_partition(df, partition=partition, homogeneity=homogeneity_func)
    print(f"Total homogenity: {str(homogeneity_func(df.drop(ID_COLUMN, CLUSTER_COLUMN)))}")


def example2(spark):    
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


def pattern_mining_approach(df: DataFrame):
    # worth to consider, giving much faster results than the greedy approach due to highly optimization it has for spark    
    partition = pattern_partitioning(df, CLUSTER_COUNT)
    homogeneity_func = generateEntropyColumnHomogenity(df)
    evaluate_partition(df, partition=partition, homogeneity=homogeneity_func)
