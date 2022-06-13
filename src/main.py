import shutil
import os
from numpy import maximum
from pandas import DataFrame
from pyspark import SparkContext
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.mllib.clustering import PowerIterationClustering, PowerIterationClusteringModel
import pyspark.sql.functions as F

from homogenity import generateEntropyColumnHomogenity, generateColumnCountHomogenity

# Variables
DATASET_PATH = "data/imdbProfiles.csv"
CLUSTER_MODEL_PATH = "results/PICModel"
COLUMN_NAMES = ["id", "title", "starring", "writer", "editor"]
CLUSTER_COUNT = 6
CLUSTER_ITERATIONS = 10

# Constants
ID_COLUMN = "uniqueId"
LEVENSHTEIN_COLUMN = "levenshtein"
CLUSTER_COLUMN = "cluster"

LEFT_COLUMN = "_1"
RIGHT_COLUMN = "_2"
L_COLUMN = "_l"


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

def train_clusterer(df:DataFrame,sc:SparkContext):
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
            
    # Save the model    
    if os.path.exists(CLUSTER_MODEL_PATH) and os.path.isdir(CLUSTER_MODEL_PATH):
        shutil.rmtree(CLUSTER_MODEL_PATH)
        
    model.save(sc, CLUSTER_MODEL_PATH)
    
def load_clusterer(sc):
    return PowerIterationClusteringModel.load(sc, "results/PICModel")            

def clusters_homogeneity(df:DataFrame, assignments, homogeneity):
    # Join the clusters with the initial dataframe
    df = df.join(assignments, ID_COLUMN)
    df.show()

    homogeneity_sum = 0
    
    for clusterId in range(0, CLUSTER_COUNT):        
        clusterHomo = homogeneity(df.where(df[CLUSTER_COLUMN] == clusterId).drop(ID_COLUMN, CLUSTER_COLUMN))
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


def greedy_heuristic(df:DataFrame, k: int, homogenity):    
    df = df.drop(ID_COLUMN)
    df_left = df.select('*')
    
    columns_counts = [df.groupby(c).count().sort(F.col("count"), ascending=False).collect() for c in df.columns]
    
    parts = []
    for _ in range(k):
        maximum = max([max(x, key= lambda y:y[1]) for x in columns_counts], key=lambda x:x[1])
        
        max_attr_val = maximum[0]
        col = list(maximum.asDict().keys())[0]
        
        
        parts.append(df.where(df[col] == max_attr_val))            
        df_left = df_left.where(df[col] != max_attr_val)         
        
        columns_counts = [df_left.groupby(c).count().sort(F.col("count"), ascending=False).collect() for c in df.columns]
        
        parts[-1].show()
    
    
    while df_left.count() != 0:
        # until partition is not complete for each part add the subparts the increases homogenity the most
        
        # getting the maximum homogen subpart over column
        maximum = max([max(x, key= lambda y:y[1]) for x in columns_counts], key=lambda x:x[1])        
        max_attr_val = maximum[0]
        col = list(maximum.asDict().keys())[0]

        subpart = df.where(df[col] == max_attr_val)
        # searching the part that increases homogenity the most with the subpart
        best_part = 0
        homogenity_best = 0
        for i in range(k):
            homo = homogenity(parts[i].union(subpart))
            if homo > homogenity_best:
                homogenity_best = homo
                best_part = i 
                
        # merging subpart with the part
        parts[best_part] = parts[best_part].union(subpart)
        parts[best_part].show()
        # removing the subpart from the dataframe
        df_left = df_left.where(df[col] != max_attr_val)        
        # removing the subpart from column_counts        
        columns_counts = [df_left.groupby(c).count().sort(F.col("count"), ascending=False).collect() for c in df.columns]
        
        print(df_left.count())
    
    return parts
        
    
def main():
    spark, sc = create_session()

    df = read_data(spark)
    
    #train_clusterer(df, sc) # run it only once to train the model and then comment and just load the model for faster experiments
    
    #assignments = load_clusterer(sc).assignments().toDF().toDF(ID_COLUMN, CLUSTER_COLUMN)        

    distinct = df.agg(*(F.countDistinct(F.col(column)).alias(column)
                             for column in df.columns))
    distinct.show()
    N_counts = distinct.collect()[0]

    homogeneity_func = generateEntropyColumnHomogenity(N_counts) #HOMOGENEITY

    #clusters_homogeneity(df, assignments, homogeneity=homogeneity_func)

    partition = greedy_heuristic(df.drop('editor'), CLUSTER_COUNT, homogeneity_func)

    print("Total homogenity:")    
    print(homogeneity_func(df.drop(ID_COLUMN, CLUSTER_COLUMN)))    

    evaluate_partition(partition=partition, homogeneity=homogeneity_func)

if __name__ == '__main__':
    main()
