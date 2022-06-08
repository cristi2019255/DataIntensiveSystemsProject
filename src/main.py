import shutil
import os
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
    
    
    
def main():
    spark, sc = create_session()

    df = read_data(spark)
    
    train_clusterer(df, sc) # run it only once to train the model and then comment and just load the model for faster experiments
    
    assignments = load_clusterer(sc).assignments().toDF().toDF(ID_COLUMN, CLUSTER_COLUMN)        

    distinct = df.agg(*(F.countDistinct(F.col(column)).alias(column)
                             for column in df.columns))
    distinct.show()
    N_counts = distinct.collect()[0]

    homogeneity_func = generateEntropyColumnHomogenity(N_counts) #HOMOGENEITY
    

    print("Total homogenity:")    
    print(homogeneity_func(df.drop(ID_COLUMN, CLUSTER_COLUMN)))    


    clusters_homogeneity(df, assignments, homogeneity=homogeneity_func)

if __name__ == '__main__':
    main()
