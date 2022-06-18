from datetime import datetime
import os
import shutil

import pyspark.sql.functions as F
from constants import *
from pyspark.sql import DataFrame
from pyspark import RDD, SparkContext
from pyspark.mllib.clustering import (PowerIterationClustering,
                                      PowerIterationClusteringModel)


def prepare_similarities_matrix(df:DataFrame):
    COLUMN_NAMES = df.columns[:-1]
    joined = df.toDF(*[columnName + LEFT_COLUMN for columnName in COLUMN_NAMES], ID_COLUMN + LEFT_COLUMN).crossJoin(
        df.toDF(*[f"{columnName}{RIGHT_COLUMN}" for columnName in COLUMN_NAMES], ID_COLUMN + RIGHT_COLUMN)).where(f"{ID_COLUMN + LEFT_COLUMN}<{ID_COLUMN + RIGHT_COLUMN}")
    #joined.show()

    levenshteined = joined.select("*", *[F.levenshtein(F.col(columnName + LEFT_COLUMN), F.col(
        columnName + RIGHT_COLUMN)).alias(columnName + L_COLUMN) for columnName in COLUMN_NAMES])
    #levenshteined.show()

    levenshteinedSummed = levenshteined.withColumn(
        LEVENSHTEIN_COLUMN, sum([F.col(columnName + L_COLUMN) for columnName in COLUMN_NAMES]))
    #levenshteinedSummed.show()

    similarities = levenshteinedSummed.select([ID_COLUMN + LEFT_COLUMN, ID_COLUMN + RIGHT_COLUMN, LEVENSHTEIN_COLUMN])
    similarities.show()

    similaritiesRdd = similarities.rdd.map(
        lambda row: [int(row[ID_COLUMN + LEFT_COLUMN]), int(row[ID_COLUMN + RIGHT_COLUMN]), float(row[LEVENSHTEIN_COLUMN])]).cache()

    return similaritiesRdd

def train_clusterer(similarities: RDD, sc: SparkContext, k:int, save_model = False):
    # Train the model.
    print(f"Training the model for {k} clusters...")
    start_time = datetime.now()
    
    # Clustering
    model = PowerIterationClustering.train(similarities, k=k, maxIterations=CLUSTER_ITERATIONS)

    # Save the model
    if save_model:
        if os.path.exists(CLUSTER_MODEL_PATH) and os.path.isdir(CLUSTER_MODEL_PATH):
            shutil.rmtree(CLUSTER_MODEL_PATH)

        model.save(sc, CLUSTER_MODEL_PATH)
    
    end_time = datetime.now()
    run_time = end_time - start_time
    print(f"Training time: {run_time}")
    return run_time, model

def cluster_partitioning(df, sc, k:int, train = False):
    similarities_matrix = prepare_similarities_matrix(df)
    if train:
        run_time, model = train_clusterer(similarities_matrix, sc, k, save_model=True) # run it only once to train the model and then comment and just load the model for faster experiments        
    else: 
        model = PowerIterationClusteringModel.load(sc, "results/PICModel")
        run_time = -1
    return run_time, model.assignments().toDF().toDF(ID_COLUMN, CLUSTER_COLUMN)

