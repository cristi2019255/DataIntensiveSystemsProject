import os
import shutil

import pyspark.sql.functions as F
from constants import *
from pandas import DataFrame
from pyspark import SparkContext
from pyspark.mllib.clustering import (PowerIterationClustering,
                                      PowerIterationClusteringModel)


def train_clusterer(df: DataFrame, sc: SparkContext, k:int):
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
        similaritiesRdd, k=k, maxIterations=CLUSTER_ITERATIONS)

    # Save the model
    if os.path.exists(CLUSTER_MODEL_PATH) and os.path.isdir(CLUSTER_MODEL_PATH):
        shutil.rmtree(CLUSTER_MODEL_PATH)

    model.save(sc, CLUSTER_MODEL_PATH)


def cluster_partitioning(df, sc, k:int, train = False):
    if train:
        train_clusterer(df, sc, k) # run it only once to train the model and then comment and just load the model for faster experiments        
    return PowerIterationClusteringModel.load(sc, "results/PICModel").assignments().toDF().toDF(ID_COLUMN, CLUSTER_COLUMN)

