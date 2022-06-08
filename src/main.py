from operator import add
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
                        sep='|', schema=schema)

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
        lambda id1, id2, levenshtein: [int(id1), int(id2), float(levenshtein)])

    # Clustering
    model = PowerIterationClustering.train(
        similaritiesRdd, k=5, maxIterations=10)

    # model.assignments().foreach(lambda x: print(str(x.id) + " -> " + str(x.cluster)))

    # Save and load model
    # model.save(
    #     sc, "target/org/apache/spark/PythonPowerIterationClusteringExample/PICModel")
    # sameModel = PowerIterationClusteringModel\
    #     .load(sc, "target/org/apache/spark/PythonPowerIterationClusteringExample/PICModel")


if __name__ == '__main__':
    main()
