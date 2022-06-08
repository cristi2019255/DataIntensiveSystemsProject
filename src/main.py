from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
import matplotlib.pyplot as plt

DATASET_PATH = "data/imdbProfiles.csv"


def main():    
    # starting Sprak session
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    sc = spark.sparkContext

    # Reading the data from the csv
    df = spark.read.csv(DATASET_PATH, header=True,inferSchema=True)
    df.printSchema()


if __name__ == '__main__':
    main()