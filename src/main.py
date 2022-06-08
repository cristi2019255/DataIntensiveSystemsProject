from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

DATASET_PATH = "data/imdbProfiles.csv"


def main():
    # starting Sprak session
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    sc = spark.sparkContext

    # Reading the data from the csv
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("title", StringType(), True),
        StructField("starring", StringType(), True),
        StructField("writer", StringType(), True),
        StructField("editor", StringType(), True)
    ])

    df = spark.read.csv(DATASET_PATH, header=True,
                        sep='|', schema=schema)
    df.printSchema()

    df.summary().show()


if __name__ == '__main__':
    main()
