
from pyspark import SparkContext
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from constants import DATASET_PATH, ID_COLUMN

def create_session():
    # starting Sprak session
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def read_data(spark, limit=None, separator = ',', path=DATASET_PATH):
    # Reading the data from the csv
    df = spark.read.csv(path, header=True, sep= separator)\
        .fillna("")\
        .withColumn(ID_COLUMN, F.monotonically_increasing_id())        
        
    if limit is not None:
        df = df.limit(limit)
    
    # Print some data information.
    print("General data...")
    df.printSchema()    
    return df

def generate_synthetic_data(sc:SparkContext, data_path = "data/synthetic_data.csv", size = 100, cols = 3):
    # Generating synthetic data
    pass    