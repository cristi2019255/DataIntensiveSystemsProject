
import os
from pyspark import SparkContext
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from constants import DATASET_PATH, ID_COLUMN
import matplotlib.pyplot as plt

def create_session():
    # starting Sprak session
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
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

def analyse_data(df:DataFrame, dataset_name="data"):    
    results_path = os.path.join("results",dataset_name, "distribution")
    os.makedirs(results_path, exist_ok=True)
    df = df.drop(ID_COLUMN)
    
    for c in df.columns:
        dfp = df.select(c).groupBy(c).count().orderBy("count").toPandas()    
    
        plot = dfp.plot(kind="bar", x=c, y="count", figsize=(10, 7), alpha=0.5, color="blue")        
        plot.set_xlabel(f"Column {c} labels")
        plot.set_ylabel("Number of rows")
        plot.set_title(f"Data distribution over column {c}")
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
        plt.savefig(f"{results_path}/{c}.png")
        plt.cla()
