import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from constants import *
from experiments.experiments_PIC import experiemnts_PIC
from experiments.experiments_greedy import experiemnts_greedy
from examples.simple_flow import example1
from experiments.utilities import plot_experiments


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


def main():
    spark, sc = create_session()
    df = read_data(spark, limit=100, separator='|', path= DATASET_PATH)
    example1(df, sc)
    #experiemnts_PIC(df, sc)
    #experiemnts_greedy(df, sc)
    #plot_experiments()
    

if __name__ == '__main__':
    main()    


"""
NOTES: greedy optimized approach is taking ~50 secs to run on the complete dataset. even though for small (like limiting dataset to 100) it is taking ~30 secs,
       advantages are felt when the dataset is large.
       pattern set mining is taking ~40 secs to run on the complete dataset.
       
TODO: improve the clustering approach, infeasible for large data now
          decide if including the pattern set mining idea is good 
          experiments and results
          plots and visualizations of the results          
"""
       