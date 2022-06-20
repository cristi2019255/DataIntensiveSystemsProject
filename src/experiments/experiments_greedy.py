from pyspark import SparkContext
from clustering.greedy import greedy_partitioning
from clustering.greedy_efficient import greedy_efficient, prepare_windows
from constants import ID_COLUMN
from experiments.utilities import plot_experiment, write_results
from homogenity.entropy import generateEntropyColumnHomogenity
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

def evaluate_partition(homogeneity, partition=None, k = None):
    homogeneity_sum = 0    
    for i in range(k):
        homogeneity_sum += homogeneity(partition[i])                      
    avg = homogeneity_sum/k    
    print(f"Avg Homogeneity:  {avg}")  
    return avg

def experiment_greedy(df:DataFrame, sc:SparkContext, homogeneity_func, k = 2):
    df = df.drop(ID_COLUMN)        
    windows = prepare_windows(df)
    run_time, partition = greedy_efficient(df, k, windows_by_cols=windows) # greedy_partitioning(df, k)                   
    homogenity  = evaluate_partition(partition=partition, homogeneity=homogeneity_func, k = k)
    return run_time, homogenity            
    
def experiment_greedy_scalability(df:DataFrame, sc:SparkContext, spark: SparkSession, k = 2):
    df = df.drop(ID_COLUMN)        
    windows = prepare_windows(df)
    run_time, _ = greedy_efficient(df, k, windows_by_cols=windows)        
    
    # clearing the cache for fair further experiments
    spark.catalog.clearCache()      
      
    return run_time