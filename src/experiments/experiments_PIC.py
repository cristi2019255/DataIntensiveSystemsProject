from pyspark import SparkContext
from clustering.pic import prepare_similarities_matrix, train_clusterer
from constants import CLUSTER_COLUMN, ID_COLUMN
from experiments.utilities import plot_experiment, write_results
from homogenity.entropy import generateEntropyColumnHomogenity
from pyspark.sql import DataFrame


def evaluate_partition_clustering(df, homogeneity, partition=None, k = None):
    homogeneity_sum = 0    
    df = df.join(partition, ID_COLUMN)
    for clusterId in range(k):
        homogeneity_sum += homogeneity(df.where(df[CLUSTER_COLUMN] == clusterId).drop(ID_COLUMN, CLUSTER_COLUMN))                             
    avg = homogeneity_sum/k    
    print(f"Avg Homogeneity:  {avg}")  
    return avg


def experiemnt_PIC(df:DataFrame, sc:SparkContext, homogeneity_func, k = 2):    
    
    run_time_preparations, similarities_matrix = prepare_similarities_matrix(df)    
    run_time_train, model = train_clusterer(similarities_matrix, sc, k)
        
    cluster_assignments = model.assignments().toDF().toDF(ID_COLUMN, CLUSTER_COLUMN)            
    homogenity  = evaluate_partition_clustering(df, partition=cluster_assignments, homogeneity=homogeneity_func, k = k)
    
    return run_time_preparations + run_time_train , homogenity 
    
def experiment_PIC_scalability(df:DataFrame,  limit_data_size: int, sc: SparkContext, k = 2):
    run_time_preparations, similarities_matrix = prepare_similarities_matrix(df.limit(limit_data_size)) 
    run_time_train, _ = train_clusterer(similarities_matrix, sc, k)        
    return run_time_preparations + run_time_train        