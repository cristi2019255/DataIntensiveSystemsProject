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

def experiemnts_PIC(df:DataFrame, sc: SparkContext, results_file_path = "results/PIC/results.txt"):
    print("Experiments PIC")
    homogeneity_func = generateEntropyColumnHomogenity(df)

    run_times = []
    homogenities = []
    cluster_counts = [k for k in range(2, 20, 2)]    
    
    similarities_matrix = prepare_similarities_matrix(df)
    
    # Running the experiments
    for k in cluster_counts:
        run_time, model = train_clusterer(similarities_matrix, sc, k)
        
        cluster_assignments = model.assignments().toDF().toDF(ID_COLUMN, CLUSTER_COLUMN)
        
        run_times.append(run_time)
        #evaluate_partition(df, partition=cluster_assignments, homogeneity=homogeneity_func, clustering=True)
        homogenity  = evaluate_partition_clustering(df, partition=cluster_assignments, homogeneity=homogeneity_func, k = k)
        homogenities.append(homogenity)
        

    write_results(run_times=run_times, homogeneities=homogenities, cluster_counts=cluster_counts, results_file_path=results_file_path)
    plot_experiment(results_file_path)
        
    print("Done!")

