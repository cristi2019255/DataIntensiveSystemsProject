from pyspark import SparkContext
from clustering.greedy import greedy_partitioning
from clustering.greedy_efficient import greedy_efficient, prepare_windows
from constants import ID_COLUMN
from experiments.utilities import plot_experiment, write_results
from homogenity.entropy import generateEntropyColumnHomogenity
from pyspark.sql import DataFrame


def evaluate_partition(homogeneity, partition=None, k = None):
    homogeneity_sum = 0    
    for i in range(k):
        homogeneity_sum += homogeneity(partition[i])            
          
    avg = homogeneity_sum/k    
    print(f"Avg Homogeneity:  {avg}")  
    return avg

def experiemnts_greedy(df:DataFrame, sc: SparkContext, results_file_path = "results/greedy/results.txt"):
    print("Experiments greedy approach")
    homogeneity_func = generateEntropyColumnHomogenity(df)

    run_times = []
    homogenities = []
    cluster_counts = [k for k in range(2, 20, 2)]    
    
    
    df = df.drop(ID_COLUMN)
    
    windows = prepare_windows(df)
    # Running the experiments
    for k in cluster_counts:
        run_time, partition = greedy_partitioning(df, k) #greedy_efficient(df, k, windows_by_cols=windows)        
        
        run_times.append(run_time)
        #evaluate_partition(df, partition=cluster_assignments, homogeneity=homogeneity_func, clustering=True)
        homogenity  = evaluate_partition(partition=partition, homogeneity=homogeneity_func, k = k)
        homogenities.append(homogenity)        

    write_results(run_times=run_times, homogeneities=homogenities, cluster_counts=cluster_counts, results_file_path=results_file_path)
    plot_experiment(results_file_path)
        
    print("Done!")