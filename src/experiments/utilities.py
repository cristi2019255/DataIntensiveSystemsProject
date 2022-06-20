import os
from matplotlib import pyplot as plt
from datetime import timedelta
from pyspark import SparkContext
from pyspark.sql import DataFrame
from homogenity.entropy import generateEntropyColumnHomogenity
from pyspark.sql import SparkSession

from utilities import create_session, read_data

def plot_experiment(results_file_path = "results/PIC/results.txt"):
    results_dir = "/".join([x for x in results_file_path.split("/")[:-2]])
    
    with open(file=results_file_path, mode="r") as f:
        lines = f.readlines()        
        cluster_counts = [int(k) for k in lines[3].split("\n")[0].split(" ")]
        run_times = [float(r.split(":")[-2])*60 + float(r.split(":")[-1]) for r in lines[6].split("\n")[0].split(" ")]
        homogenity = [round(float(h),3) for h in lines[9].split("\n")[0].split(" ")]
        f.close()
    
    experiment = results_file_path.split("/")[-2]    
    
    plt.title("Run time vs. Cluster count")
    plt.xlabel("Number of clusters")
    plt.ylabel("Run timne (sec)")
    plt.plot(cluster_counts, run_times, label="Run time")
    plt.savefig(f"{results_dir}/{experiment}/run_time_{experiment}.png")
    plt.clf()
    
    plt.title("Homogeneity vs. Cluster count")
    plt.xlabel("Number of clusters")
    plt.ylabel("Homogeneity")
    plt.plot(cluster_counts, homogenity, label="Homogenity")
    plt.savefig(f"{results_dir}/{experiment}/homogenity_{experiment}.png")    
    plt.clf()  

    
def write_results(results, results_names, results_file_path = "results/results_PIC.txt"):        
    with open(results_file_path, "w") as f:
        for (result, name) in zip(results, results_names):            
            f.write(f"\n\n{name}\n")
            f.write(" ".join([str(k) for k in result]))        
        f.close()

def plot_experiments(experiments_paths = ["results/PIC/results.txt", "results/greedy/results.txt"]):
    cluster_counts_experiments = {}    
    run_times_experiments = {}    
    homogenities_experiments = {}
    
    results_dir = "/".join([x for x in experiments_paths[0].split("/")[:-2]])
    
    for experiment_path in experiments_paths:        
        with open(file=experiment_path, mode="r") as f:
            lines = f.readlines()        
            cluster_counts = [int(k) for k in lines[3].split("\n")[0].split(" ")]
            run_times = [float(r.split(":")[-2])*60 + float(r.split(":")[-1]) for r in lines[6].split("\n")[0].split(" ")]
            homogenity = [round(float(h),3) for h in lines[9].split("\n")[0].split(" ")]
            f.close()        
        
        cluster_counts_experiments[experiment_path] = cluster_counts
        run_times_experiments[experiment_path] = run_times
        homogenities_experiments[experiment_path] = homogenity
        
    plt.title("Run time vs. Cluster count")
    plt.xlabel("Number of clusters")
    plt.ylabel("Run timne (sec)")    
    for experiment_path in experiments_paths:
        plt.plot(cluster_counts_experiments[experiment_path], run_times_experiments[experiment_path], label=experiment_path.split("/")[-2])    
    plt.legend()
    plt.savefig(f"{results_dir}/run_time_comparison.png")
    plt.clf()
    
    plt.title("Homogeneity vs. Cluster count")
    plt.xlabel("Number of clusters")
    plt.ylabel("Homogeneity")
    for experiment_path in experiments_paths:
        plt.plot(cluster_counts_experiments[experiment_path], homogenities_experiments[experiment_path], label=experiment_path.split("/")[-2])    
    plt.legend()
    plt.savefig(f"{results_dir}/homogenity_comparison.png")    
    plt.clf()
    
def experiments_scalability(experiment, df:DataFrame, sc: SparkContext, spark:SparkSession, k = 2):        
    print(f"Experiments for scalability")        
    experiment(df, sc, spark, k)    
    print("Done!")
    
def experiments(experiment, df:DataFrame, sc: SparkContext, results_file_path = "results/greedy/results.txt"):
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    experiment_name = results_file_path.split("/")[-2]
    print(f"Experiments {experiment_name} approach")
    homogeneity_func = generateEntropyColumnHomogenity(df)
    
    run_times = []
    homogenities = []
    cluster_counts = []                               

    run_time, homogenity = experiment(df, sc, homogeneity_func, 2)
                
    # Running the experiments        
    for k in range(2, 22, 2):
        
        run_time_total = timedelta(seconds=0)        
        # Running 3 times for each k in order to get medium run time
        for _ in range(3):                    
            run_time, homogenity = experiment(df, sc, homogeneity_func, k)
            run_time_total += run_time            
        
        run_time = run_time_total / 3        
        
        run_times.append(run_time)                
        homogenities.append(homogenity)        
        cluster_counts.append(k)                        

    write_results(results=[cluster_counts, run_times, homogenities], results_names=['Cluster count', 'Run time', 'Homogeneity'], results_file_path=results_file_path)
    plot_experiment(results_file_path)
        
    print("Done!")