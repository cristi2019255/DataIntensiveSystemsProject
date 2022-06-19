from matplotlib import pyplot as plt
from datetime import timedelta
from pyspark import SparkContext
from pyspark.sql import DataFrame

from homogenity.entropy import generateEntropyColumnHomogenity

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


def plot_experiment_scalability(results_file_path = "results/greedy/results_scalability.txt"):
    results_dir = "/".join([x for x in results_file_path.split("/")[:-2]])
    
    with open(file=results_file_path, mode="r") as f:
        lines = f.readlines()        
        data_sizes = [int(k) for k in lines[3].split("\n")[0].split(" ")]
        run_times = [float(r.split(":")[-2])*60 + float(r.split(":")[-1]) for r in lines[6].split("\n")[0].split(" ")]        
        f.close()
    
    experiment = results_file_path.split("/")[-2]    
    
    plt.title("Run time vs. Data size")
    plt.xlabel("Data size")
    plt.ylabel("Run timne (sec)")
    plt.plot(data_sizes, run_times, label="Run time")
    plt.savefig(f"{results_dir}/{experiment}/run_time_{experiment}_scalability.png")
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

def plot_experiments_scalability(experiments_paths = ["results/PIC/results_scalability.txt", "results/greedy/results_scalability.txt"]):
    data_sizes_experiments = {}    
    run_times_experiments = {}        
    results_dir = "/".join([x for x in experiments_paths[0].split("/")[:-2]])
    
    for experiment_path in experiments_paths:        
        with open(file=experiment_path, mode="r") as f:
            lines = f.readlines()        
            data_sizes = [int(s) for s in lines[3].split("\n")[0].split(" ")]
            run_times = [float(r.split(":")[-2])*60 + float(r.split(":")[-1]) for r in lines[6].split("\n")[0].split(" ")]            
            f.close()        
        
        data_sizes_experiments[experiment_path] = data_sizes
        run_times_experiments[experiment_path] = run_times        
        
    plt.title("Run time vs. Data size")
    plt.xlabel("Data size")
    plt.ylabel("Run timne (sec)")    
    for experiment_path in experiments_paths:
        plt.plot(data_sizes_experiments[experiment_path], run_times_experiments[experiment_path], label=experiment_path.split("/")[-2])    
    plt.legend()
    plt.savefig(f"{results_dir}/run_time_comparison_scalability.png")
    plt.clf()
    
def experiments_scalability(experiment, df:DataFrame, sc: SparkContext, results_file_path = "results/PIC/results_scalability.txt", k = 2, max_waiting_time = 60):
    experiment_name = results_file_path.split("/")[-2]
    print(f"Experiments {experiment_name} approach for scalability")    

    FRACTION = 0.05
    run_times = []
    data_sizes = []
    
    run_time = timedelta(seconds=0)
    max_waiting_time = timedelta(seconds=max_waiting_time)
    
    total_data_size = df.count()
    fraction = FRACTION
    limit_data_size = int(fraction * total_data_size)    
    
    while run_time <= max_waiting_time and limit_data_size <= total_data_size:
        print(f"Current data size: {limit_data_size}")        
        data_sizes.append(limit_data_size)
        
        run_time = experiment(df, limit_data_size, sc, k)
        run_times.append(run_time)        
        
        fraction += FRACTION
        limit_data_size = int(fraction * total_data_size)
                        
    write_results(results=[data_sizes, run_times], results_names=['Data size', 'Run time'], results_file_path=results_file_path)                          
    plot_experiment_scalability(results_file_path)
    
    print("Done!")
    
def experiemnts(experiment, df:DataFrame, sc: SparkContext, results_file_path = "results/greedy/results.txt"):
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
