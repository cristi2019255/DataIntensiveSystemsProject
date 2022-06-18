
from matplotlib import pyplot as plt


def plot_experiment(results_file_path = "results/PIC/results.txt"):
    with open(file=results_file_path, mode="r") as f:
        lines = f.readlines()        
        cluster_counts = [int(k) for k in lines[1].split("\n")[0].split(" ")]
        run_times = [float(r.split(":")[-1]) for r in lines[4].split("\n")[0].split(" ")]
        homogenity = [round(float(h),3) for h in lines[7].split("\n")[0].split(" ")]
        f.close()
    
    experiment = results_file_path.split("/")[-2]    
    
    plt.title("Run time vs. Cluster count")
    plt.xlabel("Number of clusters")
    plt.ylabel("Run timne (sec)")
    plt.plot(cluster_counts, run_times, label="Run time")
    plt.savefig(f"results/{experiment}/run_time_{experiment}.png")
    plt.clf()
    
    plt.title("Homogeneity vs. Cluster count")
    plt.xlabel("Number of clusters")
    plt.ylabel("Homogeneity")
    plt.plot(cluster_counts, homogenity, label="Homogenity")
    plt.savefig(f"results/{experiment}/homogenity_{experiment}.png")    
    
def write_results(run_times, homogeneities, cluster_counts, results_file_path = "results/results_PIC.txt"):
    with open(results_file_path, "w") as f:
        f.write("Cluster count\n")
        f.write(" ".join([str(k) for k in cluster_counts]))
        f.write("\n\nRun time\n")
        f.write(" ".join([str(r) for r in run_times]))
        f.write("\n\nHomogenity\n")
        f.write(" ".join([str(h) for h in homogeneities]))
        f.close()

def plot_experiments(experiments_paths = ["results/PIC/results.txt", "results/greedy/results.txt"]):
    cluster_counts_experiments = {}    
    run_times_experiments = {}    
    homogenities_experiments = {}
    
    for experiment_path in experiments_paths:        
        with open(file=experiment_path, mode="r") as f:
            lines = f.readlines()        
            cluster_counts = [int(k) for k in lines[1].split("\n")[0].split(" ")]
            run_times = [float(r.split(":")[-2])*60 + float(r.split(":")[-1]) for r in lines[4].split("\n")[0].split(" ")]
            homogenity = [round(float(h),3) for h in lines[7].split("\n")[0].split(" ")]
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
    plt.savefig(f"results/run_time_comparison.png")
    plt.clf()
    
    plt.title("Homogeneity vs. Cluster count")
    plt.xlabel("Number of clusters")
    plt.ylabel("Homogeneity")
    for experiment_path in experiments_paths:
        plt.plot(cluster_counts_experiments[experiment_path], homogenities_experiments[experiment_path], label=experiment_path.split("/")[-2])    
    plt.legend()
    plt.savefig(f"results/homogenity_comparison.png")    
    