from experiments.experiments_PIC import experiment_PIC_scalability, generate_experiment_PIC
from experiments.experiments_greedy import experiment_greedy_scalability,  experiment_greedy
from examples.simple_flow import example1
from experiments.utilities import experiemnts, experiments_scalability, plot_experiments, plot_experiments_scalability
from utilities import create_session, read_data

def main():
    spark, sc = create_session()
    
    dataset_path, separator = "data/dbpediaProfiles.csv", "|"
    #dataset_path, separator = "data/dunkin_stores.csv", ","
    #dataset_path, separator = "data/syntheticFinances.csv", ","
    
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    
    df = read_data(spark, limit=None, separator=separator, path= dataset_path)    
    
    #example1(df, sc)
        
    #experiemnts(experiment=experiment_greedy, df=df, sc=sc, results_file_path=f"results/{dataset_name}/greedy/results.txt")    
    #experiemnts(experiment=generate_experiment_PIC(df), df=df, sc=sc, results_file_path=f"results/{dataset_name}/PIC/results.txt")
    #plot_experiments(experiments_paths=[f"results/{dataset_name}/greedy/results.txt", f"results/{dataset_name}/PIC/results.txt"])
    
    experiments_scalability(experiment_greedy_scalability, df, sc, results_file_path=f"results/{dataset_name}/greedy/results_scalability.txt", k = 6)
    #experiments_scalability(experiment_PIC_scalability, df, sc, results_file_path=f"results/{dataset_name}/PIC/results_scalability.txt", k = 6)    
    #plot_experiments_scalability(experiments_paths=[f"results/{dataset_name}/PIC/results_scalability.txt", f"results/{dataset_name}/greedy/results_scalability.txt"])


if __name__ == '__main__':
    main()    


"""
NOTES: advantages for greedy approach are felt when the dataset is large.
       clustering approach, infeasible for large data > 100
       for 1000 data points, the PIC algorithm is taking more than 10 mins to run. // stopped after 10 mins // N(N-1)/2 = 1 million for N = 1000 
       thus, the PIC algorithm is not scalable.
       pattern set mining is taking ~40 secs to run on the complete dataset.
       
TODO:     decide if including the pattern set mining idea is good 
          experiments and results
          plots and visualizations of the results          
"""
       