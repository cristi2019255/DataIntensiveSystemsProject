from experiments.experiments_PIC import experiment_PIC_scalability, generate_experiment_PIC
from experiments.experiments_greedy import experiment_greedy_scalability,  experiment_greedy
from examples.simple_flow import example1
from experiments.utilities import experiments, experiments_scalability, plot_experiments
from utilities import create_session, read_data

def main():
    spark, sc = create_session()

    dataset_path, separator = "data/dbpediaProfiles.csv", "|"
    #dataset_path, separator = "data/dunkin_stores.csv", ","
    #dataset_path, separator = "data/syntheticFinances.csv", ","
    
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    
    df = read_data(spark, limit=100, separator=separator, path= dataset_path)    
    
    example1(df, sc)    
        
    #experiemnts(experiment=experiment_greedy, df=df, sc=sc, results_file_path=f"results/{dataset_name}/greedy/results.txt")    
    #experiemnts(experiment=generate_experiment_PIC(df), df=df, sc=sc, results_file_path=f"results/{dataset_name}/PIC/results.txt")
    #plot_experiments(experiments_paths=[f"results/{dataset_name}/greedy/results.txt", f"results/{dataset_name}/PIC/results.txt"])
    
    #experiments_scalability(experiment_greedy_scalability, df, sc, spark, k = 6)
    #experiments_scalability(experiment_PIC_scalability, df, sc, spark, k = 6)
    
    # closing spark session
    sc.stop()

if __name__ == '__main__':
    main()    