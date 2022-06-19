from constants import DATASET_PATH
from experiments.experiments_PIC import experiemnt_PIC, experiment_PIC_scalability
from experiments.experiments_greedy import experiment_greedy, experiment_greedy_scalability
from examples.simple_flow import example1
from experiments.utilities import experiments_scalability, plot_experiments, plot_experiments_scalability
from utilities import create_session, generate_synthetic_data, read_data

def main():
    spark, sc = create_session()
    df = read_data(spark, limit=100, separator='|', path= DATASET_PATH)
    #generate_synthetic_data(sc)
    
    example1(df, sc)
    
    #experiemnts(experiment=experiemnt_PIC, df=df, sc=sc, results_file_path="results/PIC/results.txt")
    #experiemnts(experiment=experiment_greedy, df=df, sc=sc, results_file_path="results/greedy/results.txt")    
    #plot_experiments()
    
    #experiments_scalability(experiment_PIC_scalability, df, sc, results_file_path="results/PIC/results_scalability.txt", k = 6)
    #experiments_scalability(experiment_greedy_scalability, df, sc, results_file_path="results/greedy/results_scalability.txt", k = 6)
    #plot_experiments_scalability()
    
    

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
       