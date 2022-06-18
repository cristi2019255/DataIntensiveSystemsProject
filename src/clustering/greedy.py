from datetime import datetime
import math
import pyspark.sql.functions as F
from constants import *
from pyspark.sql import DataFrame

def greedy_partitioning(df: DataFrame, k: int):
    print("\nGreedy partitioning...")
    start_time = datetime.now()

    cnf = {}
    clusters = [(df.drop(ID_COLUMN).cache(), df.count(), cnf)]        
            
    # need to split k-1 times.
    for _ in range(k-1):

        #  initialize max to None
        max_col_count = max_cluster = max_col_name = max_col_val = max_product = None

        # Loop over all clusters
        for (cluster, cluster_count, cnf) in clusters:                                                                                            
            # Loop over all columns in the cluster
            for col_name in cluster.columns:
                # max column count
                col_val, in_count = cluster.groupBy(col_name).count().rdd.max(lambda x: x[1])                
                                
                product = in_count * math.sqrt(cluster_count - in_count)

                # if the current column product is higher than the max, set it as the max.
                if (max_product == None or product >= max_product):
                    max_col_count = in_count
                    max_cluster = (cluster, cluster_count, cnf)
                    max_col_name = col_name
                    max_col_val = col_val
                    max_product = product

        #print("col_name: ", max_col_name, " col_val: \"",
        #       max_col_val, "\"", " product: ", max_product)
        
        inPart = max_cluster[0].where(F.col(max_col_name) == max_col_val).cache()            
        outPart = max_cluster[0].where(F.col(max_col_name) != max_col_val).cache()
                
        # Remove the old splitted cluster
        clusters.remove(max_cluster)
        
        # Add the new splitted clusters
        old_cnf = cnf.copy()
        cnf[max_col_name] = max_col_val
        clusters.append((inPart, max_col_count, cnf))
        clusters.append((outPart, max_cluster[1] - max_col_count, old_cnf))
    
    end_time = datetime.now()
    run_time = end_time - start_time
    
    [print(f"cnf: {c[2]}, \t count:{(c[0].count())}") for c in clusters]    
    
    print(f"\nTime taken: {run_time}")
    return run_time, list(map(lambda x: x[0], clusters))
