from datetime import datetime
import math
import pyspark.sql.functions as F
from constants import *
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

def greedy_efficient(df: DataFrame, k: int):
    print("\nGreedy partitioning...")
    start_time = datetime.now()
    
    df = df.drop(ID_COLUMN)
    cnf = {}
    clusters = [(df.cache(), df.count(), cnf)]        
            
    # partitioning the dataset on each columns
    windows_by_cols = {}
    for c in df.columns:
        windows_by_cols[c] = Window().partitionBy(c)
                      
            
    # need to split k-1 times.
    for _ in range(k-1):

        #  initialize max to None
        max_col_count = max_cluster = max_col_name = max_col_val = max_product = None
        
        # Loop over all clusters
        for cluster_info in clusters:     
            cluster, cluster_count, cnf = cluster_info                                                                                                               
            # skip the clusters that can not improve the split 
            if max_col_count != None and cluster_count < max_col_count:
                continue
            
            # Loop over all columns in the cluster            
            for col_name in cluster.columns:                
                col_val, in_count = cluster.select(col_name).withColumn("count", F.count(F.col(col_name)).over(windows_by_cols[col_name])).orderBy("count", ascending=False).first()
                                
                product = in_count * math.sqrt(cluster_count - in_count)

                # if the current column product is higher than the max, set it as the max.
                if (max_product == None or product >= max_product):
                    max_col_count = in_count
                    max_cluster = cluster_info
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

    [print(f"cnf: {c[2]}, \t count:{(c[0].count())}") for c in clusters]
    
    end_time = datetime.now()
    print(f"\nTime taken: {end_time - start_time}")
    return list(map(lambda x: x[0], clusters))


def greedy_parameterless(df: DataFrame):
    print("\nGreedy partitioning...")
    start_time = datetime.now()
    
    df = df.drop(ID_COLUMN)
    cnf = {}
    df.cache()
    total_size = df.count()
    clusters = [(df, total_size, cnf)]        
            
    # partitioning the dataset on each columns
    windows_by_cols = {}
    for c in df.columns:
        windows_by_cols[c] = Window().partitionBy(c)
                      
    max_col_count = total_size        

    # split until splits are too small
    while not (max_col_count <= int(0.03 * total_size)):    
        changed = False
        #  initialize max to None
        max_cluster = max_col_name = max_col_val = max_product = None
        
        # Loop over all clusters
        for cluster_info in clusters:     
            cluster, cluster_count, cnf = cluster_info                                                                                                               
            # skip the clusters that can not improve the split 
            if max_col_count != None and cluster_count < max_col_count:
                continue
            
            # Loop over all columns in the cluster            
            for col_name in cluster.columns:                
                col_val, in_count = cluster.select(col_name).withColumn("count", F.count(F.col(col_name)).over(windows_by_cols[col_name])).orderBy("count", ascending=False).first()
                                
                product = in_count * math.sqrt(cluster_count - in_count)

                # if the current column product is higher than the max, set it as the max.
                if (max_product == None or product >= max_product):
                    max_col_count = in_count
                    max_cluster = cluster_info
                    max_col_name = col_name
                    max_col_val = col_val
                    max_product = product                    

        inPart = max_cluster[0].where(F.col(max_col_name) == max_col_val).cache()            
        outPart = max_cluster[0].where(F.col(max_col_name) != max_col_val).cache()
                
        # Remove the old splitted cluster
        clusters.remove(max_cluster)
        
        # Add the new splitted clusters
        old_cnf = cnf.copy()
        cnf[max_col_name] = max_col_val
        clusters.append((inPart, max_col_count, cnf))
        clusters.append((outPart, max_cluster[1] - max_col_count, old_cnf))
        
    [print(f"cnf: {c[2]}, \t count:{(c[0].count())}") for c in clusters]
    
    end_time = datetime.now()
    print(f"\nTime taken: {end_time - start_time}")
    return list(map(lambda x: x[0], clusters))