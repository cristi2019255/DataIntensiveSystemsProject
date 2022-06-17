
from datetime import datetime
from pyspark.ml.fpm import FPGrowth
import pyspark.sql.functions as F
from constants import *
from pyspark.sql import DataFrame

def pattern_partitioning(df: DataFrame, k: int):
    print("\nPattern partitioning...")
    start_time = datetime.now()    
    size = df.count()
    df = df.withColumn("items", F.array_distinct(F.array(*[F.col(columnName) for columnName in df.columns[:-1]])))
        
    min_supp = 1/size
    print(min_supp)
    
    fpGrowth = FPGrowth(itemsCol="items", minSupport=min_supp)
    model = fpGrowth.fit(df.cache())

    freq = model.freqItemsets.withColumn("size", F.size(F.col("items"))).withColumn("area", F.col("freq") * F.col("size"))    
    # Display frequent itemsets.
    df_patterns = freq.sort(F.col("area"), ascending=False).select(F.col("items"))
    df_patterns.show(k)
    clusters_cnf = df_patterns.take(k)    
    
    clusters = []
    for i in range(k):
        cnf = set(clusters_cnf[i][0])
        
        # get the transactions that contain the current cnf list        
        cluster = df.rdd.filter(lambda x: cnf.issubset(set(x["items"]))).toDF().drop("items")        
        print(cnf, ": ", cluster.count())        
        clusters.append(cluster)        
 
    end_time = datetime.now()
    print(f"\nTime taken: {end_time - start_time}")
    
    return clusters
    
    
