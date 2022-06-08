
import math
from pandas import DataFrame
from pyspark.sql.functions import log2 
import pyspark.sql.functions as F


def generateColumnCountHomogenity(N_counts):
    
    def ColumnCountHomogenity(cluster: DataFrame):
        # Count distinct values in columns
        distinct = cluster.agg(*(F.countDistinct(F.col(column)).alias(column)
                                for column in cluster.columns))
        distinct.show()
    
        
        counts = distinct.collect()[0]
        
        #------------        
        p = [n / N for n, N in zip(counts, N_counts)]
        homogenity = 1 - sum(p)/len(p)
        #------------
        
        #homogenity = 1 - sum(counts) / (len(counts) * cluster.count())
        
        return homogenity
    
    return ColumnCountHomogenity



def generateEntropyColumnHomogenity(N_counts):        
    
    def EntropyColumnHomogenity(df: DataFrame):                
        size = df.count()
        
        homo_sum = 0
        for c in df.columns:
            E = df.groupby(c).count().withColumn('entropy', F.col("count") / size * log2(F.col("count") / size))                
            N = N_counts[c]
            if N == 1:
                homo_sum += 1
            else:            
                entropy = - (E.agg({'entropy': 'sum'}).collect()[0][0])
                homo_sum += 1 - (entropy / math.log2(N)) # 1 - Normalized entropy // entropy is normalized over the unique values in the columns of un-clustered data       
        return homo_sum / len(df.columns)

    return EntropyColumnHomogenity
