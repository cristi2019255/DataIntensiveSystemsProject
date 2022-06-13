import pyspark.sql.functions as F
from pyspark.sql import DataFrame

def generateColumnCountHomogenity(df: DataFrame):
    distinct = df.agg(*(F.countDistinct(F.col(column)).alias(column)
                        for column in df.columns))

    N_counts = distinct.collect()[0]
    
    def ColumnCountHomogenity(cluster: DataFrame):
        # Count distinct values in columns
        distinct = cluster.agg(*(F.countDistinct(F.col(column)).alias(column)
                                 for column in cluster.columns))
        #distinct.show()

        counts = distinct.collect()[0]

        # ------------
        p = [n / N for n, N in zip(counts, N_counts)]
        homogenity = 1 - sum(p)/len(p)
        # ------------

        #homogenity = 1 - sum(counts) / (len(counts) * cluster.count())

        return homogenity

    return ColumnCountHomogenity
