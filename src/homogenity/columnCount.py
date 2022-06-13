import pyspark.sql.functions as F
from pandas import DataFrame


def generateColumnCountHomogenity(N_counts):

    def ColumnCountHomogenity(cluster: DataFrame):
        # Count distinct values in columns
        distinct = cluster.agg(*(F.countDistinct(F.col(column)).alias(column)
                                 for column in cluster.columns))
        distinct.show()

        counts = distinct.collect()[0]

        # ------------
        p = [n / N for n, N in zip(counts, N_counts)]
        homogenity = 1 - sum(p)/len(p)
        # ------------

        #homogenity = 1 - sum(counts) / (len(counts) * cluster.count())

        return homogenity

    return ColumnCountHomogenity
