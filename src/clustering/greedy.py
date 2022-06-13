import pyspark.sql.functions as F
from constants import *

from pyspark.sql import DataFrame


def greedy_heuristic(df: DataFrame, k: int, homogenity):
    df = df.drop(ID_COLUMN)
    df_left = df.select('*')

    columns_counts = [df.groupby(c).count().sort(
        F.col("count"), ascending=False).collect() for c in df.columns]

    parts = []
    for _ in range(k):
        maximum = max([max(x, key=lambda y:y[1])
                      for x in columns_counts], key=lambda x: x[1])

        max_attr_val = maximum[0]
        col = list(maximum.asDict().keys())[0]

        parts.append(df.where(df[col] == max_attr_val))
        df_left = df_left.where(df[col] != max_attr_val)

        columns_counts = [df_left.groupby(c).count().sort(
            F.col("count"), ascending=False).collect() for c in df.columns]

        parts[-1].show()

    while df_left.count() != 0:
        # until partition is not complete for each part add the subparts the increases homogenity the most

        # getting the maximum homogen subpart over column
        maximum = max([max(x, key=lambda y:y[1])
                      for x in columns_counts], key=lambda x: x[1])
        max_attr_val = maximum[0]
        col = list(maximum.asDict().keys())[0]

        subpart = df.where(df[col] == max_attr_val)
        # searching the part that increases homogenity the most with the subpart
        best_part = 0
        homogenity_best = 0
        for i in range(k):
            homo = homogenity(parts[i].union(subpart))
            if homo > homogenity_best:
                homogenity_best = homo
                best_part = i

        # merging subpart with the part
        parts[best_part] = parts[best_part].union(subpart)
        parts[best_part].show()
        # removing the subpart from the dataframe
        df_left = df_left.where(df[col] != max_attr_val)
        # removing the subpart from column_counts
        columns_counts = [df_left.groupby(c).count().sort(
            F.col("count"), ascending=False).collect() for c in df.columns]

        print(df_left.count())

    return parts


def greedy(df: DataFrame, k: int, homogenity):

    # columns_counts = [df.groupby(c).count().sort(F.col("count"), ascending=False).collect() for c in df.columns]
    # maximum = max([max(x, key=lambda y:y[1] for x in columns_counts], key=lambda x: x[1])

    clusters = [df]

    for i in range(k):

        max_cluster = max_col = max_col_val = max_count = None

        for cluster in clusters:

            cluster.show()
            columns_counts = [cluster.groupby(c).count()
                              for c in cluster.drop(ID_COLUMN).columns]

            for column_counts in columns_counts:

                # max column count
                in_count = column_counts.rdd.max(lambda x: x[1])
                out_count = column_counts.where(
                    lambda count: max_col[0] != count[0]).sum(lambda count: count[1])
                product = in_count * out_count

                maximumColumnValue = max(columns_counts, key=lambda x: x[1])

                col_val, col = maximumColumnValue[0], list(
                    maximumColumnValue.asDict().keys())[0]
                col_count = maximumColumnValue[1]

                if (max_count == None or col_count > max_count):
                    max_cluster = cluster
                    max_col = col
                    max_col_val = col_val
                    max_count = col_count

        print(max_col)
        print(max_col_val)

        inPart = max_cluster.where(max_cluster[max_col] == max_col_val)
        outPart = max_cluster.where(max_cluster[max_col] != max_col_val)

        # Remove the old splitted cluster
        clusters.remove(max_cluster)

        # Add the new splitted clusters
        clusters.append(inPart)
        clusters.append(outPart)

    [c.show() for c in clusters]
