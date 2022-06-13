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

    # need to split k-1 times.
    for _ in range(k-1):

        #  initialize max to None
        max_cluster = max_col_name = max_col_val = max_product = None

        # Loop over all clusters
        for cluster in clusters:

            # Get the column counts for the current cluster
            columns_counts = [(c, cluster.groupby(c).count())
                              for c in cluster.drop(ID_COLUMN).columns]

            # Loop over all columns in the cluster
            for (col_name, column_counts) in columns_counts:

                # max column count
                count_dict = column_counts.rdd.max(lambda x: x[1]).asDict()
                col_val = count_dict[col_name]

                in_count = count_dict["count"]
                out_count = column_counts.rdd.filter(
                    lambda row: row[col_name] != col_val).map(lambda row: row["count"]).sum()
                product = in_count * out_count

                # if the current column product is higher than the max, set it as the max.
                if (max_product == None or product > max_product):
                    max_cluster = cluster
                    max_col_name = col_name
                    max_col_val = col_val
                    max_product = product

        # print("col_name: ", max_col_name, " col_val: \"",
        #       max_col_val, "\"", " product: ", max_product)

        inPart = max_cluster.rdd.filter(
            lambda row: row[max_col_name] == max_col_val).toDF()
        outPart = max_cluster.rdd.filter(
            lambda row: row[max_col_name] != max_col_val).toDF()

        # Remove the old splitted cluster
        clusters.remove(max_cluster)

        # Add the new splitted clusters
        clusters.append(inPart)
        clusters.append(outPart)

    # [c.show() for c in clusters]
    return clusters
