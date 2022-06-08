from math import log


def entropy(X, total_count):
    return - X.reduce(lambda a, b: a/total_count * log(a/total_count) + b/total_count * log(b/total_count))[0]
