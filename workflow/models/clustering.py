from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np


def kmeans_clustering(x_train, y_train, x_test, y_test, reverse_mapping):
    x = x_test
    y = y_test

    scaler = preprocessing.StandardScaler()
    scaled_values = scaler.fit_transform(x)
    x.loc[:, :] = scaled_values
    x = x[['Method', 'Fields', 'LocalVariables', 'RatioEmptyLine', 'AstMaxDepth', 'RatioBlockComment', 'AvgMethodsLines', 'RatioSpace', 'RatioJavaDocComment']]
    x = x.sample(frac=1, random_state=239)
    for clusters in [2, 3, 5, 7, 10, 15, 25]:
        print("Clusters: {}".format(clusters))
        kmeans = KMeans(n_clusters=clusters, n_jobs=4)
        predictions = np.array(kmeans.fit_predict(x))
        # print(predictions)
        # print(np.unique(predictions, return_counts=True))
        members = [[] for _ in range(clusters)]
        for cluster, index in zip(predictions, x.index):
            members[cluster].append(y['Correct'][index])
        members = np.array(members)
        for c in range(clusters):
            nums, counts = np.unique(members[c], return_counts=True)
            if len(nums) == 2:
                print("Total: {}, ratio: {}".format(counts[0] + counts[1], counts[0] / (counts[0] + counts[1])))
            elif len(nums) == 1:
                print("Total: {}, only {}".format(counts[0], nums[0]))
            else:
                print("What?")
