import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from workflow.models.neural_network import train_neural_network
from workflow.models.random_forest import train_random_forest_clf
from workflow.models.svm import train_svm_clf, train_linear_svm_clf
from workflow.models.xgb import train_xgb_clf

classes = 'Project'
min_files_in_project = 10
max_files_in_project = 50
test_size = 0.2
root = '../'
random_seed = 566


def load_data(filename):
    print("Loading data...")
    datafile = filename
    data = pd.read_csv(datafile)
    print("Data loaded")
    return data


def normalize_data(data):
    print("Normalizing data...")
    int_features = ['Method', 'Fields', 'LocalVariables']
    to_add = []
    for feature in data.columns.values:
        if feature.startswith("Nominal"):
            print(feature)
            values = data[feature].unique()
            new_features = list(map(lambda s: "Encoded" + feature + s, values))
            arrs = [[] for _ in range(len(values))]
            for index, row in data.iterrows():
                for value, arr in zip(values, arrs):
                    if row[feature] == value:
                        arr.append(1)
                    else:
                        arr.append(0)
            to_add.append((new_features, arrs))
        if feature in int_features:
            le = preprocessing.MinMaxScaler()
            data[feature] = le.fit_transform(data[[feature]])
    for new_features, arrs in to_add:
        for col_name, col_values in zip(new_features, arrs):
            data[col_name] = pd.Series(col_values)
    print("Data normalized")
    return data


def split_data(train, test):
    print("Splitting data...")
    to_drop = []
    for feature in train.columns.values:
        if feature.startswith("Nominal"):
            to_drop.append(feature)
    to_drop.append('Path')
    to_drop.append(classes)
    x_train = train.drop(to_drop, axis=1)
    x_test = test.drop(to_drop, axis=1)
    y_train = train[classes]
    y_test = test[classes]
    print("Data split")
    return x_train, x_test, y_train, y_test


def save_data(data, filename):
    print("Saving data to {0}...".format(filename))
    data.to_csv(filename, index=False)
    print("Saved")


def drop_edge_classes(data, min_threshold, max_threshold=10000000):
    print("Dropping classes with population lower than {0} or greater than {1}...".format(min_threshold, max_threshold))
    counts = data[classes].value_counts()
    for index, row in data.iterrows():
        if counts[row[classes]] < min_threshold or max_threshold < counts[row[classes]]:
            data.drop(index, inplace=True)
    print("Dropped")
    return data


def shuffle_data(data):
    return data.sample(frac=1)


def get_equal_samples(data, size=None):
    print("Picking random samples from all classes...")
    train, test = pd.DataFrame(), pd.DataFrame()
    class_names = data[classes].unique()[:30]
    counts = data[classes].value_counts()
    for name in class_names:
        sample = data.query('{0} == {1}'.format(classes, "'" + name + "'"))  # .sample(min(size, counts[name]))
        train_sample, test_sample = train_test_split(sample, test_size=0.2, random_state=random_seed)
        train = train.append(train_sample)
        test = test.append(test_sample)
    print("classes:", len(class_names))
    print("train:", len(train))
    print("test:", len(test))
    # train = shuffle_data(train)
    # test = shuffle_data(test)
    print("Samples are ready")
    return split_data(train, test)


def main():
    data = load_data(root + 'normalized_10_data.csv')
    # data = normalize_data(data)
    # save_data(data, root + 'normalized_data_old.csv')
    # data = drop_edge_classes(data, 10)
    # save_data(data, root + 'normalized_10_data_old.csv')
    x_train, x_test, y_train, y_test = get_equal_samples(data)
    # results = [train_random_forest_clf(x_train, y_train, x_test, y_test),
    #            train_neural_network(x_train, y_train, x_test, y_test),
    #            train_svm_clf(x_train, y_train, x_test, y_test),
    #            train_linear_svm_clf(x_train, y_train, x_test, y_test)]
    results = train_random_forest_clf(x_train, y_train, x_test, y_test)
    print(results)
    # results = train_linear_svm_clf(x_train, y_train, x_test, y_test)
    # print(results)
    # save_clf(clf)


if __name__ == '__main__':
    main()
