import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from workflow.models.neural_network import train_neural_network
from workflow.models.random_forest import train_random_forest_clf
from workflow.models.svm import train_svm_clf, train_linear_svm_clf

classes = 'ProjectName'
min_files_in_project = 10
max_files_in_project = 50
test_size = 0.2
root = '../'
random_seed = 566


def load_data(filename):
    print("Loading data...")
    datafile = filename
    print("Data loaded")
    return pd.read_csv(datafile)


def normalize_data(data):
    print("Normalizing data...")
    features_avg_symbols = ["For", "While", "If", "Else", "ElseIf", "TernaryOperator", "Lambda", "Break", "Continue",
                            "Null", "LineComment", "BlockComment", "JavaDocComment", "Tab", "Space", "Whitespace",
                            "Method", "Fields", "LocalVariables", "InnerClasses", "StringConstant", "IntConstant",
                            "CharConstant"]
    features_avg_lines = ["EmptyLine"]
    features_avg_methods = ["MethodsCharacters", "MethodsParameters", "MethodsLines"]
    features_avg_fields = ["PublicFields", "PrivateFields", "FieldsLength"]
    features_avg_variables = ["VariableLength"]
    features_typed = ["TabsLeadLines", "PunctuationBeforeBrace"]
    lines = "Line"
    symbols = "TotalLength"
    methods = "Method"
    fields = "Fields"
    variables = "LocalVariables"

    for feature in features_typed:
        le = preprocessing.LabelEncoder()
        le = le.fit(data[feature])
        data[feature] = le.transform(data[feature])

    data = data.astype('float32', copy=False, errors='ignore')
    for index, row in data.iterrows():
        for feature in features_avg_symbols:
            if row[feature] != 0:
                data.at[index, feature] = -np.log(row[feature] * 1. / row[symbols])
            else:
                data.at[index, feature] = 0
        for feature in features_avg_lines:
            if row[lines] != 0:
                data.at[index, feature] /= 1. * row[lines]
        for feature in features_avg_methods:
            if row[methods] != 0:
                data.at[index, feature] /= 1. * row[methods]
        for feature in features_avg_fields:
            if row[fields] != 0:
                data.at[index, feature] /= 1. * row[fields]
        for feature in features_avg_variables:
            if row[variables] != 0:
                data.at[index, feature] /= 1. * row[variables]

    print("Data normalized")
    return data


def split_data(train, test):
    print("Splitting data...")
    x_train = train.drop(['Path', 'ProjectName', 'TotalLength', 'Line'], axis=1)
    x_test = test.drop(['Path', 'ProjectName', 'TotalLength', 'Line'], axis=1)
    y_train = train[classes]
    y_test = test[classes]
    print("Data split")
    return x_train, x_test, y_train, y_test


def save_data(data, filename):
    print("Saving data to {0}...".format(filename))
    data.to_csv(filename, index=False)
    print("Saved")


def drop_edge_classes(data, min_threshold, max_threshold):
    print("Dropping classes with population lower than {0} or greater than {1}...".format(min_threshold, max_threshold))
    counts = data[classes].value_counts()
    for index, row in data.iterrows():
        if counts[row[classes]] < min_threshold or max_threshold < counts[row[classes]]:
            data.drop(index, inplace=True)
    print("Dropped")
    return data


def shuffle_data(data):
    return data.sample(frac=1)


def get_equal_samples(data, size):
    print("Picking random samples from all classes...")
    train, test = pd.DataFrame(), pd.DataFrame()
    class_names = data[classes].unique()
    counts = data[classes].value_counts()
    for name in class_names:
        sample = data.query('{0} == {1}'.format(classes, "'" + name + "'")).sample(min(size, counts[name]))
        train_sample, test_sample = train_test_split(sample, test_size=0.2, random_state=random_seed)
        train = train.append(train_sample)
        test = test.append(test_sample)
    # train = shuffle_data(train)
    # test = shuffle_data(test)
    print("Samples are ready")
    return split_data(train, test)


def main():
    data = drop_edge_classes(load_data(root + 'populated_data.csv'), min_files_in_project, max_files_in_project)
    x_train, x_test, y_train, y_test = get_equal_samples(data, max_files_in_project)
    results = [train_random_forest_clf(x_train, y_train, x_test, y_test),
               train_neural_network(x_train, y_train, x_test, y_test),
               train_svm_clf(x_train, y_train, x_test, y_test),
               train_linear_svm_clf(x_train, y_train, x_test, y_test)]
    print(results)
    # save_clf(clf)


if __name__ == '__main__':
    main()
