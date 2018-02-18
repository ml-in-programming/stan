import gc

import numpy as np
import pandas as pd
from pympler import tracker
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from lstm.character_lstm import train_char_lstm


def load_data():
    datafile = 'data.csv'
    return pd.read_csv(datafile)


def print_header(output):
    output.write("Path")
    for i in range(256):
        output.write(", Weight-{0}".format(i))

    output.write("\n")


def add_lstm_weights(data):
    weightsfile = 'weights.csv'
    weights = pd.read_csv(weightsfile)
    processed = weights['Path'].str
    f = open(weightsfile, "a")

    tr = tracker.SummaryTracker()
    for index, row in data.iterrows():
        path = row['Path']
        if processed.contains(path).sum() == 0:
            tr.print_diff()
            train_char_lstm(path, f, tr)
            # tr.print_diff()
            gc.collect()
            print("Collected")
            tr.print_diff()
    f.close()

    weights = pd.read_csv(weightsfile)
    return pd.merge(data, weights, on='Path')


def prepare_data(data):
    print("Preparing data...")
    features_avg_symbols = ["For", "While", "If", "Else", "ElseIf", "TernaryOperator", "Lambda", "Break", "Continue",
                            "Null", "LineComment", "BlockComment", "JavaDocComment", "Tab", "Space", "Whitespace",
                            "Method", "Fields", "LocalVariables", "InnerClasses", "StringConstant", "IntConstants",
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

    # counter = dict()
    # dropping = []
    # for index, row in data.iterrows():
    #     project = row['ProjectName']
    #     current = counter.get(project, 0)
    #     if current < 10 and project != 'jTRACE' and project != 'OnlyLZ2' and project != 'rsdnAndroid':
    #         counter[project] = current + 1
    #     else:
    #         dropping.append(index)
    # for index in reversed(dropping):
    #     data = data.drop(data.index[index])
    data_with_weights = add_lstm_weights(data)

    train, test = train_test_split(data, test_size=0.2)
    x_train = train.drop(['Path', 'ProjectName', "TotalLength", "Line"], axis=1)
    x_test = test.drop(['Path', 'ProjectName', "TotalLength", "Line"], axis=1)
    y_train = train['ProjectName']
    y_test = test['ProjectName']
    print("Data prepared")

    return x_train, y_train, x_test, y_test


def train_clf(x_train, y_train, x_test, y_test):
    print("Start training...")

    # Choose the type of classifier.
    clf = RandomForestClassifier()

    # Choose some parameter combinations to try
    parameters = {'n_estimators': [100],
                  'max_features': ['auto'],
                  'criterion': ['gini'],
                  'max_depth': [10],
                  'min_samples_split': [3],
                  'min_samples_leaf': [5]
                  }

    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(x_train, y_train)

    # Set the clf to the best combination of parameters
    clf = grid_obj.best_estimator_

    print("RF trained")
    predictions = clf.predict(x_test)
    print(accuracy_score(y_test, predictions))
    return clf


def save_clf(clf):
    joblib.dump(clf, 'rf_classifier.pkl')


def main():
    data = load_data()
    x_train, y_train, x_test, y_test = prepare_data(data)
    clf = train_clf(x_train, y_train, x_test, y_test)
    save_clf(clf)


if __name__ == '__main__':
    main()
