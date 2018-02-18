import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split


root = '../'
classes = 'ProjectName'
min_files_in_project = 10
test_size = 0.2


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


def drop_low_populated_classes(data, threshold):
    print("Dropping classes with population lower than {0}...".format(threshold))
    counts = data[classes].value_counts()
    for index, row in data.iterrows():
        if counts[row[classes]] < threshold:
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
        sample = data.query('{0} == {1}'.format(classes, "'" + name + "'"))
        train_sample, test_sample = train_test_split(sample, test_size=0.2)
        train = train.append(train_sample)
        test = test.append(test_sample)
    train = shuffle_data(train)
    test = shuffle_data(test)
    print("Samples are ready")
    return split_data(train, test)


def train_clf(x_train, y_train, x_test, y_test):
    print("Start training...")

    # Choose the type of classifier.
    clf = RandomForestClassifier()

    # Choose some parameter combinations to try
    parameters = {'n_estimators': [50, 75, 100, 125, 150, 175, 200],
              'max_features': ['log2'],
              'criterion': ['entropy'],
              'max_depth': [12],
              'min_samples_split': [3],
              'min_samples_leaf': [1]
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
    print(grid_obj.best_params_)
    return clf


def save_clf(clf):
    joblib.dump(clf, 'rf_classifier.pkl')


def main():
    data = load_data(root + 'populated_data.csv')
    x_train, x_test, y_train, y_test = get_equal_samples(data, 10 * min_files_in_project)
    clf = train_clf(x_train, y_train, x_test, y_test)
    save_clf(clf)


if __name__ == '__main__':
    main()
