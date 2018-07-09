from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import time
import xgboost as xgb
from xgboost.compat import XGBLabelEncoder
import pickle


def train_xgb_clf(x_train, y_train, x_test, y_test):
    print("Start training...")

    classes = np.unique(y_train)
    n_classes = len(classes)

    le = XGBLabelEncoder().fit(y_train)
    training_labels = le.transform(y_train)

    param = {
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': 0.1,
        'min_child_weight': 1,
        "n_estimators": 300,
        'tree_method': 'hist',
        'silent': 1,
        'objective': "multi:softprob",
        'num_class': n_classes,
        'nthread': 4
    }
    num_round = 300

    dtrain = xgb.DMatrix(x_train, label=training_labels)
    tic = time.time()
    model = xgb.train(param, dtrain, num_round)
    print('passed time with xgb (hist, cpu): %.3fs' % (time.time() - tic))

    classes = set(y_test)
    dtest = xgb.DMatrix(x_test)
    # print(model.get_fscore())
    # print(model.get_score())
    predictions = model.predict(dtest)
    print(predictions)
    column_indexes = np.argmax(predictions, axis=1)
    predictions = le.inverse_transform(column_indexes)
    for i, prediction in enumerate(predictions):
        best = -100
        prediction = round(prediction)
        for answer in classes:
            if abs(answer - prediction) < abs(best - prediction):
                best = answer
            predictions[i] = best
    print(predictions)
    # print(y_test)
    print(accuracy_score(y_test, predictions))

    return accuracy_score(y_test, predictions)


def train_sklearn_xgb_classifier(x_train, y_train, x_test, y_test, target, path_to_classifier=None, nthread=4):
    print("Start training...")

    params = {
        "n_estimators": 200,
        'tree_method': 'hist',
        'max_depth': 3,
        'learning_rate': 0.2,
        'n_jobs': nthread
    }

    model = XGBClassifier(**params)
    tic = time.time()
    model.fit(x_train, y_train[target])
    print('passed time with XGBClassifier (hist, cpu): %.3fs' % (time.time() - tic))

    if path_to_classifier:
        pickle.dump(model, open(path_to_classifier, "wb"))

    feature_importances = sorted(zip(x_train.columns.values, model.feature_importances_), key=lambda x: x[1])
    print(list(map(lambda p: p[0], feature_importances[-10:])))

    classes = model.classes_
    predictions = model.predict_proba(x_test)
    n_classes = model.n_classes_
    positions = np.zeros(n_classes)
    pos_misses = []
    correctness = []
    for prediction, (index, row) in zip(predictions, y_test.iterrows()):
        answer = row[target]
        proba = prediction.copy()
        prediction = np.argsort(prediction)[::-1]
        if classes[prediction[0]] != answer:
            # print("Missed on: {} ({}), decided: {}".format(row['Path'], answer, classes[prediction[0]]))
            correctness.append(0)
        else:
            correctness.append(1)
        for i in range(n_classes):
            if classes[prediction[i]] == answer:
                positions[i] += 1
                if i != 0:
                    pos_misses.append(proba)
                break

    pos_misses = np.array(pos_misses)
    for i in range(1, n_classes):
        positions[i] += positions[i - 1]
    print(positions)
    accuracy = positions / float(len(y_test))
    plt.plot(np.arange(1, n_classes + 1, 1), accuracy, 'b-')
    plt.show()
    return correctness
