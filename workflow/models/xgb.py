from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score
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


def train_sklearn_xgb_classifier(x_train, y_train, x_test, y_test, *, full=False):
    print("Start training...")

    params = {
        "n_estimators": 300,
        'tree_method': 'hist',
        'max_depth': 3,
        'learning_rate': 0.1,
        'min_child_weight': 1,
        'n_jobs': 4
    }

    model = XGBClassifier(**params)
    tic = time.time()
    model.fit(x_train, y_train)
    print('passed time with XGBClassifier (hist, cpu): %.3fs' % (time.time() - tic))

    pickle.dump(model, open("pima.pickle.dat", "wb"))
    feature_importances = sorted(zip(x_train.columns.values, model.feature_importances_), key=lambda x: x[1])
    # print(feature_importances)

    if not full:
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        print(accuracy)
        return accuracy
    else:
        classes = model.classes_
        predictions = model.predict_proba(x_test)
        n_classes = model.n_classes_
        positions = np.zeros(n_classes)
        pos_misses = []
        for prediction, answer in zip(predictions, y_test):
            proba = prediction.copy()
            prediction = np.argsort(prediction)[::-1]
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
        return accuracy
