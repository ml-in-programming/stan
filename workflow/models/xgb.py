from xgboost import XGBClassifier, callback
import numpy as np
import matplotlib.pyplot as plt
import time
import xgboost as xgb
from xgboost.compat import XGBLabelEncoder
import pickle
import warnings

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, confusion_matrix


def select_best(x, y, k):
    selector = SelectKBest(f_classif, k=k)
    selector.fit(x, y)
    mask = selector.get_support()  # list of booleans
    new_features = []  # The list of your K best features

    for contained, feature in zip(mask, x.columns.values):
        if contained:
            new_features.append(feature)
        else:
            print("Dropped {}".format(feature))

    return pd.DataFrame(selector.transform(x), columns=new_features)


def cv(data, target, bucket_target, nthread, n_buckets):
    # data = data.loc[data['AvgMethodsLines'] > .5]
    classes = np.unique(data[target])
    n_classes = len(classes)
    le = XGBLabelEncoder().fit(data[target])

    data_y = data[target]
    data_x = data.drop([target, bucket_target, 'Path', 'NominalTabsLeadLines', 'NominalPunctuationBeforeBrace'], axis=1)

    folds = []
    for b in range(n_buckets):
        indices = data[bucket_target] == b
        folds.append((np.argwhere(np.invert(indices)), np.argwhere(indices)))

    param = {
        'objective': "multi:softprob",
        'eval_metric': ['mlogloss', 'merror'],
        'seed': 239,
        'eta': 0.2,
        'max_depth': 3,
        'tree_method': 'hist',
        'silent': 1,
        'num_class': n_classes,
        'nthread': nthread
    }

    num_round = 20

    xg_train = xgb.DMatrix(data_x, label=le.transform(data_y))
    xgb.cv(param, xg_train, num_round, folds=folds,
           metrics={'merror', 'mlogloss'}, seed=239,
           callbacks=[xgb.callback.print_evaluation(show_stdv=True)])


def run_train(data, target, bucket_target, nthread, n_buckets):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    counts = data[target].value_counts()
    n_classes = len(counts)
    le = XGBLabelEncoder().fit(data[target])

    big = []
    small = []
    for i in range(n_classes):
        if counts[le.inverse_transform(i)] < 100:
            small.append(le.inverse_transform(i))
        else:
            big.append(le.inverse_transform(i))
    data = data.loc[data[target].isin(small)]

    data_y = data[target]
    data_x = data.drop([target, bucket_target, 'Path', 'NominalTabsLeadLines', 'NominalPunctuationBeforeBrace'], axis=1)

    # print(min(data_x['AvgMethodsLines']), max(data_x['AvgMethodsLines']))
    # data_x = select_best(data_x, data_y, data_x.shape[1] // 2)
    # pca = PCA(n_components=data_x.shape[1] // 2)
    # data_x = StandardScaler().fit_transform(data_x)
    # data_x = pd.DataFrame(pca.fit_transform(data_x))
    # print(data_x.shape)

    train_indices = data[bucket_target].isin(range(4, n_buckets))
    test_indices = data[bucket_target].isin(range(0, 4))

    replication_coeff = 6

    rep_indices = data[target].isin(small) & train_indices

    replicate_x = data_x.loc[rep_indices]
    # x_train = data_x.loc[train_indices]
    x_train = data_x.loc[train_indices].append([replicate_x] * replication_coeff, ignore_index=True)

    replicate_y = data_y.loc[rep_indices]
    # y_train = le.transform(data_y.loc[train_indices])
    y_train = le.transform(data_y.loc[train_indices].append([replicate_y] * replication_coeff, ignore_index=True))

    x_test = data_x.loc[test_indices]
    y_test = le.transform(data_y.loc[test_indices])

    weights_train = np.ones(len(y_train))
    max_cnt = max(counts)
    for i, cls in enumerate(y_train):
        if le.inverse_transform(cls) in small:
            weights_train[i] = max_cnt / (counts[le.inverse_transform(cls)] * replication_coeff)
    # print(weights_train)

    xg_train = xgb.DMatrix(x_train, label=y_train, weight=weights_train)
    xg_test = xgb.DMatrix(x_test, label=y_test)

    param = {
        'objective': "multi:softprob",
        'eval_metric': ['mlogloss', 'merror'],
        'seed': 239,
        'eta': 0.2,
        'max_depth': 3,
        'tree_method': 'hist',
        'silent': 1,
        'num_class': n_classes,
        'nthread': nthread
    }

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 150
    boost = xgb.train(param, xg_train, num_round, watchlist)
    pred_prob = boost.predict(xg_test).reshape(len(y_test), n_classes)
    pred_label = np.argmax(pred_prob, axis=1)
    error_rate = np.sum(pred_label != y_test) / len(y_test)
    print('Test error using softprob = {}'.format(error_rate))
    print('Baseline = {}'.format(0.25326370757180156))

    precision, recall, fscore, support = score(y_test, pred_label)
    np.set_printoptions(precision=4, linewidth=150)
    print('precision: {}'.format(precision))
    print('recall:    {}'.format(recall))
    print('fscore:    {}'.format(fscore))
    print('support:   {}'.format(support))

    old_matrix = np.array([[82, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                           [7, 4, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 7, 0, 3, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 10, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 1, 0, 9, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 2, 1, 0, 0, 1, 0, 1, 7, 2, 0, 0, 0, 0, 0, 0, 0],
                           [9, 0, 0, 0, 0, 1, 0, 0, 1, 0, 72, 0, 0, 0, 1, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 0, 1, 0, 0, 0, 0],
                           [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 9, 0, 0, 0, 0],
                           [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 0, 2, 0],
                           [7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 0, 3, 0, 0, 6, 0],
                           [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 1, 0, 3]]
                          )
    print(confusion_matrix(y_test, pred_label))
    print()
    print(confusion_matrix(y_test, pred_label) - old_matrix)

    cnt_s_b = 0
    cnt_b_s = 0
    for pred, corr in zip(pred_label, y_test):
        if counts[le.inverse_transform(corr)] < 100 and counts[le.inverse_transform(pred)] > 100:
            cnt_s_b += 1

        if counts[le.inverse_transform(corr)] > 100 and counts[le.inverse_transform(pred)] < 100:
            cnt_b_s += 1

    print('Number of small->big mistakes: {}'.format(cnt_s_b))
    print('Number of big->small mistakes: {}'.format(cnt_b_s))


def xgb_train_buckets(data, target, bucket_target='Bucket', nthread=4, n_buckets=10):
    print("Start training with {} buckets...".format(n_buckets))
    # cv(data, target, bucket_target, nthread, n_buckets)
    run_train(data, target, bucket_target, nthread, n_buckets)


def train_xgb_clf(x_train, y_train, x_test, y_test, target, nthread=4):
    print("Start training...")

    y_train = y_train[target]
    classes = np.unique(y_train)
    n_classes = len(classes)

    le = XGBLabelEncoder().fit(y_train)
    training_labels = le.transform(y_train)

    param = {
        'max_depth': 3,
        'eta': 0.2,
        'min_child_weight': 1,
        "n_estimators": 200,
        'tree_method': 'hist',
        'silent': 1,
        'verbose_eval': False,
        'objective': "multi:softprob",
        'num_class': n_classes,
        'nthread': nthread
    }

    num_round = 200

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
        'n_jobs': nthread,
        'eval_metric': ['mlogloss', 'merror']
    }

    model = XGBClassifier(**params)
    tic = time.time()
    model.fit(x_train, y_train[target], eval_set=[(x_train, y_train[target]), (x_test, y_test[target])], verbose=True,
              early_stopping_rounds=10)
    print('passed time with XGBClassifier (hist, cpu): %.3fs' % (time.time() - tic))

    if path_to_classifier:
        pickle.dump(model, open(path_to_classifier, "wb"))

    # feature_importances = sorted(zip(x_train.columns.values, model.feature_importances_), key=lambda x: x[1])
    # print(list(map(lambda p: p[0], feature_importances[-10:])))

    # classes = model.classes_
    # predictions = model.predict_proba(x_test)
    # n_classes = model.n_classes_
    # positions = np.zeros(n_classes)
    # pos_misses = []
    # correctness = []
    # for prediction, (index, row) in zip(predictions, y_test.iterrows()):
    #     answer = row[target]
    #     proba = prediction.copy()
    #     prediction = np.argsort(prediction)[::-1]
    #     if classes[prediction[0]] != answer:
    #         # print("Missed on: {} ({}), decided: {}".format(row['Path'], answer, classes[prediction[0]]))
    #         correctness.append(0)
    #     else:
    #         correctness.append(1)
    #     for i in range(n_classes):
    #         if classes[prediction[i]] == answer:
    #             positions[i] += 1
    #             if i != 0:
    #                 pos_misses.append(proba)
    #             break
    #
    # pos_misses = np.array(pos_misses)
    # for i in range(1, n_classes):
    #     positions[i] += positions[i - 1]
    # print(positions)
    # accuracy = positions / float(len(y_test))
    # plt.plot(np.arange(1, n_classes + 1, 1), accuracy, 'b-')
    # plt.show()
    # return correctness
