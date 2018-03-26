from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV


def train_svm_clf(x_train, y_train, x_test, y_test):
    print("Start training...")

    # Choose the type of classifier.
    clf = SVC()

    # Choose some parameter combinations to try
    parameters = {'kernel': ['rbf'],
                  'C': [25],
                  'gamma': [0.0002],
                  'random_state': [239]
                  }

    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer, verbose=50)
    grid_obj = grid_obj.fit(x_train, y_train)

    # Set the clf to the best combination of parameters
    clf = grid_obj.best_estimator_

    print("SVM trained")
    predictions = clf.predict(x_test)
    return accuracy_score(y_test, predictions)


def train_linear_svm_clf(x_train, y_train, x_test, y_test):
    print("Start training...")

    # Choose the type of classifier.
    clf = LinearSVC()

    # Choose some parameter combinations to try
    # parameters = {'penalty': ['l2'],
    #               'C': [20],
    #               'dual': [False],
    #               'random_state': [239]
    #               }

    parameters = {'penalty': ['l2'],
                  'C': [10],
                  'dual': [True],
                  'random_state': [239]
                  }

    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer, verbose=50)
    grid_obj = grid_obj.fit(x_train, y_train)

    # Set the clf to the best combination of parameters
    clf = grid_obj.best_estimator_

    print("SVM trained")
    predictions = clf.predict(x_test)
    return accuracy_score(y_test, predictions)
