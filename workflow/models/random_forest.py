from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV


def train_random_forest_clf(x_train, y_train, x_test, y_test):
    print("Start training...")

    # Choose the type of classifier.
    clf = RandomForestClassifier()

    # Choose some parameter combinations to try
    parameters = {'n_estimators': [400],
                  'max_features': ['log2'],
                  'criterion': ['entropy'],
                  'max_depth': [20],
                  'min_samples_split': [2],
                  'min_samples_leaf': [1],
                  'random_state': [239]
                  }

    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer, verbose=50)
    grid_obj = grid_obj.fit(x_train, y_train)

    # Set the clf to the best combination of parameters
    clf = grid_obj.best_estimator_

    print("RF trained")
    predictions = clf.predict(x_test)
    print(accuracy_score(y_test, predictions))
    print(grid_obj.best_params_)
    return accuracy_score(y_test, predictions)


def save_clf(clf):
    joblib.dump(clf, 'rf_classifier.pkl')
