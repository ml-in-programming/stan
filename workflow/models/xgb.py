from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV


def train_xgb_clf(x_train, y_train, x_test, y_test):
    print("Start training...")

    # Choose the type of classifier.
    model = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

    # # Choose some parameter combinations to try
    # parameters = {'n_estimators': [400],
    #               'max_features': ['log2'],
    #               'criterion': ['entropy'],
    #               'max_depth': [20],
    #               'min_samples_split': [2],
    #               'min_samples_leaf': [1],
    #               'random_state': [239]
    #               }
    #
    # # Type of scoring used to compare parameter combinations
    # acc_scorer = make_scorer(accuracy_score)
    #
    # # Run the grid search
    # grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer, verbose=50)
    # grid_obj = grid_obj.fit(x_train, y_train)

    # Set the clf to the best combination of parameters
    # clf = grid_obj.best_estimator_

    model.fit(x_train, y_train)
    print(model)
    print("XGB trained")
    predictions = model.predict(x_test)
    print(predictions)
    print(accuracy_score(y_test, predictions))
    return accuracy_score(y_test, predictions)
