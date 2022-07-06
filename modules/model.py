from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC


def get_best_model(X_train, y_train):
    param_range = [0.0001, 0.001, 0.01, 0.1,
                   1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'C': param_range},
                  # {'C': param_range,
                  # 'gamma': param_range,
                  # 'kernel': ['rbf']}
                  ]
    grid = GridSearchCV(LinearSVC(), param_grid, cv=3)
    grid.fit(X_train, y_train)
    print(grid.best_score_)
    model = grid.best_estimator_
    return model
