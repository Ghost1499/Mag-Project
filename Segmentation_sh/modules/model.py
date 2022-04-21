from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
import numpy as np

from Segmentation_sh.params_config import data_path


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


def plot_training_curve(model, X_train, y_train):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator=model,
                       X=X_train,
                       y=y_train,
                       train_sizes=np.linspace(0.1, 1.0, 5),
                       cv=10,
                       n_jobs=1,
                       shuffle=True)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='Training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='Validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.03])
    plt.tight_layout()
    plt.savefig(data_path / 'training_curve.png', dpi=300)
    plt.show()
