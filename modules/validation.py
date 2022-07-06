from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve

from modules.regions import append_border


class AValidator():
    __metaclass__ = ABCMeta
    _validation_data = []
    _root_dir: Path

    def add_data(self, data):
        """
        Добавление валидационных данных
        :return:
        """
        self._check_data(data)
        self._validation_data.append(data)

    def save_data(self, save_folder: Union[str, Path]):
        """
        Сохранение валидационных данных в директорию
        :return:
        """
        if not save_folder:
            raise Exception("Папка для сохранения результатов не задана")
        save_folder = self._root_dir / save_folder
        if not save_folder.is_dir():
            save_folder.mkdir(parents=True)
        self._save(save_folder)

    @abstractmethod
    def _check_data(self, data):
        pass

    @abstractmethod
    def _save(self, folder):
        pass


class AImageValidator(AValidator):
    __metaclass__ = ABCMeta
    _save_ext: str

    def __init__(self, save_ext):
        self._save_ext = save_ext

    def _check_data(self, data):
        pass

    def _save(self, folder):
        for i, (img, name, cmap) in enumerate(self._validation_data):
            path = folder / '.'.join([str(i + 1) + "." + name, self._save_ext])
            plt.imsave(path, img, cmap=cmap)


def show_validation_data(validation_data, dpi=300):
    for i, img, name, cmap in enumerate(validation_data):
        plt.figure(i, dpi=dpi)
        plt.title(name)
        plt.imshow(img, cmap=cmap)
        plt.show()


def draw_regions(img, boxes, with_border=True, draw_all=False, fill_rect=False):
    img = np.ndarray.copy(img)
    if with_border:
        boxes = append_border(boxes)
    if draw_all:
        indexes = np.arange(boxes.shape[0])
    else:
        try:
            indexes = np.hstack((np.arange(5), np.arange(boxes.shape[0] // 10, boxes.shape[0], boxes.shape[0] // 10)))
        except ZeroDivisionError:
            indexes = np.arange(boxes.shape[0])
    for i, box in enumerate(boxes[indexes]):
        y, x, height, width = box
        if i == 0:
            color = (0, 255, 0)
            thickness = 3
        elif i in range(5):
            color = (0, 0, 255)
            thickness = 1
        else:
            color = (255, 0, 0)
            thickness = 1

        if fill_rect:
            thickness = -1
        cv.rectangle(img, (x, y), (x + width, y + height), color, thickness)
    return img


def make_learning_curve(clf, X, y, save_folder, number_of_points=10, train_scale=None, metric=None):
    if not train_scale:
        train_scale = np.linspace(0.1, 1.0, number_of_points)
    if not metric:
        metric = accuracy_score
    scoring = make_scorer(metric)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=clf,
        X=X,
        y=y,
        train_sizes=train_scale,
        cv=3,
        scoring=scoring,
        n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    print(f"Train accuracy: {train_mean[-1]}")
    print(f"Test accuracy: {test_mean[-1]}")
    plot_learning_curve(train_mean, train_std, test_mean, test_std, train_sizes, save_folder, metric.__name__)


def plot_learning_curve(train_mean,train_std,test_mean,test_std,train_sizes,save_folder,metric_name):
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
    plt.ylabel(metric_name)
    plt.legend(loc='lower right')
    # plt.ylim([0.8, 1.03])
    plt.tight_layout()
    plt.savefig(save_folder / f'Learning_curve_{metric_name}.png', dpi=300)
    plt.show()

