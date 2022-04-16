from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from Segmentation_sh.modules.regions import append_border


def save_validation_data(validation_data, save_folder: Union[str, Path], save_ext="jpg"):
    if not save_folder:
        raise Exception("Папка для сохранения результатов не задана")
    if type(save_folder) is not Path:
        save_folder = Path(save_folder)

    for i, (img, name, cmap) in enumerate(validation_data):
        path = save_folder / '.'.join([str(i + 1) + "." + name, save_ext])
        if not save_folder.is_dir():
            save_folder.mkdir(parents=True)
        plt.imsave(path, img, cmap=cmap)


def show_validation_data(validation_data, dpi=300):
    for i, img, name, cmap in enumerate(validation_data):
        plt.figure(i, dpi=dpi)
        plt.title(name)
        plt.imshow(img, cmap=cmap)
        plt.show()


def draw_regions(img,boxes, with_border=True,draw_all = False,fill_rect=False):
    img = np.ndarray.copy(img)
    if with_border:
        boxes = append_border(boxes)
    if draw_all:
        indexes = np.arange(boxes.shape[0])
    else:
        indexes =np.hstack((np.arange(5),np.arange(boxes.shape[0] // 10, boxes.shape[0], boxes.shape[0] // 10)+5))
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