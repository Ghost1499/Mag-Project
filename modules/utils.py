import os
import time
from pathlib import Path

import cv2 as cv
import numpy as np


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print('[*] Время выполнения: {} секунд.'.format(end - start))
        return res

    return wrapper


def is_horizontal(size):
    return size[0] < size[1]


def check_image(img: np.ndarray):
    if img is None:
        raise ValueError("Изображение None")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Тип входного аргумента {type(img)}, а не Numpy Array")
    if 2 > img.ndim > 3:
        raise ValueError(f"Входной аргумент img имеет неподходящее число измерений: {img.ndim}")
    if img.ndim == 3:
        if img.shape[2] not in (3, 4):
            raise ValueError(f"Входной аргумент img имеет неподходящее число каналов: {img.shape[2]}")


# noinspection PyPep8Naming
def is_RGB_format(img: np.ndarray):
    if img.ndim == 3 and img.shape[2] == 3:
        return True
    return False


def is_grayscale(img: np.ndarray):
    if img.ndim == 2:
        return True
    return False


def image2gray(img: np.ndarray):
    if is_RGB_format(img):
        return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def images_in_dir_generator(source_dir: [str, Path], result_dir: [str, Path]):
    if type(source_dir) == str:
        source_dir = Path(source_dir)
    if type(result_dir) == str:
        result_dir = Path(result_dir)
    if not source_dir.is_dir():
        raise Exception("Тестовая директория не существует")
    if not result_dir.is_dir():
        result_dir.mkdir(parents=True)
    for root, dirs, files in os.walk(str(source_dir)):
        for file in files:
            file_path = Path(os.path.join(root, file))
            ext = file_path.suffix
            possible_exts = ".jpg", ".jpeg", ".png", ".bmp"
            if ext in possible_exts:
                cur_res_dir = Path(root).relative_to(source_dir)
                save_dir = result_dir / cur_res_dir
                yield file_path, save_dir
