import os
import time
from pathlib import Path
from typing import Tuple, Optional
import itertools as it

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import feature

rsort_keys = {"area": 0, "sum_area_ratio": 1, "border_sum_area_ratio": 2, "box_border_ratio": 3}


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print('[*] Время выполнения: {} секунд.'.format(end - start))
        return res

    return wrapper


def _image_region_sum(integral, y, x, height, width):
    try:
        region_sum = integral[y, x] - integral[y + height, x] - integral[y, x + width] + \
                     integral[y + height, x + width]
        return region_sum
    except IndexError:
        raise ValueError(
            f"Один или несколько параметров (y={y},x={x},height={height},width={width} находятся за предлами "
            f"интеграла изображения с размером {integral.shape}")


def get_boxes_configurations(min_box_area, max_box_area, n_box_side_steps, min_sides_ratio):
    min_box_side = np.sqrt(min_box_area)
    max_box_side = np.sqrt(max_box_area)
    if min_box_side == max_box_side:
        max_box_side = 1.25 * min_box_side
        n_box_side_steps = 1
    box_side_step = (max_box_side - min_box_side) / n_box_side_steps  # шаг поиска по масштабу (корень из
    # площади)

    # Организация начального поиска областей
    box_areas = np.arange(min_box_side, max_box_side + box_side_step, box_side_step) ** 2
    box_ratios = np.arange(min_sides_ratio, 1 + min_sides_ratio, min_sides_ratio)
    configurations = []  # генерируемые расположения и конфигура
    for area_i, area in enumerate(box_areas):  # поиск по масштабу
        for ratio_i, ratio in enumerate(box_ratios):  # поиск по соотношению сторон бокса
            # определение конфигурации бокса
            box_width = np.floor((area / ratio) ** 0.5)
            box_height = np.floor((area * ratio) ** 0.5)
            if box_height < min_box_side:
                box_height = min_box_side
            if box_height > max_box_side:
                box_height = max_box_side
            if box_width < min_box_side:
                box_width = min_box_side
            if box_width > max_box_side:
                box_width = max_box_side
            box_width = box_width.astype('uint32')
            box_height = box_height.astype('uint32')
            configurations.append((box_width, box_height))

    return configurations


def propose_regions(img: np.ndarray, threshold: int, alpha: float, min_box_area: int, max_box_area: int,
                    min_sides_ratio: float, n_box_side_steps: int, delta: int, min_box_ratio: float,
                    max_border_ratio: float, rsort_key: str,
                    save_results: bool, show_results: bool, save_folder: Path) \
        -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    :param img: изображение
    :param threshold: порог для бинаризации внутри edge_boxes
    :param alpha: расчет смещений боксов по X и по Y c перекрытием overloapratio=alpha. Допустимые значения [0,1), где 0 перекрытие отсутствует.
    :param min_box_area: минимальная площадь
    :param max_box_area: максимальная площадь
    :param min_sides_ratio: минимальное соотношение сторон
    :param n_box_side_steps: количество шагов поиска по масштабу
    :param delta: сдвиг окна
    :param min_box_ratio: минимальное значение соотношения интеграла бокса и его площади
    :param max_border_ratio: максимальное значение соотношения интеграла рамки и её площади
    :param rsort_key: атрибут, по которому сортируется массив результатов
    :param save_results: сохранять промежуточные изображения-результаты в файл (папка results)
    :param show_results: показывать промежуточные изображения-результаты
    :param save_folder: папка для сохранения промежуточных результов
    :return: vscore_select, boxes_select, boxes_count

    Формирование предложений по областям интереса using a variant of the Edge Boxes algorithm.

    """
    # alpha = 0.65; #расчет смещений боксов по X и по Y c перекрытием overloapratio=alpha
    # #beta  = 0.75; # for each bbox i, suppress all surrounded bbox j where j>i and overlap
    # #ratio is larger than overlapThreshold (beta)
    # min_box_area=1000; max_box_area=700000; #площади региона
    if img is None:
        raise ValueError("Изображение None")
    if alpha < 0 or alpha >= 1:
        raise ValueError(f"Параметр alpha={alpha} находится вне допустимого диапазона", alpha)

    img_width = img.shape[1]
    img_height = img.shape[0]
    # параметры поиска областей

    img = img.astype('uint8')
    img_prepared_barcode_h, img_prepared_barcode_v = edge_map(img, threshold, save_results, show_results,
                                                              save_folder)  # контурный анализ и обработка изображений

    img_prepared_float_scaled_h = img_prepared_barcode_h / 255
    img_prepared_float_scaled_v = img_prepared_barcode_v / 255
    img_integral_sum_h, _, _ = cv2.integral3(img_prepared_float_scaled_h)
    img_integral_sum_v, _, _ = cv2.integral3(img_prepared_float_scaled_v)
    plt.imsave(save_folder / "img_integral_sum_h.jpg", img_integral_sum_h)
    plt.imsave(save_folder / "img_integral_sum_v.jpg", img_integral_sum_v)
    # Организация начального поиска областей
    configurations = get_boxes_configurations(min_box_area, max_box_area, n_box_side_steps, min_sides_ratio)
    boxes_count = 0
    step_x = np.zeros(len(configurations), np.float32)
    step_y = np.zeros(len(configurations), np.float32)
    boxes = []  # генерируемые расположения и конфигурации
    scores = []
    for conf_number, conf in enumerate(configurations):  # поиск по масштабу
        # определение конфигурации бокса
        # расчет смещений боксов фиксированной конфигурации box_width, box_height по X и по Y c перекрытием
        # overloapratio=alpha
        box_width, box_height = conf
        step_x[conf_number] = np.floor(box_width * (1 - alpha) / (1 + alpha))
        step_y[conf_number] = np.floor(box_height * (1 - alpha) / (1 + alpha))
        X_barcode_h = np.floor(np.arange(delta, img_width - box_width - delta, step_x[conf_number])).astype(
            'uint32')
        X_barcode_v = np.floor(np.arange(delta, img_width - box_height - delta, step_y[conf_number])).astype(
            "uint32")
        Y_barcode_h = np.floor(np.arange(delta, img_height - box_height - delta, step_y[conf_number])).astype(
            "uint32")
        Y_barcode_v = np.floor(np.arange(delta, img_height - box_width - delta, step_x[conf_number])).astype(
            "uint32")
        XY_directions = [(X_barcode_h, Y_barcode_h, box_width, box_height, img_integral_sum_h),
                         (X_barcode_v, Y_barcode_v, box_height, box_width, img_integral_sum_v)]
        box_area_final = box_width * box_height
        border_area = (box_width + 2 * delta) * (box_height + 2 * delta) - box_area_final
        for X, Y, cur_box_width, cur_box_height, img_integral_sum in XY_directions:
            for x in X:
                for y in Y:
                    box = [y, x, cur_box_height, cur_box_width]
                    box_sum = _image_region_sum(img_integral_sum, y, x, cur_box_height, cur_box_width)
                    border_sum = _image_region_sum(img_integral_sum, y - delta, x - delta, cur_box_height + delta,
                                                   cur_box_width + delta)
                    border_sum -= box_sum
                    if min_box_area < box_sum < max_box_area:  # ???!!!
                        if box_sum / box_area_final > min_box_ratio and border_sum / border_area < max_border_ratio:
                            # в боксе есть крупные объекты и вдоль границ бокса мало объектов
                            score = [box_sum, box_sum / box_area_final, border_sum / border_area,
                                     box_sum / box_area_final * (1 - border_sum / border_area)]
                            scores.append(score)
                            box.append(delta)
                            boxes.append(box)
                            boxes_count += 1
    if boxes_count == 0:
        return None, None, 0
    else:
        boxes = np.array(boxes, np.uint32)
        scores = np.array(scores)
        indexes_sorted = np.argsort(
            -scores[:,
             rsort_keys[rsort_key]])  # сортировка в порядке убывания соотношения box_sum / box_area_final * (1 -
        # border_sum / border_area)
        scores_selected = scores[indexes_sorted, :]
        boxes_selected = boxes[indexes_sorted, :]
        return scores_selected, boxes_selected, boxes_count


def edge_map(img, threshold, save_results=False, show_results=False, save_folder=None):
    # img=img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    grad_x = np.absolute(grad_x)
    kernel_recth = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    grad_x_closed_rect = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel_recth)

    grad_y = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    grad_y = np.absolute(grad_y)
    kernel_recthv = np.transpose(kernel_recth, axes=(1, 0))
    grad_y_closed_rect = cv2.morphologyEx(grad_y, cv2.MORPH_CLOSE, kernel_recthv)

    # grad_subtract = cv2.subtract(grad_x, grad_y)  # вычитание градиентов
    # grad_subtract = np.absolute(grad_subtract)

    # kernel_round = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # grad_subtract_bin = cv2.morphologyEx(grad_subtract_bin, cv2.MORPH_OPEN, kernel_round, iterations=1)

    # grad_x_closed_rect = cv2.erode(grad_x_closed_rect, None, iterations=4)
    # grad_x_closed_rect = cv2.dilate(grad_x_closed_rect, None, iterations=4)

    kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    grad_x_opened_square = cv2.morphologyEx(grad_x_closed_rect, cv2.MORPH_OPEN, kernel_square)
    grad_y_opened_square = cv2.morphologyEx(grad_y_closed_rect, cv2.MORPH_OPEN, kernel_square)
    # grad_x_opened_square = cv2.erode(grad_x_opened_square, None, iterations=4)
    # grad_x_opened_square = cv2.dilate(grad_x_opened_square, None, iterations=4)

    # !!!!!!!!!!!!!!!! пофиксить нормировку (после открытия поднимается изображение "поднимается")
    # grad_x_opened_square = grad_x_opened_square / (2^64) * 255
    # grad_x_opened_square = grad_x_opened_square.astype("uint8")

    # grad_y_opened_square = grad_y_opened_square / (2^64) * 255
    # grad_y_opened_square = grad_y_opened_square.astype("uint8")

    (_, grad_x_bin) = cv2.threshold(grad_x_opened_square, threshold, 255, cv2.THRESH_BINARY)
    (_, grad_y_bin) = cv2.threshold(grad_y_opened_square, threshold, 255, cv2.THRESH_BINARY)

    if save_results or show_results:
        imgs = [img_gray,
                grad_x, grad_y,
                grad_x_closed_rect, grad_y_closed_rect,
                grad_x_opened_square, grad_y_opened_square,
                grad_x_bin, grad_y_bin]
        names = ["gray",
                 "grad x", 'grad y',  # "grad subtract", "grad subtract binary",
                 "grad x closed by rectangle 2x16", "grad x closed by rectangle 16x2",
                 "grad x opening square 8x8", "grad y opening square 8x8",
                 "grad x binary", "grad y binary"]

        assert len(imgs) == len(names), "Длина списка тестовых изображений не равна длине списка имен тестовых " \
                                        "изображений"
        cmap = "gray"
        if save_results:
            if not save_folder:
                raise Exception("Папка для сохранения результатов не задана")
            save_folder = Path(save_folder)

        for i, img, name in zip(range(len(imgs)), imgs, names):
            if save_results:
                fmt = 'jpg'
                path = save_folder / '.'.join([str(i + 1) + "." + name, fmt])
                # os.img_path.join(save_folder, '.'.join([name, fmt]))
                if not save_folder.is_dir():
                    save_folder.mkdir(parents=True)
                # if not os.img_path.isdir(save_folder):
                #     raise Exception(f"Папки для сохранения результатов с именем {save_folder} не существует")
                plt.imsave(path, img, cmap=cmap)
            if show_results:
                dpi = 300
                plt.figure(i, dpi=dpi)
                plt.title(name)
                plt.imshow(img, cmap=cmap)
                plt.show()
    return grad_x_bin, grad_y_bin


def mrglob(path: Path, *patterns):
    return it.chain.from_iterable(path.rglob(pattern) for pattern in patterns)


def get_barcodes_from_dir(dir: Path):
    if not dir.is_dir():
        raise Exception("Директория не существует")
    patterns = ["*.jpeg", "*.jpg", "*.png", "*.bmp"]
    paths = list(mrglob(dir, *patterns))
    names = [path.stem for path in paths]
    barcodes = [plt.imread(dir / path) for path in paths]
    return barcodes, names

# def test_walk(test_dir):
#     for root, dirs, files in os.walk(test_dir):
#         print(root)
#         for d in dirs:
#             print(d)
#         print("_____________________")
#         for f in files:
#             print(os.img_path.join(root,f))
#         print("*********************")


# def my_overlopratio(boxA,boxB):
# #   Определение матрицы мер попарного пересечения боксов
# #На входе совокупность боксов с координатами вершин длинами сторон
#     # left top corner
#     x1boxA = boxA[:,0];    y1boxA = boxA[:,1];
#     x1boxB = boxB[:,0];    y1boxB = boxB[:,1];
#     #right bottom corner
#     x2boxA = x1boxA + boxA[:, 2];    y2boxA = y1boxA + boxA[:, 3]; 
#     x2boxB = x1boxB + boxB[:, 2];    y2boxB = y1boxB + boxB[:, 3];

#     #area of the bounding box
#     areaA = np.multiply(boxA[:, 2], boxA[:, 3])
#     areaB = np.multiply(boxB[:, 2], boxB[:, 3])

#     iou = np.zeros((boxA.shape[0],boxB.shape[0]),np.float32);
#     if boxA.shape[0]!=0:
#         if boxB.shape[0]!=0:
#             for m in range(boxA.shape[0]):
#                 for n in range(boxB.shape[0]):
#                     # print(n)
#                     # print(m)
#                     #compute the corners of the intersect
#                     x1 = np.maximum(x1boxA[m], x1boxB[n])
#                     y1 = np.maximum(y1boxA[m], y1boxB[n])
#                     x2 = np.minimum(x2boxA[m], x2boxB[n])
#                     y2 = np.minimum(y2boxA[m], y2boxB[n])
#                     #пропуск, если нет пресечения
#                     w = x2 - x1
#                     h = y2 - y1
#                     if w > 0 and h > 0:
#                         intersectAB = w * h
#                         r= intersectAB/(areaA[m]+areaB[n]-intersectAB)
#                         iou[m,n]=r
#                     else:
#                         iou[m,n]=0
#     return iou
# #Пример 
# boxA=np.zeros((2,4),np.float32)  
# boxB=np.zeros((2,4),np.float32) 
# boxA[0,0]=1;   boxA[0,1]=1; boxA[0,2]=10;   boxA[0,3]=10;
# boxB[0,0]=1;   boxB[0,1]=1; boxB[0,2]=10;   boxB[0,3]=10;        

# boxA[1,0]=2;   boxA[1,1]=2; boxA[1,2]=10;   boxA[1,3]=10;
# boxB[1,0]=3;   boxB[1,1]=4; boxB[1,2]=22;   boxB[1,3]=22; 
# iou=my_overlopratio(boxA,boxB)
