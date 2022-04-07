import os
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import feature

# from Segmentation_sh.params_config import *


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print('[*] Время выполнения: {} секунд.'.format(end - start))
        return res

    return wrapper


def propose_regions(img: np.ndarray, threshold: int, alpha: float, minbarea: int, maxbarea: int, minratio: float,
                    n_sc: int, delta: int, save_results: bool,
                    show_results: bool, save_folder) -> [Tuple[np.ndarray, np.ndarray, int], Tuple[None, None, int]]:
    """
    :param img: изображение
    :param threshold: порог для бинаризации внутри edge_boxes
    :param alpha: расчет смещений боксов по x и по y c перекрытием overloapratio=alpha
    :param minbarea: минимальная площадь ?
    :param maxbarea: максимальная площадь
    :param minratio: минимальное соотношение сторон
    :param n_sc: количество шагов поиска по масштабу ?
    :param delta: сдвиг окна
    :param save_results: сохранять промежуточные изображения-результаты в файл (папка results)
    :param show_results: показывать промежуточные изображения-результаты
    :param save_folder: папка для сохранения промежуточных результов
    :return: vscore_select, boxes_select, boxes_count

    Формирование предложений по областям интереса using a variant of the Edge Boxes algorithm.

    """
    # .
    # alpha = 0.65; #расчет смещений боксов по x и по y c перекрытием overloapratio=alpha
    # #beta  = 0.75; # for each bbox i, suppress all surrounded bbox j where j>i and overlap
    # #ratio is larger than overlapThreshold (beta)
    # minbarea=1000; maxbarea=700000; #площади региона
    # delta=10 #ширина полосы вдоль внешней границы бокса
    # step_sc=(max_sc-min_sc)/n_sc; #шаг поиска по масштабу (по площади)
    # step_vr=minratio; #шаг поиска по соотношению  сторон
    if img is None:
        raise Exception("Изображение None")
    nx = img.shape[1]
    ny = img.shape[0]
    # параметры поиска областей
    min_sc = np.sqrt(minbarea)
    max_sc = np.sqrt(maxbarea)
    if min_sc == max_sc:
        max_sc = 1.25 * min_sc
        n_sc = 1
    step_sc = (max_sc - min_sc) / n_sc  # шаг поиска по масштабу (кореннь из площади)
    maxratio = 1 / minratio
    # step_vr=minratio #шаг поиска по соотношению  сторон

    # Convert image to uint8
    img = img.astype('uint8')
    # threshold=200; #порог бинаризации после вычисления градиентов
    img_prepared = edge_map(img, threshold, save_results, show_results,
                            save_folder)  # контурный анализ и обработка изображений

    # filename='img_CC.png'
    # cv2.imwrite(filename,img_prepared)
    CC = img_prepared / 255
    int_C, sqsum, tilted = cv2.integral3(CC)
    # filename='int_CC.png'
    # cv2.imwrite(filename,int_C)
    # Организация начального поиска областей
    va = np.arange(min_sc, max_sc + step_sc, step_sc)
    lva = len(va)
    va = va ** 2
    if minratio == 1 / 3:
        vr = np.array([1 / 3, 2 / 3, 1])
    elif minratio == 1 / 2:
        vr = np.array([1 / 2, 1])
    elif minratio == 1 / 4:
        vr = np.array([1 / 4, 1 / 2, 3 / 4, 1])
    # vr=np.arange(minratio,maxratio,step_vr); 
    lvr = len(vr)
    boxes_count = 0
    step_x = np.zeros((lva, lvr), np.float32)
    step_y = np.zeros((lva, lvr), np.float32)
    boxes = []  # генерируемые расположения и конфигура
    vscore = []
    for iva in range(lva):  # поиск по масштабу
        for ivr in range(lvr):  # поиск по соотношению сторон бокса
            # определение конфигурации бокса
            bx = np.floor((va[iva] / vr[ivr]) ** 0.5)
            by = np.floor((va[iva] * vr[ivr]) ** 0.5)
            if by < min_sc:
                by = min_sc
            if by > max_sc:
                by = max_sc
            if bx < min_sc:
                bx = min_sc
            if bx > max_sc:
                bx = max_sc
            bx = bx.astype('uint32')
            by = by.astype('uint32')
            # расчет смещений боксов фиксированной конфигурации bx,by по x и по y c перекрытием overloapratio=alpha
            step_x[iva, ivr] = np.floor(bx * (1 - alpha) / (1 + alpha))
            step_y[iva, ivr] = np.floor(by * (1 - alpha) / (1 + alpha))
            x = np.arange(delta, nx - bx - delta, step_x[iva, ivr])
            Kx = len(x)
            y = np.arange(delta, ny - by - delta, step_y[iva, ivr])
            Ky = len(y)
            x = np.floor(x).astype('uint32')
            y = np.floor(y).astype('uint32')
            S0 = bx * by
            S1 = (bx + 2 * delta) * (by + 2 * delta) - S0
            for kx in range(Kx):
                for ky in range(Ky):
                    box = [y[ky], x[kx], by, bx]
                    ib0 = int_C[y[ky], x[kx]] + int_C[y[ky] + by, x[kx] + bx] - int_C[y[ky] + by, x[kx]] - int_C[
                        y[ky], x[kx] + bx]  # с учетом того что в массива
                    ib1 = int_C[y[ky] - delta, x[kx] - delta] + int_C[y[ky] + by + delta, x[kx] + bx + delta] - int_C[
                        y[ky] + by + delta, x[kx] - delta] - int_C[y[ky] - delta, x[kx] + bx + delta]
                    ib1 = ib1 - ib0
                    if minbarea < ib0 < maxbarea:
                        # print(ib0,ib0/S0,ib1/S1)
                        if ib0 / S0 > 0.25 and ib1 / S1 < 0.25:  # в боксе есть крупные объекты и вдоль границ бокса мало объектов
                            vs = [ib0, ib0 / S0, ib1 / S1, ib0 / S0 * (1 - ib1 / S1)]
                            vscore.append(vs)
                            boxes.append(box)
                            boxes_count += 1
    if boxes_count == 0:
        return None, None, 0
    else:
        boxes = np.array(boxes, np.uint32)
        vscore_ = np.array(vscore)
        ef1 = np.argsort(-vscore_[:, 3])  # сортировка в порядке убывания площади объекта
        vscore_select = vscore_[ef1, :]
        boxes_select = boxes[ef1, :]
        return vscore_select, boxes_select, boxes_count


def edge_map(img, threshold, save_results=False, show_results=False, save_folder=None):
    # img=img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    grad_x = np.absolute(grad_x)
    kernel_recth = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    img_closed_rectv = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel_recth)

    grad_y = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    grad_y = np.absolute(grad_y)
    kernel_recthv = np.transpose(kernel_recth, axes=(1, 0))
    img_closed_recth = cv2.morphologyEx(grad_y, cv2.MORPH_CLOSE, kernel_recthv)

    # grad_subtract = cv2.subtract(grad_x, grad_y)  # вычитание градиентов
    # grad_subtract = np.absolute(grad_subtract)

    # kernel_round = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # grad_subtract_bin = cv2.morphologyEx(grad_subtract_bin, cv2.MORPH_OPEN, kernel_round, iterations=1)

    # img_closed_rectv = cv2.erode(img_closed_rectv, None, iterations=4)
    # img_closed_rectv = cv2.dilate(img_closed_rectv, None, iterations=4)

    kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    img_opened_square = cv2.morphologyEx(img_closed_rectv, cv2.MORPH_OPEN, kernel_square)
    # img_opened_square = cv2.erode(img_opened_square, None, iterations=4)
    # img_opened_square = cv2.dilate(img_opened_square, None, iterations=4)
    img_opened_square = img_opened_square / img_opened_square.max() * 255
    img_opened_square = img_opened_square.astype("uint8")
    (_, bin) = cv2.threshold(img_opened_square, threshold, 255, cv2.THRESH_BINARY, )

    if save_results or show_results:
        imgs = [img_gray, grad_x, grad_y,  # grad_subtract, grad_subtract_bin,
                img_closed_rectv, img_closed_recth,
                img_opened_square, bin]
        names = ["gray", "grad x", 'grad y',  # "grad subtract", "grad subtract binary",
                 "closing horizontal rect 2x16",
                 "closing vertical rect 16x2",
                 "opening square 8x8", "binary"]
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
    return img_opened_square





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

#

#       edge_map(test_img, threshold, show_test_results=False, save_test_results=True)
#     test_img("test_data/2021-10-20_13_32_07_783.png", save_folder="results/2021-10-20_13_32_07_783")
