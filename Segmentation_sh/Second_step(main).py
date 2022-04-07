# Импорт всех необходимых компонентов и утилит
import numpy as np
import cv2
import os
from cv2 import createCLAHE
import numpy as np
import random
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import feature, morphology
from Segmentation_sh.utils import propose_regions, test_model

# from google.colab.patches import cv2_imshow
filename = "test_data\\2021-10-20_13_32_07_783.png"
I = plt.imread(filename)
# a = filename[0:len(filename) - 4]
alpha = 0.65  # расчет смещений боксов по x и по y c перекрытием overloapratio=alpha
minbarea = 8000
maxbarea = 20000  # площади региона
n_sc = 5  # число шагов поиска по масштабу
minratio = 1 / 2  # минимальное соотношение сторон (варианты 1/4,1/3,1/2)
delta = 10  # ширина полосы вдоль внешней границы бокса
por = 200  # порог баниризации
viz = True  # визаулизация контурной обработки в папку valid_data
num_proposal = 2  # количество предложений лучших регионов
# по показателю максимума ib0/S0*(1-ib1/S1)

# Формирование изображений масок
vscore_select, boxes_select, nb = propose_regions(I, por, alpha, minbarea, maxbarea, minratio, n_sc, delta, viz)
# Тестирование по одиночному изображению
mask = np.zeros((I.shape[0], I.shape[1], 3))
if num_proposal > boxes_select.shape[0]:
    num_proposal = boxes_select.shape[0]
for i in range(num_proposal):
    x1 = boxes_select[i, 1]
    y1 = boxes_select[i, 0]
    x2 = x1 + boxes_select[i, 3]
    y2 = y1
    x3 = x1
    y3 = y1 + boxes_select[i, 2]
    x4 = x1 + boxes_select[i, 3]
    y2 = y1 + boxes_select[i, 2]
    mask[y1:y2, x1, :] = 255
    mask[y2, x1:x2, :] = 255
    mask[y1, x1:x2, :] = 255
    mask[y1:y2, x2, :] = 255
    img = I + mask
filename = 'valid_data/Res.png'
cv2.imwrite(filename, img)
# Тестирование изображений из папки
path = 'test_data'
path_new = 'test_data_new'
test_model(path, path_new, por, alpha, minbarea, maxbarea, minratio, n_sc, delta, num_proposal)
