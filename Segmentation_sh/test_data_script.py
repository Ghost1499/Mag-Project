import cv2
import numpy as np
from matplotlib import pyplot as plt

from Segmentation_sh.my_fun_sh1 import test_model, propose_regions

alpha = 0.65  # расчет смещений боксов по x и по y c перекрытием overloapratio=alpha
minbarea = 8000
maxbarea = 20000  # площади региона
n_sc = 5  # число шагов поиска по масштабу
minratio = 1 / 2  # минимальное соотношение сторон (варианты 1/4,1/3,1/2)
delta = 10  # ширина полосы вдоль внешней границы бокса
threshold = 120  # порог баниризации
viz = True  # визаулизация контурной обработки в папку valid_data
num_proposal = 2  # количество предложений лучших регионов
# по показателю максимума ib0/S0*(1-ib1/S1)
path = 'test_data'
path_new = 'test_data_new'

test_img = plt.imread(
    # "test_data/2021-10-20_13_32_07_783.png"
    # "test_data/2021-10-20_13_41_54_822.png"
    # "test_data/2021-10-20_13_52_19_237.png"
    # "test_data/2021-10-20_14_15_17_418.png"

    # "C:/Users/zgstv/OneDrive/Изображения/vend_machines2/IMG_20211029_163500.jpg"
)
mean = np.mean(test_img)
if 0 < mean < 1:
    test_img = test_img * 255
test_img = test_img.astype("uint8")
draw_border = lambda img: img
vscore, boxes, boxes_number = propose_regions(test_img, threshold, alpha, minbarea, maxbarea, minratio, n_sc, delta,
                                              False, False)
for box in boxes[:2]:
    y, x, height, width = box
    cv2.rectangle(test_img, (x, y), (x + width, y + height), (255,0,0),1)
plt.figure(0,dpi=300)
plt.imshow(test_img)
plt.show()
plt.imsave("regions5.jpg", test_img)
# test_model(path, path_new, threshold, alpha, minbarea, maxbarea, minratio, n_sc, delta, num_proposal)
