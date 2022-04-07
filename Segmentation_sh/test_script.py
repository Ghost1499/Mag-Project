import os

from Segmentation_sh.test_data import test_from_dir, find_barcode

if __name__ == '__main__':
    # test_dir = os.path.normpath("C:/Users/zgstv/OneDrive/Изображения/vend_machines")
    # test_dir = os.path.normpath("C:/Users/zgstv/OneDrive/Изображения/vend_machines2")
    test_dir = os.path.abspath("test_data")
    # result_dir = os.path.abspath("results/binary_img/v")
    # result_dir = os.path.abspath("results/binary_img/v2")
    result_dir = os.path.abspath("results/test")
    test_from_dir(test_dir, result_dir)
    # find_barcode("test_data/2021-10-20_13_32_07_783.png")