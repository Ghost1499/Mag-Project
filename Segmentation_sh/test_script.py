import itertools as it
import os
from pathlib import Path

import numpy as np

from Segmentation_sh.test_methods import test_from_dir, find_barcode
from Segmentation_sh.utils import get_files_from_dir, get_boxes_configurations, get_images_from_dir, extract_patches, \
    make_horizontal, resize_img, benchmark
from Segmentation_sh.params_config import *


@benchmark
def main():
    # test_dir = os.path.normpath("C:/Users/zgstv/OneDrive/Изображения/vend_machines")
    # test_dir = os.path.normpath("C:/Users/zgstv/OneDrive/Изображения/vend_machines2")
    # test_dir = os.path.abspath("test_data")
    # result_dir = os.path.abspath("results/binary_img/v")
    # result_dir = os.path.abspath("results/binary_img/v2")
    # result_dir = os.path.abspath("results/test")
    # test_from_dir(test_dir, result_dir)
    # find_barcode("test_data/2021-10-20_13_32_07_783.png")

    barcodes, barcodes_names = get_images_from_dir(Path("C:/Users/zgstv/OneDrive/Изображения/barcodes_full"))
    images, imgs_names = get_images_from_dir(Path("C:/Users/zgstv/OneDrive/Изображения/vend_machines"), barcodes_names)
    configs = get_boxes_configurations(min_box_area, max_box_area, n_box_sides_steps, min_sides_ratio)
    patch_sizes = configs + [(height, width) for width, height in configs]
    count_patches = 10
    negative_patches = []
    negative_patches.extend(
        it.chain.from_iterable((extract_patches(img, patch_sizes, count_patches) for img in images)))

    # провести аугментацию штрихкодов ( поворот по вертикали, горизонтали, и так, и так; поворот на несколько
    # градусов по часовой и против часовой
    sample_size = (100, 150)
    positive = np.array([resize_img(make_horizontal(barcode, sample_size), sample_size) for barcode in barcodes])
    negative = np.array([resize_img(make_horizontal(patch, sample_size), sample_size) for patch in negative_patches])
    assert positive.shape[1:] == negative.shape[1:]

    print()


if __name__ == '__main__':
    main()
