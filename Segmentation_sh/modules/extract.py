import itertools as it
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import image

from Segmentation_sh.modules.preprocess import prepare_patch, patch_from_box, apply_hog
from Segmentation_sh.modules.read import get_images_from_dir


def extract_negative_samples(path, barcodes_names, configs, count_patches_per_conf=10):
    if type(path) is not Path:
        path = Path(path)
    if not path.is_dir():
        raise Exception("Путь для считывания изображений тары не является директорией")
    images, imgs_names = get_images_from_dir(path, barcodes_names)
    patch_sizes = configs + [(height, width) for width, height in configs]
    negative_patches = []
    negative_patches.extend(
        it.chain.from_iterable((extract_patches(img, patch_sizes, count_patches_per_conf) for img in images)))

    return negative_patches


def extract_patches(img, patch_sizes, count=None):
    patches = []
    for patch_size in patch_sizes:
        patches.extend(image.extract_patches_2d(img, patch_size=patch_size, max_patches=count, random_state=0))
    return patches


def get_patches_from_boxes(img, sample_size, boxes):
    def get_patch(box):
        return prepare_patch(patch_from_box(img, box), sample_size)
    test_patches = np.apply_along_axis(get_patch, 1, boxes)
    return test_patches


def get_test_data(img, sample_size, boxes):
    test_patches = get_patches_from_boxes(img, sample_size, boxes)
    assert test_patches.shape[1:-1] == sample_size
    X_test=[]
    for i in np.ndindex(test_patches.shape[0]):
        X_test.append(apply_hog(test_patches[i]))
    X_test=np.array(X_test)
    return X_test