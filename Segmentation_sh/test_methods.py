import functools
import os
from os import PathLike
from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import feature
from skimage.transform import resize, rotate
from sklearn.svm import LinearSVC
from tqdm import tqdm

from Segmentation_sh.utils import propose_regions, benchmark, resize_img, make_horizontal, patch_from_box
from Segmentation_sh.params_config import *


@benchmark
def test_from_dir(test_dir: [str, Path], result_dir: [str, Path]):
    if type(test_dir) == str:
        test_dir = Path(test_dir)
    if type(result_dir) == str:
        result_dir = Path(result_dir)
    if not test_dir.is_dir():
        raise Exception("Тестовая директория не существует")
    # if not os.listdir(test_dir):
    #     raise Exception("Тестовая директория пустая")
    if not result_dir.is_dir():
        result_dir.mkdir(parents=True)
    for root, dirs, files in os.walk(str(test_dir)):
        for file in files:
            path = Path(os.path.join(root, file))
            ext = path.suffix
            possible_exts = ".jpg", ".jpeg", ".png", ".bmp"
            if ext in possible_exts:
                cur_res_dir = Path(root).relative_to(test_dir)
                save_dir = result_dir / cur_res_dir
                find_barcode(path, save_folder=save_dir)


@benchmark
def find_barcode(img_path, show_test_results=False, save_test_results=True, save_folder="results"):
    img = plt.imread(img_path)
    expected_size = 964, 1292

    def is_horizontal(size):
        return size[0] < size[1]

    if is_horizontal(expected_size) != is_horizontal(img.shape[:2]):
        img = img.transpose(1, 0, 2)
        # img = rotate(img,90,resize=True)

        # rows,cols = img.shape[:2]
        # m = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
        # img = cv.warpAffine(img, m, (cols, rows))
    if img.shape[:2] != expected_size:
        img = resize(img, expected_size, anti_aliasing=True)
    mean = np.mean(img)
    if 0 < mean < 1:
        img = img * 255
    img = img.astype("uint8")

    path_p = Path(img_path)
    result_dir = Path(save_folder) / path_p.stem
    scores, boxes, regions_number = propose_regions(img, threshold, alpha, min_box_area, max_box_area, min_sides_ratio,
                                                    n_box_sides_steps, delta, min_box_ratio, max_border_ratio,
                                                    rsort_key,
                                                    save_test_results,
                                                    show_test_results, result_dir)
    if boxes is not None and (save_test_results or show_test_results):
        boxes = np.array([[box[0] - box[4], box[1] - box[4], box[2] + 2 * box[4], box[3] + 2 * box[4]] for box in
                          boxes])
        for i, box in enumerate(boxes[:5]):
            y, x, height, width = box
            if i == 0:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (0, 0, 255)
                thickness = 1
            cv.rectangle(img, (x, y), (x + width, y + height), color, thickness)

        for i, box in enumerate(boxes[np.arange(boxes.shape[0] // 10, boxes.shape[0], boxes.shape[0] // 10)]):
            y, x, height, width = box
            color = (255, 0, 0)
            cv.rectangle(img, (x, y), (x + width, y + height), color, 1)
        if show_test_results:
            plt.figure(0, dpi=300)
            plt.imshow(img)
            plt.show()
        if save_test_results:
            plt.imsave(result_dir / "regions.jpg", img)
        print(f"Count of regions: {regions_number}")
    if boxes is None:
        return

    X_train_filename = "X_train.npy"
    y_train_filename = "y_train.npy"
    X_train = np.load(X_train_filename)
    y_train = np.load(y_train_filename)

    model = LinearSVC(C=4.0, dual=False)
    model.fit(X_train, y_train)
    sample_size = (100, 150)

    def get_patch(box):
        return resize_img(make_horizontal(patch_from_box(img, box), sample_size), sample_size)

    # preprocess = functools.reduce(lambda x, y : y(x), [make_horizontal,resize_img], initial_value)
    test_patches = np.apply_along_axis(get_patch, 1, boxes)
    assert test_patches.shape[1:-1] == sample_size

    def apply_hog(patch):
        return feature.hog(patch, channel_axis=2)

    # X_test = np.array([feature.hog(patch,channel_axis=2) for patch in tqdm(test_patches)])
    # X_test = np.apply_along_axis(apply_hog, (1, 2, 3), test_patches)
    X_test=[]
    for i in np.ndindex(test_patches.shape[0]):
        X_test.append( apply_hog(test_patches[i]))
    X_test=np.array(X_test)
    # test = np.array([resize_img(make_horizontal(box, sample_size), sample_size) for box in boxes])
    labels = model.predict(X_test)
    # number of barcode detections from all the patches in the image
    print(labels.sum())

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')

    for y,x,height,width,border in boxes[labels == 1][0::10]:
        ax.add_patch(plt.Rectangle((x-border, y-border), width+border, height+border, edgecolor='red',
                                   alpha=0.3, lw=2, facecolor='none'))
    fig.show()
    print()
