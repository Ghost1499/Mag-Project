import os
from pathlib import Path
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from skimage import feature
from skimage.transform import resize
from sklearn.svm import LinearSVC

from Segmentation_sh.modules.validation import draw_regions, save_validation_data, show_validation_data
from Segmentation_sh.modules.utils import benchmark, resize_img, match_orientation, patch_from_box
from Segmentation_sh.modules.regions import propose_regions, get_boxes_configurations
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
def find_barcode(img_path:Union[str,Path], show_valid_data=False, save_valid_data=True, save_folder="results"):
    validation_data = []
    if type(img_path) is not Path:
        img_path = Path(img_path)
    result_dir = Path(save_folder) / img_path.stem

    img = plt.imread(img_path)
    validation_data.append((img,'source',None))

    expected_size = 964, 1292
    img = match_orientation(img,expected_size)
    if img.shape[:2] != expected_size:
        img = resize(img, expected_size, anti_aliasing=True)
    mean = np.mean(img)
    if 0 < mean < 1:
        img = img * 255
    img = img.astype("uint8")

    configs = get_boxes_configurations(min_box_area, max_box_area, n_box_sides_steps, min_sides_ratio)
    scores, boxes, regions_number,cur_v_data = propose_regions(img, threshold,configs, alpha, min_box_area, max_box_area,delta, min_box_ratio, max_border_ratio,
                                                    rsort_key)
    if cur_v_data:
        validation_data.extend(cur_v_data)
    if boxes is None:
        return

    regions = draw_regions(img,boxes)
    validation_data.append((regions,"regions",None))
    print(f"Count of regions: {regions_number}")

    #     barcodes, barcodes_names = get_images_from_dir(Path("C:/Users/zgstv/OneDrive/Изображения/barcodes_full"))
    #     images, imgs_names = get_images_from_dir(Path("C:/Users/zgstv/OneDrive/Изображения/vend_machines"), barcodes_names)
    #     configs = get_boxes_configurations(min_box_area, max_box_area, n_box_sides_steps, min_sides_ratio)
    #     patch_sizes = configs + [(height, width) for width, height in configs]
    #     count_patches = 10
    #     negative_patches = []
    #     negative_patches.extend(
    #         it.chain.from_iterable((extract_patches(img, patch_sizes, count_patches) for img in images)))
    #
    #     # провести аугментацию штрихкодов ( поворот по вертикали, горизонтали, и так, и так; поворот на несколько
    #     # градусов по часовой и против часовой
    #     sample_size = (100, 150)
    #     positive = np.array([resize_img(make_horizontal(barcode, sample_size), sample_size) for barcode in barcodes])
    #     negative = np.array([resize_img(make_horizontal(patch, sample_size), sample_size) for patch in negative_patches])
    #     assert positive.shape[1:] == negative.shape[1:]
    #     np.save("positive",positive)
    #     np.save("negative",negative)
    #
    # X_train = np.array([feature.hog(im,multichannel=True)
    #                     for im in tqdm(it.chain(positive,
    #                                             negative))])
    #
    # y_train = np.zeros(X_train.shape[0])
    # y_train[:positive.shape[0]] = 1

    # grid = GridSearchCV(LinearSVC(dual=False), {'C': [1.0, 2.0, 4.0, 8.0]}, cv=3)
    # grid.fit(X_train, y_train)
    # print(grid.best_score_)
    # model = grid.best_estimator_
    # X_train_filename = "X_train.npy"
    # y_train_filename = "y_train.npy"
    # X_train = np.load(X_train_filename)
    # y_train = np.load(y_train_filename)
    #
    # model = LinearSVC(C=4.0, dual=False)
    # model.fit(X_train, y_train)

    X_train_filename = "X_train.npy"
    y_train_filename = "y_train.npy"
    X_train = np.load(X_train_filename)
    y_train = np.load(y_train_filename)

    model = LinearSVC(C=4.0, dual=False)
    model.fit(X_train, y_train)
    sample_size = (100, 150)

    def get_patch(box):
        return resize_img(match_orientation(patch_from_box(img, box), sample_size), sample_size)

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
    labels = model.predict(X_test)
    # number of barcode detections from all the patches in the image
    print(labels.sum())

    result_regions = draw_regions(img,boxes[labels==1],draw_all=True,fill_rect=True)
    validation_data.append((result_regions,"result regions",None))

    if save_valid_data:
        save_validation_data(validation_data,save_folder)
    if show_valid_data:
        show_validation_data(validation_data)
    print()
