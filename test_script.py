import itertools as it
import json
import os
from pathlib import Path
# from pickle import load
from typing import Union

import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
from joblib import load
from matplotlib import pyplot as plt
from skimage import transform
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import LinearSVC
from tqdm import tqdm

from modules.barcode_decode import save
from modules.extract import get_test_data
from modules.preprocess import prepare_patch, apply_hog, \
    augment_barcode, prepare_image, patch_from_box
from modules.read import get_images_from_dir, get_files_from_dir
from modules.regions import get_boxes_configurations, propose_regions
from modules.utils import benchmark
from modules.validation import draw_regions, save_validation_data, show_validation_data
from params_config import min_box_area, max_box_area, n_box_sides_steps, min_sides_ratio, threshold, \
    alpha, delta, min_box_ratio, max_border_ratio, rsort_key, barcodes_path, sample_size, train_data_path, random_state, \
    negative_samples_dir


@benchmark
def main():
    test_dir = os.path.normpath("C:/Users/zgstv/OneDrive/Изображения/vend_machines")
    # test_dir = os.path.normpath("C:/Users/zgstv/OneDrive/Изображения/vend_machines2")
    result_dir = Path(r"C:\Users\zgstv\JupyterLab Notebooks\Mag-Project\Segmentation_sh\data\validation_images\v")
    clf = load(r'data\\model.joblib')
    # result_dir = os.path.abspath("data/train_data")
    # result_dir = os.path.abspath("data/barcodes2")
    # find_barcode(r"C:\Users\zgstv\JupyterLab Notebooks\Mag-Project\Segmentation_sh\data\test_images\2021-10"
    #              r"-20_13_32_07_783.png",valid_data_path)
    # test_from_dir(test_dir, result_dir)
    # walk_dir(test_dir,result_dir,find_barcode,clf)
    # scores = evaluate(data_path)
    scores = load(r'data/scores.joblib')
    json.dump( scores, open(r"data/scores.json", 'w' ) )
    print(scores)


def project_barcode():
    b = plt.imread("data/barcode_to_project.jpg")
    src = np.array([[0, 0], [300,0], [0,100], [300,100]])
    dst = np.array([[0,0], [350,80], [0,180], [350,255]])

    tform3 = transform.ProjectiveTransform()
    tform3.estimate(src, dst)
    warped = transform.warp(b, tform3, output_shape=(100, 300))


    fig, ax = plt.subplots(nrows=2, figsize=(8, 3))

    ax[0].imshow(b, cmap=plt.cm.gray)
    ax[0].plot(dst[:, 0], dst[:, 1], '.r')
    ax[1].imshow(warped, cmap=plt.cm.gray)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

    warped = np.pad(warped, ((10,), (10,), (0,)), mode='edge')
    warped = np.uint8(warped*255)
    gray = cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
    ret,bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    plt.imshow(bin,cmap='gray')
    plt.show()
    save(bin, pyzbar.decode(bin), None)


# @benchmark
def find_barcode(img_path: Union[str, Path], valid_res_path: Union[str, Path], classifier, show_valid_data=False,
                 save_valid_data=True,
                 save_img_data=True, load_img_data=True,
                 save_train_data=True, load_train_data=True):
    validation_data = []
    if type(img_path) is not Path:
        img_path = Path(img_path)
    valid_res_path = Path(valid_res_path) / img_path.stem

    img = plt.imread(img_path)
    validation_data.append((img, 'source', None))

    expected_size = 964, 1292
    img = prepare_image(img, expected_size)

    configs = get_boxes_configurations(min_box_area, max_box_area, n_box_sides_steps, min_sides_ratio)
    scores, boxes, regions_number, cur_v_data = propose_regions(img, threshold, configs, alpha, min_box_area,
                                                                max_box_area, delta, min_box_ratio, max_border_ratio,
                                                                rsort_key)

    if cur_v_data:
        validation_data.extend(cur_v_data)
    if boxes is None:
        return None

    regions = draw_regions(img, boxes)
    validation_data.append((regions, "regions", None))
    # print(f"Count of regions: {regions_number}")

    # testing from img
    X_test = get_test_data(img, sample_size, boxes)
    y_pred = classifier.predict(X_test)

    # number of barcode detections from all the patches in the image
    print(y_pred.sum())

    result_regions = draw_regions(img, boxes[y_pred == 1], draw_all=True, fill_rect=True)

    validation_data.append((result_regions, "result regions", None))

    pos_boxes = boxes[y_pred == 1]
    barcodes_extracted = [patch_from_box(img,box) for box in pos_boxes]
    for i,barcode in enumerate(barcodes_extracted):
        validation_data.append((barcode, f"barcode_{i}.png", None))

    if save_valid_data:
        save_validation_data(validation_data, valid_res_path)
    if show_valid_data:
        show_validation_data(validation_data)
    print()


def prepare():
    # if load_img_data and img_data_path.is_file():
    #     with np.load(img_data_path) as img_data:
    #         positive = img_data['positive']
    #         negative = img_data['negative']
    barcodes, barcodes_names = get_images_from_dir(barcodes_path)
    paths = get_files_from_dir(negative_samples_dir,["*.jpeg", "*.jpg", "*.png", "*.bmp"])
    negative_patches = [plt.imread(path) for path in paths if "p" not in path.parts]
    # negative_patches = extract_negative_samples(bottles_imgs_path, barcodes_names, configs)

    barcodes_augmented = []
    for barcode in barcodes:
        barcodes_augmented.extend(augment_barcode(barcode))
    # barcodes_augmented.extend([augment_barcode(barcode) for barcode in barcodes])
    positive = np.array([prepare_patch(barcode, sample_size) for barcode in barcodes_augmented])
    negative = np.array([prepare_patch(patch, sample_size) for patch in negative_patches])
    assert positive.shape[1:] == negative.shape[1:]
    # if save_img_data:
    #     np.savez_compressed(img_data_path, positive=positive, negative=negative)
    #
    # if load_train_data and train_data_path.is_file():
    #     with np.load(train_data_path) as train_data:
    #         X = train_data['X']
    #         y = train_data['y']
    X = np.array([apply_hog(im)
                  for im in tqdm(it.chain(positive,
                                          negative))])

    y = np.zeros(X.shape[0])
    y[:positive.shape[0]] = 1
    # if save_train_data:
    #     np.savez_compressed(train_data_path, X=X, y=y)
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    classifier = LinearSVC(C=0.1)
    classifier.fit(X_train, y_train)
    #
    #
    # # testing
    # y_pred = classifier.predict(X_test)
    # print(f1_score(y_test,y_pred))


def evaluate(save_folder):
    with np.load(train_data_path) as train_data:
        X = train_data['X']
        y = train_data['y']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    # classifier = LinearSVC(C=0.1)
    # classifier.fit(X_train, y_train)


    # testing
    # y_pred = classifier.predict(X_test)
    # print(f1_score(y_test,y_pred))

    # make_learning_curve(LinearSVC(C=0.1),X,y,save_folder,metric = f1_score)
    # make_learning_curve(LinearSVC(C=0.1),X,y,save_folder,metric = precision_score)
    # make_learning_curve(LinearSVC(C=0.1),X,y,save_folder,metric = recall_score)
    # make_learning_curve(LinearSVC(C=0.1),X,y,save_folder,metric = accuracy_score)
    # print(cross_val_score(LinearSVC(C=0.1),X,y,cv=5))
    scoring = {"f1":f1_score,"precision":precision_score,"recall":recall_score,"accuracy": accuracy_score}
    scoring = dict(zip(scoring.keys(),map(make_scorer,scoring.values())))
    scores = cross_validate(LinearSVC(C=0.1), X, y, scoring=scoring,cv=5,n_jobs=-1)
    for key,score in scores.items():
        scores[key] = {'mean':np.mean(score),'std':np.std(score)}
    return scores


if __name__ == '__main__':
    main()
