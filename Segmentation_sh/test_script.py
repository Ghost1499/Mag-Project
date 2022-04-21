import itertools as it
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
from matplotlib import pyplot as plt
from skimage import transform
from skimage.transform import resize
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm

from Segmentation_sh.modules.model import get_best_model, plot_training_curve
from Segmentation_sh.modules.preprocess import match_orientation, prepare_patch, apply_hog, \
    augment_barcode, prepare_image
from Segmentation_sh.modules.extract import extract_negative_samples, get_patches_from_boxes, get_test_data
from Segmentation_sh.modules.read import get_images_from_dir
from Segmentation_sh.modules.regions import get_boxes_configurations, propose_regions
from Segmentation_sh.modules.third import save_barcode, display
from Segmentation_sh.modules.utils import benchmark, walk_dir
from Segmentation_sh.modules.validation import draw_regions, save_validation_data, show_validation_data
from Segmentation_sh.params_config import min_box_area, max_box_area, n_box_sides_steps, min_sides_ratio, threshold, \
    alpha, delta, min_box_ratio, max_border_ratio, rsort_key, img_data_path, barcodes_path, bottles_imgs_path, \
    sample_size, train_data_path, valid_data_path, random_state


def main():
    test_dir = os.path.normpath("C:/Users/zgstv/OneDrive/Изображения/vend_machines")
    # test_dir = os.path.normpath("C:/Users/zgstv/OneDrive/Изображения/vend_machines2")
    # test_dir = os.path.abspath("test_data")
    # result_dir = os.path.abspath("results/binary_img/v")
    # result_dir = os.path.abspath("results/binary_img/v2")
    # result_dir = os.path.abspath("results/test")
    # result_dir = os.path.abspath("data/train_data")
    result_dir = os.path.abspath("data/barcodes2")
    # test_from_dir(test_dir, result_dir)
    # walk_dir(test_dir,result_dir,save_barcode)
    # find_barcode("data/test/2021-10-20_13_32_07_783.png",valid_data_path)#,load_img_data=False,load_train_data=False)
    # project_barcode()
    find_lines()
    # with np.load(str(train_data_path) + ".npz") as train_data:
    #     X_train = train_data['X_train']
    #     y_train = train_data['y_train']
    # model = get_best_model(X_train, y_train)
    # model = LinearSVC(C=0.1)
    # plot_training_curve(model, X_train, y_train)
    print()


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
    display(bin, pyzbar.decode(bin),None)


def find_lines():
    img = plt.imread("data/barcode_to_project.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    # for line in lines:
    #     rho, theta = line[0]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    plt.imsave('data/houghlines3.jpg', img)

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
                # if str(cur_res_dir) == "Background" or str(cur_res_dir) == "Aluminum can":
                #     continue
                save_dir = result_dir / cur_res_dir

                find_barcode(path, valid_res_path=save_dir)
                # res,tpath = find_barcode(path, valid_res_path=save_dir)
                # if res is None:
                #     continue
                # for i in range(res.shape[0]):
                #     if not tpath.is_dir():
                #         tpath.mkdir(parents=True)
                #     plt.imsave(tpath/(str(i)+".jpg"),res[i])


@benchmark
def find_barcode(img_path: Union[str, Path], valid_res_path: Union[str, Path], show_valid_data=False,
                 save_valid_data=True,
                 save_img_data=True, load_img_data=True,
                 save_train_data=True,load_train_data=True):
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
        return None#,valid_data_path
    # return get_patches_from_boxes(img, sample_size, boxes), valid_res_path

    regions = draw_regions(img, boxes)
    validation_data.append((regions, "regions", None))
    print(f"Count of regions: {regions_number}")

    if load_img_data and img_data_path.is_file():
        with np.load(str(img_data_path) + ".npz") as img_data:
            positive = img_data['positive']
            negative = img_data['negative']
    else:
        barcodes, barcodes_names = get_images_from_dir(barcodes_path)
        negative_patches = extract_negative_samples(bottles_imgs_path, barcodes_names, configs)
        # провести аугментацию штрихкодов (поворот по вертикали, горизонтали, и так, и так; поворот на несколько
        # градусов по часовой и против часовой)
        barcodes_augmented = []
        for barcode in barcodes:
            barcodes_augmented.extend(augment_barcode(barcode))
        # barcodes_augmented.extend([augment_barcode(barcode) for barcode in barcodes])
        positive = np.array([prepare_patch(barcode, sample_size) for barcode in barcodes_augmented])
        negative = np.array([prepare_patch(patch, sample_size) for patch in negative_patches])
        assert positive.shape[1:] == negative.shape[1:]
        if save_img_data:
            np.savez_compressed(img_data_path, positive=positive, negative=negative)

    if load_train_data and train_data_path.is_file():
        with np.load(str(train_data_path) + ".npz") as train_data:
            X = train_data['X']
            y = train_data['y']
    else:
        X = np.array([apply_hog(im)
                      for im in tqdm(it.chain(positive,
                                              negative))])

        y = np.zeros(X.shape[0])
        y[:positive.shape[0]] = 1
        if save_train_data:
            np.savez_compressed(train_data_path, X=X, y=y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    model = LinearSVC(C=0.1)
    model.fit(X_train, y_train)


    # testing
    y_pred = model.predict(X_test)
    print(f1_score(y_test,y_pred))

    # testing from img
    X_test = get_test_data(img, sample_size, boxes)
    y_pred = model.predict(X_test)
    # print(f1_score(y_test, y_pred))

    # number of barcode detections from all the patches in the image
    print(y_pred.sum())

    result_regions = draw_regions(img, boxes[y_pred == 1], draw_all=True, fill_rect=True)

    validation_data.append((result_regions, "result regions", None))

    if save_valid_data:
        save_validation_data(validation_data, valid_res_path)
    if show_valid_data:
        show_validation_data(validation_data)
    print()


if __name__ == '__main__':
    main()
