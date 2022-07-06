from __future__ import print_function

from pathlib import Path

import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import matplotlib.pyplot as plt


def decode(im):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)

    # Print results
    for obj in decodedObjects:
        print_barcode('Type : ', obj.type)
        print_barcode('Data : ', obj.data, '\n')

    return decodedObjects


def print_barcode(decoded_object):
    barcode_info = f"Type: {decoded_object.type}\nData: {decoded_object.data} (bytes)\n" \
                   f"Rect: x={decoded_object.rect.left} y={decoded_object.rect.top} " \
                   f"w={decoded_object.rect.width} h={decoded_object.rect.height}\n"
    return barcode_info


def write_barcodes_info(barcode_info_file,barcodes_info):
    for i, barcode_info in enumerate(barcodes_info):
        barcode_info_file.write(str(i + 1) + ". " + barcode_info + "\n")


# Display barcode and QR code location
def save(img, decoded_objects, save_path):
    # Loop over all decoded objects
    barcodes_info = []
    barcodes_roi = []
    barcodes_roim = []
    img_marked = np.ndarray.copy(img)
    for decoded_object in decoded_objects:
        # draw barcode
        points = decoded_object.polygon
        hull = cv2.convexHull(np.array(points))
        cv2.polylines(img_marked, [hull], True, (255, 0, 0), 3)
        # get rect
        rect = decoded_object.rect
        # get region
        offset = 100
        # get barcode roi
        barcode = img[rect.top - offset:rect.top + rect.height + offset, rect.left - offset:rect.left + rect.width + offset]
        barcode_marked = img_marked[rect.top - offset:rect.top + rect.height + offset, rect.left - offset:rect.left + rect.width + offset]
        barcodes_roi.append(barcode)
        barcodes_roim.append(barcode_marked)
        barcode_info = print_barcode(decoded_object)
        barcodes_info.append(barcode_info)

    if save_path:
        img_res_path = save_path / "res.jpg"
        plt.imsave(img_res_path, img_marked)
        if decoded_objects:
            for i, barcode_roi in enumerate(barcodes_roi):
                barcode_roi_path = save_path / f"barcode{i + 1}.jpg"
                plt.imsave(barcode_roi_path, barcode_roi)
            for i, barcode_roim in enumerate(barcodes_roim):
                barcode_roim_path = save_path / f"barcode marked{i + 1}.jpg"
                plt.imsave(barcode_roim_path, barcode_roim)

            barcode_info_path = save_path / f"barcodes_info.txt"
            with open(barcode_info_path,'w') as barcode_info_file:
                write_barcodes_info(barcode_info_file,barcodes_info)

    else:
        plt.imshow(img)
        plt.show()
        for barcode in barcodes_roi:
            plt.imshow(barcode)
            plt.show()
        for i,info in enumerate(barcodes_info):
            print(str(i + 1) + ". " + info + "\n")


def read_barcode(img_path, save_folder):
    if type(img_path) is not Path:
        img_path = Path(img_path)
    if type(save_folder) is not Path:
        save_folder = Path(save_folder)
    save_folder = Path(save_folder) / img_path.stem
    if not save_folder.is_dir():
        save_folder.mkdir(parents=True)
    im = plt.imread(img_path)
    if im is None:
        raise Exception("Тестовое изображение не считано")

    decoded_objects = pyzbar.decode(im)
    save(im, decoded_objects, save_folder)


#     cv2.waitKey(0);
def main():
    # Read image
    img_path = \
        "C:\\Users\\zgstv\\JupyterLab Notebooks\\Mag-Project\\Segmentation_sh\\data\\test_images\\IMG_20211029_163540.jpg"
    # img_path = 'C:\\Users\\zgstv\\JupyterLab Notebooks\\Mag-Project\\Segmentation_sh\\data\\train_data\\Glass '
    #                 'green with lid and label\\2021-10-20_14_15_17_418\\32.jpg'
    im = plt.imread(img_path)
    # img= img[2535+45:3057-40,1087+25:1357-100]
    decoded_objects = pyzbar.decode(im)
    save(im, decoded_objects, None)


if __name__ == '__main__':
    main()
