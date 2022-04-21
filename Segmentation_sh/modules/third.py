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
        print('Type : ', obj.type)
        print('Data : ', obj.data, '\n')

    return decodedObjects


# Display barcode and QR code location
def display(im, decoded_objects, save_path):
    # Loop over all decoded objects
    for decoded_object in decoded_objects:
        points = decoded_object.polygon

        # If the points do not form a quad, find convex hull
        # if len(points) > 4:
        hull = cv2.convexHull(np.array(points))
        cv2.polylines(im, [hull], True, (255, 0, 0), 3)
        # hull = hull.reshape(hull.shape[0],hull.shape[2])
        # hull = list(map(list, np.squeeze(hull)))
        # else:
        #     hull = points

        # Number of points in the convex hull
        # n = len(hull)

        # Draw the convext hull
        # for j in range(0, n):
        #     cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

        # cv2.rectangle(im, (decoded_object.rect.left, decoded_object.rect.top), (
        # decoded_object.rect.left + decoded_object.rect.width, decoded_object.rect.top + decoded_object.rect.height),
        #               (0, 255, 0), 3)
    if save_path:
        plt.imsave(save_path, im)
    else:
        # Display results
        # plt.figure(dpi=1000)
        plt.imshow(im)
        plt.show()


def save_barcode(img_path, save_path):
    if type(img_path) is not Path:
        img_path = Path(img_path)
    if not save_path.is_dir():
        save_path.mkdir(parents=True)
    save_path = Path(save_path) / img_path.name
    im = plt.imread(img_path)
    # decoded_objects = decode(im)
    display(im, pyzbar.decode(im), save_path)


#     cv2.waitKey(0);
def main():
    # Read image
    img_path = \
        "C:\\Users\\zgstv\\JupyterLab Notebooks\\Mag-Project\\Segmentation_sh\\data\\test\\IMG_20211029_163540.jpg"
    # img_path = 'C:\\Users\\zgstv\\JupyterLab Notebooks\\Mag-Project\\Segmentation_sh\\data\\train_data\\Glass '
    #                 'green with lid and label\\2021-10-20_14_15_17_418\\32.jpg'
    im = plt.imread(img_path)
    # im= im[2535+45:3057-40,1087+25:1357-100]
    # plt.figure(dpi=1000)
    # plt.imshow(im)
    # plt.show()
    decodedObjects = decode(im)
    display(im, decodedObjects, None)


if __name__ == '__main__':
    main()
