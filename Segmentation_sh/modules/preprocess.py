import itertools as it

import skimage
from skimage import feature
from skimage.transform import resize
import numpy as np

from Segmentation_sh.modules.utils import is_horizontal


def prepare_image(img, expected_size):
    img = match_orientation(img, expected_size)
    if img.shape[:2] != expected_size:
        img = resize(img, expected_size, anti_aliasing=True)
    mean = np.mean(img)
    if 0 < mean < 1:
        img = img * 255
    return img.astype("uint8")


def match_orientation(img, size):
    size_h = is_horizontal(size)
    img_h = is_horizontal(img.shape[0:2])
    if size_h != img_h:
        img = img.transpose((1, 0, 2))
    return img


def resize_img(img, size):
    return resize(img, size, anti_aliasing=True)


def prepare_patch(patch, sample_size):
    return resize_img(match_orientation(patch, sample_size), sample_size)


def patch_from_box(img, box, with_border=True):
    y, x, height, width,border = box
    offset = border if with_border else 0
    return img[y-offset:y+height+offset,x-offset:x+width+offset]


def apply_hog(patch):
    return feature.hog(patch, channel_axis=2)


def augment_barcode(barcode):
    vflip = barcode[::-1]
    hflip = barcode[...,::-1]
    hvflip = barcode[::-1,::-1]
    flipped = (vflip,hflip,hvflip)
    angles_range = list(it.chain(range(-5,0),range(1,6)))
    augmented = []
    for img in flipped:
        for angle in angles_range:
            augmented.append(np.array(skimage.transform.rotate(img,angle,resize =True,mode="edge")))
    return augmented
