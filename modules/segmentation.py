import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.transform import rescale
from skimage.transform import rotate


def _rotate_to_normal(img, angle):
    return rotate(img, angle, resize=True, mode="edge")


def _crop_to_normal(img, crop_bounds):
    return img[crop_bounds]


def _downscale(img, scale):
    return rescale(img, scale, anti_aliasing=False)


def _get_init_snake(h, w):
    # h,w = img.shape
    r = np.arange(0, h)
    c = np.arange(0, w)
    y = [0] * w + list(range(1, h - 1)) + [h - 1] * w + list(range(h - 1, 1, -1))
    x = list(range(0, w)) + [w - 1] * (h - 2) + list(range(w - 1, -1, -1)) + [0] * (h - 2)
    init = np.array([y, x]).T
    return init


def _segment(img, init, alpha, beta, w_edge):
    snake = active_contour(gaussian(img, 3, preserve_range=False), init, alpha=alpha, beta=beta, gamma=0.001,
                           w_edge=w_edge)
    return snake


def _draw(image_rescaled, init, snake, dpi=1000):
    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(image_rescaled, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=1)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=1)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, image_rescaled.shape[1], image_rescaled.shape[0], 0])
    return fig, ax


def _mask_from_snake(snake, img_shape):
    img_shape = img_shape[0:2]
    contours_img = np.zeros(img_shape, "uint8")
    snake_int = np.int32(np.round(snake))
    snake_int = np.unique(snake_int, axis=0)
    contours_img[snake_int[:, 0], snake_int[:, 1]] = 255
    contours_img = cv.dilate(contours_img, kernel=None, iterations=2)
    contours, hierarchy = cv.findContours(contours_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img_shape, 'uint8')
    cnt = contours[0]
    mask = cv.drawContours(mask, [cnt], -1, 255, -1)
    mask = cv.erode(mask, kernel=None, iterations=2)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy, mask
