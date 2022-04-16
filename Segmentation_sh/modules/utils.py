import time

from skimage.transform import resize
from sklearn.feature_extraction import image


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print('[*] Время выполнения: {} секунд.'.format(end - start))
        return res
    return wrapper


def extract_patches(img, patch_sizes, count=None):
    patches = []
    for patch_size in patch_sizes:
        patches.extend(image.extract_patches_2d(img, patch_size=patch_size, max_patches=count, random_state=0))
    return patches


def is_horizontal(size):
    return size[0] < size[1]


def match_orientation(img, size):
    size_h = is_horizontal(size)
    img_h = is_horizontal(img.shape[0:2])
    if size_h != img_h:
        img = img.transpose((1, 0, 2))
    return img


def resize_img(img, size):
    return resize(img, size, anti_aliasing=True)


def patch_from_box(img, box, with_border=True):
    y, x, height, width,border = box
    offset = border if with_border else 0
    return img[y-offset:y+height+offset,x-offset:x+width+offset]



