from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from tqdm import tqdm

# scale =0.25
from modules.segmentation import _rotate_to_normal, _crop_to_normal, _downscale, _get_init_snake, _segment, _draw, \
    _mask_from_snake
from modules.utils import images_in_dir_generator

scale = 0.2
# alpha=0.15
# alpha=1
# alphas = list(np.linspace(0.1,1,4,True))
alphas = [0.5]
beta = 1
w_edge = 1
rotate_angle = -4
axis = 1
slices_count = 5
score_thresh = 0.07


def perform_classification(img, save_dir):
    if img is None:
        raise ValueError("Входное изображение - None")
    gray = img
    if img.ndim == 3:
        gray = rgb2gray(img)
    if gray.ndim != 2:
        raise ValueError(f"Входное изображение имеет не подходящее число каналов - {gray.ndim}")
    rotated = _rotate_to_normal(gray, rotate_angle)
    # plt.imsave(save_dir/f'Rotate by {rotate_angle} degrees.png',rotated,cmap='gray')
    cropped = _crop_to_normal(rotated)
    plt.imsave(save_dir / f'Cropped.png', cropped, cmap='gray')
    # print(cropped.shape)
    downscaled = _downscale(cropped, scale)
    init = _get_init_snake(*downscaled.shape)
    if not alphas:
        raise ValueError("Значения alpha для сегментации не заданы")
    for alpha in alphas:
        snake = _segment(downscaled, init, alpha, beta, w_edge)
        fig, ax = _draw(downscaled, init, snake)
        fig.savefig(save_dir / f"Snake_rescale{scale}_alpha{alpha}_beta{beta}_w_edge{w_edge}.png")
        plt.close(fig)
    contours, hierarchy, mask = _mask_from_snake(snake, downscaled.shape)
    cnt = contours[0]
    # plt.imsave(save_dir/"Contours.png",cv.drawContours(np.uint8(np.copy(downscaled) * 255), [cnt], -1, 255, 1),cmap='gray')
    plt.imsave(save_dir / "Mask.png", mask, cmap='gray')

    score, tare_type = _classify(mask, save_dir)
    type_map = {0: "Бутылка", 1: "Банка"}
    with open(save_dir / 'result.txt', 'w') as res_file:
        res_file.write(f"Разность: {score}\nПорог: {score_thresh}\nТип: {type_map[tare_type]}")
    return tare_type


def _classify(mask, save_dir):
    slices, _ = make_slices(mask, axis=axis, slices_count=slices_count, save=True,
                            fname=str(save_dir / f"Slices axis{axis} slices_count{slices_count}"))
    ratios = slices_ratios(slices)
    diff = abs(ratios[-1] - ratios[0])
    # print(f"Заполненная площадь слайсов: {ratios}\nРазница между первым и последним: {diff}\n")
    tare_type = 0
    if diff < score_thresh:
        tare_type = 1
    return diff, tare_type


def crop_object(mask):
    nonzero = (mask == 255).nonzero()
    ymin = nonzero[0].min()
    ymax = nonzero[0].max()
    xmin = nonzero[1].min()
    xmax = nonzero[1].max()
    # mask.shape
    # ymin,ymax
    return mask[ymin:ymax, xmin:xmax], (ymin, ymax, xmin, xmax)


def make_indents(mask, y_indent, x_indent):
    mask = _make_indent(mask, y_indent, 0)
    mask = _make_indent(mask, x_indent, 1)
    return mask


def _make_indent(mask, indent, axis):
    if axis not in (0, 1):
        raise ValueError("Axis must be 0(y) or 1(x)")

    if indent > 0:
        if axis == 0:
            mask = mask[round(mask.shape[axis] * indent):round((mask.shape[axis] - 1) * (1 - indent))]
        elif axis == 1:
            mask = mask[::, round(mask.shape[axis] * indent):round((mask.shape[axis] - 1) * (1 - indent))]
    return mask


def sclice_vertical(mask, slices_count):
    return _slice_img(mask, slices_count, 0)


def sclice_horizontal(mask, slices_count):
    return _slice_img(mask, slices_count, 1)


def _slice_img(mask, slices_count, axis):
    bounds_indexes = np.linspace(0, mask.shape[axis], slices_count + 1)
    bounds_indexes = np.round(bounds_indexes).astype("int32")
    slices = np.split(mask, bounds_indexes[1:-1], axis)
    return slices, bounds_indexes


def draw_slices(mask, bounds_indexes, axis, save, title):
    # axis 0: hline
    # axis 1: vline
    plt.figure(dpi=500)
    plt.imshow(mask, cmap='gray')
    if axis == 0:
        hline(bounds_indexes, mask)
    elif axis == 1:
        vline(bounds_indexes, mask)

    if save:
        plt.savefig(title + ".jpg")
    plt.close()
    # plt.show()


def hline(bounds_indexes, mask):
    plt.hlines(bounds_indexes[1:-1], 0, mask.shape[1],
               color='g',
               linewidth=1,
               linestyle='--')


def vline(bounds_indexes, mask):
    plt.vlines(bounds_indexes[1:-1], 0, mask.shape[0],
               color='g',
               linewidth=1,
               linestyle='--')


def make_slices(mask, axis=0, slices_count=9, y_indent=0.05, x_indent=0.02, save=False, fname=None):
    # axis 0: horizontal
    # axis 1: vertical
    mask = np.copy(mask)
    mask, (ymin, ymax, xmin, xmax) = crop_object(mask)
    mask = make_indents(mask, y_indent, x_indent)

    slices, bounds_indexes = _slice_img(mask, slices_count, axis)
    draw_slices(mask, bounds_indexes, axis, save, fname)
    return slices, bounds_indexes


def slices_ratios(slices):
    slices_ratios = []
    for sl in slices:
        slices_ratios.append(np.mean(sl) / 255)
    slices_ratios = np.array(slices_ratios)
    return slices_ratios


def test_from_folder():
    test_path = Path(r"C:\Users\zgstv\OneDrive\Изображения\vend_machines_ab3_in_box")
    save_path = Path(r'data/segmentation')
    folder = None
    errors_count = {}
    right_type = None
    errors_count_filename = save_path / "errors_count.txt"
    for img_path, res_path in tqdm(images_in_dir_generator(test_path, save_path)):
        save_dir = res_path / img_path.stem
        if save_dir.is_dir():
            continue
        else:
            save_dir.mkdir(parents=True)
        img = plt.imread(img_path)
        tare_type = perform_classification(img, save_dir)
        if res_path.name != folder:
            if folder is not None:
                with open(errors_count_filename, 'a') as errors_count_file:
                    print(f'{folder}: {errors_count[folder]}', file=errors_count_file)
            folder = res_path.name
            right_type = 0
            if folder == 'Aluminium':
                right_type = 1
            errors_count[folder] = 0
        if tare_type != right_type:
            errors_count[folder] += 1


def test_single():
    test_bottle_path = Path(
        r'C:\Users\zgstv\OneDrive\Изображения\vend_machines_ab3_in_box\Aluminium\20220713_080432.jpg'
    )
    save_path = Path(rf'data/segmentation/{test_bottle_path.stem}')
    if not save_path.is_dir():
        save_path.mkdir(parents=True)
    img = plt.imread(test_bottle_path)
    perform_classification(img, save_path)


if __name__ == '__main__':
    # test_from_folder()
    test_single()

