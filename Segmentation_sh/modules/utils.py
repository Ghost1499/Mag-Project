import os
import time
from pathlib import Path


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print('[*] Время выполнения: {} секунд.'.format(end - start))
        return res
    return wrapper


def is_horizontal(size):
    return size[0] < size[1]


def walk_dir(test_dir: [str, Path], result_dir: [str, Path], func, fargs=(), fkwargs=None):
    if fkwargs is None:
        fkwargs={}
    if type(test_dir) == str:
        test_dir = Path(test_dir)
    if type(result_dir) == str:
        result_dir = Path(result_dir)
    if not test_dir.is_dir():
        raise Exception("Тестовая директория не существует")
    if not callable(func):
        raise ValueError(f"Аргумент {func} не является вызываемым")
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
                func(path, save_dir, *fargs, **fkwargs)

                # find_barcode(path, valid_res_path=save_dir)