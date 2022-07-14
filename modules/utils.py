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


def images_in_dir_generator(source_dir: [str, Path], result_dir: [str, Path]):
    # if fkwargs is None:
    #     fkwargs={}
    if type(source_dir) == str:
        source_dir = Path(source_dir)
    if type(result_dir) == str:
        result_dir = Path(result_dir)
    if not source_dir.is_dir():
        raise Exception("Тестовая директория не существует")
    # if not callable(func):
    #     raise ValueError(f"Аргумент {func} не является вызываемым")
    # if not os.listdir(source_dir):
    #     raise Exception("Тестовая директория пустая")
    if not result_dir.is_dir():
        result_dir.mkdir(parents=True)
    for root, dirs, files in os.walk(str(source_dir)):
        for file in files:
            file_path = Path(os.path.join(root, file))
            ext = file_path.suffix
            possible_exts = ".jpg", ".jpeg", ".png", ".bmp"
            if ext in possible_exts:
                cur_res_dir = Path(root).relative_to(source_dir)
                save_dir = result_dir / cur_res_dir
                yield file_path, save_dir
                # func(file_path, save_dir, *fargs, **fkwargs)
