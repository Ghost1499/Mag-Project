import itertools as it
from pathlib import Path

from matplotlib import pyplot as plt


def mrglob(path: Path, *patterns):
    return it.chain.from_iterable(path.rglob(pattern) for pattern in patterns)


def get_files_from_dir(dir: Path, patterns = None, exclude=None):
    if patterns is None:
        patterns= ["*"]
    if not dir.is_dir():
        raise Exception("Директория не существует")
    paths = list(mrglob(dir, *patterns))
    if exclude:
        paths = [path for path in paths if path.stem not in exclude]
    return paths


def get_images_from_dir(dir: Path, exclude=None):
    patterns = ["*.jpeg", "*.jpg", "*.png", "*.bmp"]
    paths = get_files_from_dir(dir, patterns, exclude)
    names = [path.stem for path in paths]
    images = [plt.imread(dir / path) for path in paths]
    return images, names