from abc import abstractmethod, ABC

import cv2
import numpy as np

from modules.validation import Validated


class IEdgeMapCreator(ABC):

    @abstractmethod
    def make_x(self, img):
        pass

    @abstractmethod
    def make_y(self, img):
        pass


class AStandardEdgeMapCreator(IEdgeMapCreator, ABC):
    _bin_thresh: int
    _closing_kernel: np.ndarray
    _opening_kernel: np.ndarray
    _x_dir = None

    @abstractmethod
    def _create_closing_kernel(self):
        pass

    @abstractmethod
    def _create_opening_kernel(self):
        pass

    def make_x(self, img):
        return self._make(img, True)

    def make_y(self, img):
        return self._make(img, False)

    def _make(self, img, x_dir: bool):
        self._x_dir = x_dir
        self._closing_kernel = self._create_closing_kernel()
        self._opening_kernel = self._create_opening_kernel()
        grad = self._gradient(img)
        closed = self._close(grad)
        opened = self._open(closed)
        edge_map = self._binarize(opened)
        return edge_map

    @abstractmethod
    def _gradient(self, img):
        pass

    @abstractmethod
    def _close(self, grad):
        pass

    @abstractmethod
    def _open(self, closed):
        pass

    @abstractmethod
    def _binarize(self, opened):
        pass


class StandardEdgeMapCreator(AStandardEdgeMapCreator):

    def __init__(self, bin_thresh=120):
        self._bin_thresh = bin_thresh

    def _create_closing_kernel(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        if not self._x_dir:
            kernel = np.transpose(kernel, axes=(1, 0))
        return kernel

    def _create_opening_kernel(self):
        return cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

    # def to_gray(self,img):
    #     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _gradient(self, gray):
        dx = 0
        dy = 0
        if self._x_dir:
            dx = 1
        else:
            dy = 1
        grad = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=dx, dy=dy, ksize=3)
        grad = np.absolute(grad)
        return grad

    def _close(self, grad):
        return cv2.morphologyEx(grad, cv2.MORPH_CLOSE, self._closing_kernel)

    def _open(self, grad):
        return cv2.morphologyEx(grad, cv2.MORPH_OPEN, self._opening_kernel)

    def _binarize(self, img):
        _, binary = cv2.threshold(img, self._bin_thresh, 255, cv2.THRESH_BINARY)
        return binary


class ValidationStandardEdgeMapCreator(StandardEdgeMapCreator, Validated):
    _cmap = "img"
    _gradient_fname = "Производная"
    _close_fname = "Замыкание прямоугольником"
    _open_fname = "Размыкание квадратом"
    _binarize_fname = "Карта перепадов"

    def _construct_fname(self, fname):
        if self._x_dir is None:
            raise Exception("Направление вычисления edge map не задано")
        postfix = "х"
        if not self._x_dir:
            postfix = "у"
        return " ".join([fname, postfix])

    def _create_valid_data(self, image, name):
        return image, self._construct_fname(name), self._cmap

    def _gradient(self, gray):
        gray = super(ValidationStandardEdgeMapCreator, self)._gradient(gray)
        self._add_vdata(self._create_valid_data(gray, self._gradient_fname))

    def _close(self, grad):
        closed = super(ValidationStandardEdgeMapCreator, self)._close(grad)
        self._add_vdata(self._create_valid_data(closed, self._close_fname))

    def _open(self, grad):
        opened = super(ValidationStandardEdgeMapCreator, self)._open(grad)
        self._add_vdata(self._create_valid_data(opened, self._open_fname))

    def _binarize(self, img):
        edge_map = super(ValidationStandardEdgeMapCreator, self)._binarize(img)
        self._add_vdata(self._create_valid_data(edge_map, self._binarize_fname + f" {self._bin_thresh}t"))
