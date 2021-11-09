import cv2
import numpy as np

from util import normalize_image, convolution


def gauss_func(sigma: float, x_pos: int, y_pos: int):
    sigma_2 = sigma ** 2
    gaussParam = 1 / (2 * np.pi * sigma_2)
    expArgue = -(x_pos ** 2 + y_pos ** 2) / (2 * sigma_2)
    return gaussParam * np.exp(expArgue)


def make_gauss_kernel(size: (int, int), sigma: float):
    w, h = size
    if w % 2 != 1 and h % 2 != 1:
        raise Exception('should be w,h % 2  == 1')
    kernel_center_w, kernel_center_h = w // 2, h // 2
    kernel = np.array(
        [[gauss_func(sigma, x_pos - kernel_center_w, y_pos - kernel_center_h) for x_pos in range(w)] for y_pos in
         range(h)])
    kernel /= np.max(kernel)
    return kernel


def gaussian_blur(image, kernel_size: (int, int), sigma: float, is_bgr: bool = True) -> np.ndarray:
    kernel = make_gauss_kernel(kernel_size, sigma)
    output = convolution(image, kernel, is_bgr)
    output = normalize_image(output, 0, 255)
    return output.astype(np.uint8)
