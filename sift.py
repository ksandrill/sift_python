import numpy as np

import gauss_filter
from keypoint import Keypoint
from util import nearest_neighbor_sampling


def get_dog(picture: np.ndarray, octaves_number: int = None, octaves_size: int = 3, min_sigma: float = 0.8,
            delta: float = 0.5) -> list[list[np.ndarray]]:
    h, w = picture.shape
    if octaves_number is None:
        octaves_number = int(np.log2(np.min(h, w))) - 3
    octaves_size += 3
    ###actually idk how correct calc sigmas and octave_size
    sigmas = [[delta * min_sigma * 2 ** (octave_ind + pos_in_octave / octaves_number) for pos_in_octave in
               range(octaves_size)] for octave_ind in range(octaves_number)]
    images = [nearest_neighbor_sampling(picture, 1 << octave_ind) for octave_ind in range(octaves_number)]
    gaussian_pyramid = [
        [gauss_filter.gaussian_blur(images[octave], (3, 3), sigmas[octave][octave_pos]) for octave_pos in
         range(octaves_size)] for octave in range(octaves_number)]
    dog = [[gaussian_pyramid[octave][pos + 1].astype(np.int) - gaussian_pyramid[octave][pos].astype(np.int) for pos in
            range(octaves_size - 1)] for
           octave in range(octaves_number)]
    return dog


def localaze_keypoints(dog: list[list[np.ndarray]]) -> list[Keypoint]:
    pass


def calc_center_hessian(pixels: np.ndarray) -> np.ndarray:
    # f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # df/dxdy (x,y)  =(f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # x - third dim, y - second dim,  scale - first dim
    # hessian = [dxx, dxy, dxs]
    #           [dxy, dyy, dys]
    #           [dxs, dys, dss]
    dxx = pixels[1, 1, 2] - 2 * pixels[1, 1, 1] + pixels[1, 1, 0]
    dyy = pixels[1, 2, 1] - 2 * pixels[1, 1, 1] + pixels[1, 0, 1]
    dss = pixels[2, 1, 1] - 2 * pixels[1, 1, 1] + pixels[0, 1, 1]
    dxy = (pixels[1, 2, 2] - pixels[1, 2, 0] - pixels[1, 0, 2] + pixels[1, 0, 0]) * 0.25
    dxs = (pixels[2, 1, 2] - pixels[0, 1, 2] - pixels[2, 1, 0] + pixels[0, 1, 0]) * 0.25
    dys = (pixels[2, 2, 1] - pixels[2, 0, 1] - pixels[0, 2, 1] + pixels[0, 0, 1]) * 0.25
    return np.ndarray([[dxx, dxy, dxs],
                       [dxy, dyy, dys],
                       [dxs, dys, dss]])


def calc_center_gradient(pixels: np.ndarray) -> np.ndarray:
    # f'(x) = (f(x + 1) - f(x - 1)) / 2
    # blur - change distance or zooming image so blur is about scale
    # calc gradient in [1,1,1], x - third dim, y - second dim,  scale - first dim
    dx = 0.5 * (pixels[1, 1, 2] - pixels[1, 1, 0])
    dy = 0.5 * (pixels[1, 2, 1] - pixels[1, 0, 1])
    ds = 0.5 * (pixels[2, 1, 1] - pixels[0, 1, 1])
    return np.ndarray([dx, dy, ds])
