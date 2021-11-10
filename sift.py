from typing import Optional, Union

import numpy as np

import gauss_filter
from keypoint import Keypoint
from util import nearest_neighbor_sampling
import operator


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
    dog = [[gaussian_pyramid[octave][pos + 1] - gaussian_pyramid[octave][pos] for pos in
            range(octaves_size - 1)] for
           octave in range(octaves_number)]
    return dog


def get_keypoints(dog: list[list[np.ndarray]]) -> list[Keypoint]:
    ###what to do with  type ? to[0,1] or just int ?
    keypoints = []
    for octave_index, octave in enumerate(dog):
        print('octave: ', octave_index)
        h, w = octave[0].shape
        for image_index in range(1, len(octave) - 1):
            prev_image = octave[image_index - 1]
            cur_image = octave[image_index]
            next_image = octave[image_index + 1]
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    prev_cube = prev_image[i - 1:i + 2, j - 1:j + 2]
                    cur_cube = cur_image[i - 1:i + 2, j - 1:j + 2]
                    next_cube = next_image[i - 1:i + 2, j - 1:j + 2]

                    if is_scale_extremum(prev_cube, cur_cube,
                                         next_cube, 5):
                        localize_result = localize_keypoint(i, j, image_index, octave_index,
                                                            len(octave) - 3, octave, 0.5)
                        if localize_result is not None:
                            keypoint, localized_image_index = localize_result
                            keypoints.append(keypoint)
    return keypoints


def localize_keypoint(i: int, j: int, image_index: int, octave_index: int, num_intervals: int,
                      octave_data: list[np.ndarray],
                      contrast_threshold, sigma: float = 1.6,
                      image_border_width: int = 1, eigenvalue_ratio: int = 10,
                      num_attempts_until_convergence: int = 5) -> Optional[tuple[Keypoint, int]]:
    extremum_is_outside_image = False
    image_shape = octave_data[0].shape
    for attempt_index in range(num_attempts_until_convergence):

        prev_image, cur_image, next_image = octave_data[image_index - 1:image_index + 2]
        pixel_cube = np.stack([prev_image[i - 1:i + 2, j - 1:j + 2],
                               cur_image[i - 1:i + 2, j - 1:j + 2],
                               next_image[i - 1:i + 2, j - 1:j + 2]]).astype('float32') / 255.
        gradient = calc_center_gradient(pixel_cube)
        hessian = calc_center_hessian(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0] # sole, taylor shit, answer is (1,3) vector
        # https://habr.com/ru/post/106302/
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= \
                image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    response = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    if abs(response) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < (
                (eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            pos_x = int((j + extremum_update[0]) * (2 ** octave_index))
            pos_y = int((i + extremum_update[1]) * (2 ** octave_index))
            radius = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (
                    2 ** (octave_index + 1))
            response = abs(response)

            return Keypoint(pos_x, pos_y, radius, response), image_index
    return None


def is_scale_extremum(prev_pixels: np.ndarray, cur_pixels: np.ndarray, next_pixels: np.ndarray, thr) -> bool:
    center_pixel_value = cur_pixels[1, 1]
    if center_pixel_value > thr:
        max_prev = np.max(prev_pixels)
        max_cur = np.max(cur_pixels)
        max_next = np.max(next_pixels)
        return center_pixel_value >= max_prev and center_pixel_value >= max_cur and center_pixel_value > max_next
    return False


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
    return np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])


def calc_center_gradient(pixels: np.ndarray) -> np.ndarray:
    # f'(x) = (f(x + 1) - f(x - 1)) / 2
    # blur - change distance or zooming image so blur is about scale
    # calc gradient in [1,1,1], x - third dim, y - second dim,  scale - first dim
    dx = 0.5 * (pixels[1, 1, 2] - pixels[1, 1, 0])
    dy = 0.5 * (pixels[1, 2, 1] - pixels[1, 0, 1])
    ds = 0.5 * (pixels[2, 1, 1] - pixels[0, 1, 1])
    return np.array([dx, dy, ds])
