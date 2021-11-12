from typing import Optional

import numpy as np

import gauss_filter
from keypoint import Keypoint
from util import nearest_neighbor_sampling, calc_center_hessian, calc_center_gradient


def get_dog(picture: np.ndarray, octaves_number: int = None, octaves_size: int = 3, min_sigma: float = 0.8,
            delta: float = 0.5) -> (list[list[np.ndarray]], list[list[np.ndarray]]):
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
    return dog, gaussian_pyramid


def get_keypoints(dog: list[list[np.ndarray]], gaussian_pyramid: list[list[np.ndarray]], peak_thr: float = 0.8) -> list[
    Keypoint]:
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
                            orientation_max, hist = get_keypoint_main_orientation(keypoint,
                                                                                  gaussian_pyramid[octave_index][
                                                                                      localized_image_index].astype(
                                                                                      np.float32) / 255,
                                                                                  octave_index)

                            keypoints += get_oriented_keypoints(keypoint, hist, orientation_max, peak_thr)
    return keypoints


def get_oriented_keypoints(keypoint: Keypoint, hist: np.ndarray, orientation_max: float, peak_thr: float) -> list[
    Keypoint]:
    oriented_keypoints = []
    bins_number = len(hist)
    threshold_max_peak = orientation_max * peak_thr
    for i in range(bins_number):
        left_peak = hist[i - 1]
        right_peak = hist[(i + 1) % bins_number]
        center_peak = hist[i]
        if left_peak < center_peak < right_peak and right_peak > threshold_max_peak:
            # https: // ccrma.stanford.edu / ~jos / sasp / Quadratic_Interpolation_Spectral_Peaks.html 6.30
            interpolated_bin = (i + 0.5 * (left_peak - right_peak) / (
                    left_peak - 2 * center_peak + right_peak)) % bins_number
            angle = interpolated_bin * 360 / bins_number
            oriented_keypoint = Keypoint(pos_x=keypoint.pos_x, pos_y=keypoint.pos_y, size=keypoint.size,
                                         response=keypoint.response, angle=angle)
            oriented_keypoints.append(oriented_keypoint)
    return oriented_keypoints


def get_keypoint_main_orientation(keypoint: Keypoint, gaussian_image: np.ndarray, octave_index: int,
                                  hist_bin_number: int = 36) -> (
        float, np.ndarray):
    scale = keypoint.size * 1.5 / np.float32(2 ** (octave_index + 1))
    radius = int(round(scale * 2))
    weight_factor = -0.5 * (scale ** -2)
    hist = np.zeros(hist_bin_number)
    img_h, img_w = gaussian_image.shape
    for i in range(-radius, radius + 1):
        y = int(keypoint.pos_y / np.float32(2 ** octave_index)) + i
        if 0 < y < img_h - 1:
            for j in range(-radius, radius + 1):
                x = int(keypoint.pos_x / np.float32(2 ** octave_index)) + j
                if 0 < x < img_w - 1:
                    dx = gaussian_image[y, x + 1] - gaussian_image[y, x - 1]
                    dy = gaussian_image[y + 1, x] - gaussian_image[y - 1, x]
                    magnitude = np.sqrt(dx ** 2 + dy ** 2)
                    orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    hist_index = int(np.round(orientation * hist_bin_number / 360)) % hist_bin_number
                    hist[hist_index] += magnitude * weight
    smooth_hist = np.array(
        [6 * hist[n] + hist[n - 1] + hist[n - 2] + 4 * hist[(n + 1) % hist_bin_number] + hist[(n + 2) % hist_bin_number] for n in
         range(hist_bin_number)])
    smooth_hist *= 1 / 16
    orientation_max = max(smooth_hist)
    return orientation_max, smooth_hist


def generate_descriptor():
    pass


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
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[
            0]  # sole, taylor shit, answer is (1,3) vector
        # http://www.ipol.im/pub/art/2014/82/  3.2
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
            size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (
                    2 ** (octave_index + 1))
            response = abs(response)

            return Keypoint(pos_x, pos_y, size, response), image_index
    return None


def is_scale_extremum(prev_pixels: np.ndarray, cur_pixels: np.ndarray, next_pixels: np.ndarray, thr) -> bool:
    center_pixel_value = cur_pixels[1, 1]
    if center_pixel_value > thr:
        max_prev = np.max(prev_pixels)
        max_cur = np.max(cur_pixels)
        max_next = np.max(next_pixels)
        return center_pixel_value >= max_prev and center_pixel_value >= max_cur and center_pixel_value > max_next
    return False



