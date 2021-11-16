from typing import Optional

from util.keypoint import Keypoint
from util.util import calc_center_gradient, calc_center_hessian, FLOAT_TOLERANCE
import numpy as np


def get_keypoints(gaussian_pyramid: np.ndarray, dog_images: np.ndarray, num_intervals: int, sigma,
                  image_border_width: int,
                  contrast_threshold):
    thr = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []
    for octave_index, dog_images_in_octave in enumerate(dog_images):
        h, w = dog_images_in_octave[0].shape
        for image_index in range(1, len(dog_images_in_octave) - 1):
            prev_image = dog_images_in_octave[image_index - 1]
            cur_image = dog_images_in_octave[image_index]
            next_image = dog_images_in_octave[image_index + 1]
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    prev_cube = prev_image[i - 1:i + 2, j - 1:j + 2]
                    cur_cube = cur_image[i - 1:i + 2, j - 1:j + 2]
                    next_cube = next_image[i - 1:i + 2, j - 1:j + 2]

                    if is_scale_extremum(prev_cube, cur_cube,
                                         next_cube, thr):
                        localized_keypoint = localize_keypoint(i, j, image_index, octave_index, num_intervals,
                                                               dog_images_in_octave,
                                                               sigma, contrast_threshold, image_border_width)
                        if localized_keypoint is not None:
                            keypoints += calc_oriented_keypoints(localized_keypoint,
                                                                 gaussian_pyramid[octave_index][
                                                                     localized_keypoint.layer])
    return keypoints


def is_scale_extremum(prev_pixels: np.ndarray, cur_pixels: np.ndarray, next_pixels: np.ndarray,
                      threshold: float) -> bool:
    center_pixel_value = cur_pixels[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return np.all(center_pixel_value >= prev_pixels) and \
                   np.all(center_pixel_value >= next_pixels) and \
                   np.all(center_pixel_value >= cur_pixels[0, :]) and \
                   np.all(center_pixel_value >= cur_pixels[2, :]) and \
                   center_pixel_value >= cur_pixels[1, 0] and \
                   center_pixel_value >= cur_pixels[1, 2]
        elif center_pixel_value < 0:
            return np.all(center_pixel_value <= prev_pixels) and \
                   np.all(center_pixel_value <= next_pixels) and \
                   np.all(center_pixel_value <= cur_pixels[0, :]) and \
                   np.all(center_pixel_value <= cur_pixels[2, :]) and \
                   center_pixel_value <= cur_pixels[1, 0] and \
                   center_pixel_value <= cur_pixels[1, 2]
    return False


def localize_keypoint(i: int, j: int, image_index: int, octave_index: int, num_intervals: int,
                      dog_images_in_octave: np.ndarray, sigma: float,
                      contrast_threshold: float, image_border_width: int, eigenvalue_ratio: float = 10,
                      num_attempts_until_convergence: int = 5) -> Optional[Keypoint]:
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        prev_image, cur_image, next_image = dog_images_in_octave[image_index - 1:image_index + 2]
        pixel_cube = np.stack([prev_image[i - 1:i + 2, j - 1:j + 2],
                               cur_image[i - 1:i + 2, j - 1:j + 2],
                               next_image[i - 1:i + 2, j - 1:j + 2]]).astype('float32') / 255.
        gradient = calc_center_gradient(pixel_cube)
        hessian = calc_center_hessian(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # new pixel_cube will lie entirely within the image
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
            pos_x = (j + extremum_update[0])
            pos_y = (i + extremum_update[1])
            size = sigma * (2 ** (image_index + extremum_update[2]))
            keypoint = Keypoint(pos_x=int(pos_x), pos_y=int(pos_y), octave=int(octave_index), angle=None,
                                response=response,
                                layer=image_index, size=size)
            return keypoint
    return None


def calc_orientation_histogram(gaussian_image: np.ndarray, num_bins: int,
                               keypoint_x: float, keypoint_y: float,
                               keypoint_size: float, scale_factor: float = 3,
                               radius_factor: float = 1.5) -> np.ndarray:
    image_shape = gaussian_image.shape
    hist = np.zeros(num_bins)
    scale = scale_factor * keypoint_size
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    for i in range(-radius, radius + 1):
        region_y = keypoint_y + i
        if 0 < region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = keypoint_x + j
                if 0 < region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    hist[histogram_index % num_bins] += weight * gradient_magnitude
    smooth_hist = np.array(  # smooth histogram with gauss filter size 5
        [6 * hist[n] + hist[n - 1] + hist[n - 2] + 4 * hist[(n + 1) % num_bins] + hist[(n + 2) % num_bins]
         for n in
         range(num_bins)])
    smooth_hist *= 1 / 16
    return smooth_hist


def calc_oriented_keypoints(keypoint: Keypoint, gaussian_image: np.ndarray, num_bins: int = 36,
                            peak_ratio: float = 0.8) -> list[Keypoint]:
    oriented_keypoints = []
    hist = calc_orientation_histogram(gaussian_image, num_bins, keypoint.pos_x, keypoint.pos_y,
                                      keypoint.size)
    orientation_max = np.max(hist)
    threshold_max_peak = orientation_max * peak_ratio
    for i in range(num_bins):
        left_peak = hist[i - 1]
        right_peak = hist[(i + 1) % num_bins]
        center_peak = hist[i]
        if left_peak < center_peak < right_peak and right_peak > threshold_max_peak:
            # https: // ccrma.stanford.edu / ~jos / sasp / Quadratic_Interpolation_Spectral_Peaks.html 6.30
            interpolated_bin = (i + 0.5 * (left_peak - right_peak) / (
                    left_peak - 2 * center_peak + right_peak)) % num_bins
            orientation = 360. - interpolated_bin * 360. / num_bins
            if abs(orientation - 360.) < FLOAT_TOLERANCE:
                orientation = 0
            new_keypoint = Keypoint(pos_x=keypoint.pos_x, pos_y=keypoint.pos_y, octave=keypoint.octave,
                                    response=keypoint.response, layer=keypoint.layer, size=keypoint.size,
                                    angle=orientation)
            oriented_keypoints.append(new_keypoint)
    return oriented_keypoints
