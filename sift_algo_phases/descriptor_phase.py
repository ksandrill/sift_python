import numpy as np

from util.keypoint import Keypoint
from util.util import FLOAT_TOLERANCE


def get_descriptors(keypoints: list[Keypoint], gaussian_images: np.ndarray, window_width: int = 4, num_bins: int = 8,
                    scale_multiplier: float = 3,
                    descriptor_max_value=0.2):
    descriptors = []
    for keypoint in keypoints:
        descriptor_vector = calc_descriptor(keypoint, gaussian_images, num_bins, window_width,
                                            scale_multiplier)
        # Threshold and normalize descriptor_vector
        descriptor_vector = norm_descriptor(descriptor_vector, descriptor_max_value)
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')


def calc_descriptor(keypoint: Keypoint, gaussian_images: np.ndarray, num_bins, window_width: int,
                    scale_multiplier: float) -> np.ndarray:
    octave = keypoint.octave
    layer = keypoint.layer
    gaussian_image = gaussian_images[octave + 1, layer]
    num_rows, num_cols = gaussian_image.shape
    point = (keypoint.pos_x, keypoint.pos_y)
    angle = 360. - keypoint.angle
    # half_width  and border by openCV implementation
    hist_width = scale_multiplier * 0.5 * keypoint.size
    half_width = int(
        round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))  # sqrt(2) corresponds to diagonal length of a pixel
    half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))  # ensure half_width lies within image
    row_bin_list, col_bin_list, magnitude_list, orientation_bin_list = calc_keypoints_params_for_descriptor_hist(
        gaussian_image, point, hist_width, half_width, window_width, angle, num_bins)
    hist = calc_descriptor_hist(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list, num_bins,
                                window_width)
    return hist[1:-1, 1:-1, :].flatten()  # Remove histogram borders


def calc_keypoints_params_for_descriptor_hist(gaussian_image: np.ndarray, point: (int, int), hist_width: float,
                                              half_width: int, window_width: int,
                                              angle: float, num_bins: int) -> (
        list[float], list[float], list[float], list[float]):
    num_rows, num_cols = gaussian_image.shape
    bins_per_degree = num_bins / 360.
    cos_angle = np.cos(np.deg2rad(angle))
    sin_angle = np.sin(np.deg2rad(angle))
    weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
    row_bin_list = []
    col_bin_list = []
    magnitude_list = []
    orientation_bin_list = []
    for row in range(-half_width, half_width + 1):
        for col in range(-half_width, half_width + 1):
            row_rot = col * sin_angle + row * cos_angle
            col_rot = col * cos_angle - row * sin_angle
            row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
            col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
            if -1 < row_bin < window_width and -1 < col_bin < window_width:
                window_row = int(round(point[1] + row))
                window_col = int(round(point[0] + col))
                if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                    dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                    dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                    weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                    row_bin_list.append(row_bin)
                    col_bin_list.append(col_bin)
                    magnitude_list.append(weight * gradient_magnitude)
                    orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)
    return row_bin_list, col_bin_list, magnitude_list, orientation_bin_list


def calc_descriptor_hist(row_bin_list: list[float], col_bin_list: list[float], magnitude_list: list[float],
                         orientation_bin_list: list[float], num_bins: int,
                         window_width: int):
    hist = np.zeros((window_width + 2, window_width + 2,
                     num_bins))  # first two dimensions are increased by 2 to account for border effects
    for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list,
                                                            orientation_bin_list):
        # inverse of trilinear interpolation here (take center value of the cube and distribute it among its eight
        # neighbors) https://en.wikipedia.org/wiki/Trilinear_interpolation
        row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
        row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
        if orientation_bin_floor < 0:
            orientation_bin_floor += num_bins
        if orientation_bin_floor >= num_bins:
            orientation_bin_floor -= num_bins

        c1 = magnitude * row_fraction
        c0 = magnitude * (1 - row_fraction)
        c11 = c1 * col_fraction
        c10 = c1 * (1 - col_fraction)
        c01 = c0 * col_fraction
        c00 = c0 * (1 - col_fraction)
        c111 = c11 * orientation_fraction
        c110 = c11 * (1 - orientation_fraction)
        c101 = c10 * orientation_fraction
        c100 = c10 * (1 - orientation_fraction)
        c011 = c01 * orientation_fraction
        c010 = c01 * (1 - orientation_fraction)
        c001 = c00 * orientation_fraction
        c000 = c00 * (1 - orientation_fraction)

        hist[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
        hist[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
        hist[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
        hist[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
        hist[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
        hist[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
        hist[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
        hist[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111
    return hist


def norm_descriptor(descriptor_vector: np.ndarray, descriptor_max_value) -> np.ndarray:
    # from openCV implementation
    threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
    descriptor_vector[descriptor_vector > threshold] = threshold
    descriptor_vector /= max(np.linalg.norm(descriptor_vector), FLOAT_TOLERANCE)
    descriptor_vector = np.round(512 * descriptor_vector)
    descriptor_vector[descriptor_vector < 0] = 0
    descriptor_vector[descriptor_vector > 255] = 255
    return descriptor_vector