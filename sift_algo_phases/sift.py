from typing import Optional

import numpy as np

from sift_algo_phases.descriptor_phase import get_descriptors
from sift_algo_phases.dog_phase import generate_sigmas, get_dog, get_gaussian_pyramid
from util.keypoint import remove_duplitcates, Keypoint
from sift_algo_phases.keypoint_phase import get_keypoints


def sift_calculation(image: np.ndarray, octave_number: Optional[int] = None, sigma: float = 1.6,
                     num_intervals: int = 3,
                     image_border_width: int = 5) -> (list[Keypoint], np.ndarray):
    if octave_number is None:
        octave_number = compute_octave_number(image.shape)
    gaussian_sigmas = generate_sigmas(sigma, num_intervals, octave_number)
    image = image.astype(np.float32)
    gaussian_pyramid = get_gaussian_pyramid(image, octave_number, gaussian_sigmas)
    dog_images = get_dog(gaussian_pyramid)
    keypoints = get_keypoints(gaussian_pyramid, dog_images, num_intervals, sigma, image_border_width)
    keypoints = remove_duplitcates(keypoints)
    descriptors = get_descriptors(keypoints, gaussian_pyramid)
    return keypoints, descriptors


def compute_octave_number(image_shape) -> int:  # from openCV2 impl
    return int(round(np.log(min(image_shape)) / np.log(2) - 1))
