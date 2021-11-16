from util.util import nearest_neighbor_sampling
import numpy as np
import cv2


def generate_sigmas(start_sigma: float, num_intervals: int, num_octaves: int) -> list[list[float]]:
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    return [[start_sigma * (k ** image_index) * (1 << octave_index) for image_index in range(num_images_per_octave)] for
            octave_index in range(num_octaves)]


def get_gaussian_pyramid(image: np.ndarray, num_octaves: int, gauss_sigmas: list[list[float]]) -> np.ndarray:
    gaussian_pyramid = []
    #images = [nearest_neighbor_sampling(image, 1 << octave_ind) for octave_ind in range(num_octaves)]
    # for octave_index in range(num_octaves):
    #     octave_images = []
    #     for gauss_sigma in gauss_sigmas[octave_index]:
    #         kernel_size = int(np.ceil(3 * gauss_sigma) * 2 + 1)
    #         image = cv2.GaussianBlur(images[octave_index], (kernel_size, kernel_size), sigmaX=gauss_sigma,
    #                                  sigmaY=gauss_sigma)
    #         octave_images.append(image)
    #     gaussian_pyramid.append(octave_images)
    for octave_index in range(num_octaves):
        gaussian_images_in_octave = [image]
        for gaussian_sigma in gauss_sigmas[octave_index]:
            kernel_size = int(np.ceil(3 * gaussian_sigma) * 2 + 1)
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)
            gaussian_images_in_octave.append(image)
        gaussian_pyramid.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[0]
        image = nearest_neighbor_sampling(octave_base, 1 << octave_index)
    return np.array(gaussian_pyramid, dtype=np.ndarray)


def get_dog(gaussian_pyramid: np.ndarray) -> np.ndarray:
    octaves_number, octaves_size = gaussian_pyramid.shape
    dog = [[gaussian_pyramid[octave][pos + 1] - gaussian_pyramid[octave][pos] for pos in
            range(octaves_size - 1)] for
           octave in range(octaves_number)]
    return np.array(dog, dtype=np.ndarray)
