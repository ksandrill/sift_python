from timeit import timeit

import cv2
import numpy as np
import gauss_filter
from sift import get_dog


def sift(picture: np.ndarray) -> None:
    dog = get_dog(picture, octaves_number=8)
    for i, octave in enumerate(dog):
        for j, img in enumerate(octave):
            print(img[img < 0])


def main() -> None:
    picture = cv2.imread('pictures/Lenna.png')
    picture = gauss_filter.gaussian_blur(picture, (5, 5), 2)
    sift(picture)
    # cv2.imshow('dafaq', picture)
    # cv2.waitKey(0)


if __name__ == '__main__':
    print('time: ', timeit(lambda: main(), number=1))
