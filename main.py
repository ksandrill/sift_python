from timeit import timeit

import cv2
import numpy as np
import gauss_filter
from sift import get_dog, get_keypoints


def sift(picture: np.ndarray):
    dog = get_dog(picture, octaves_number=8)
    keypoints = get_keypoints(dog)
    return keypoints


def main() -> None:
    picture = cv2.imread('pictures/Lenna.png')
    orig_image = picture.copy()
    picture = gauss_filter.gaussian_blur(picture, (5, 5), 2)
    keypoints = sift(picture)
    for keypoint in keypoints:
        x = keypoint.pos_x
        y = keypoint.pos_y
        print(x)
        print(y)
        orig_image[y][x] = [0, 0, 255]
    cv2.imshow('dafaq', orig_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    print('time: ', timeit(lambda: main(), number=1))
