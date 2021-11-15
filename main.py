from timeit import timeit

from sift_algo_phases.sift import sift_calculation
import cv2
import numpy as np


def main():
    picture1 = cv2.imread('pictures/robopenguin.bmp', 0)
    picture2 = cv2.imread('pictures/rotate_robopenguin.bmp', 0)
    keypoints1, descriptors1 = sift_calculation(picture1)
    keypoints2, descriptors2 = sift_calculation(picture2)
    # h1, w1 = picture1.shape
    # h2, w2 = picture2.shape
    # h_max = max(h1, h2)
    # w_max = max(w1, w2)
    # cv2.waitKey(0)
    #


if __name__ == '__main__':
    print('time: ', timeit(lambda: main(), number=1))
