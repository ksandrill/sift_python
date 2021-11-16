from timeit import timeit

import cv2
import numpy as np

from sift_algo_phases.sift import sift_calculation
from util.keypoint import Keypoint


def match_keypoints(keypoints1: list[Keypoint], descriptors1: list[np.ndarray], keypoints2: list[Keypoint],
                    descriptors2: list[np.ndarray], cross_check_flag: bool = False) -> list[
    ((Keypoint, np.ndarray), (Keypoint, np.ndarray))]:
    points1 = list(zip(keypoints1, descriptors1))
    points2 = list(zip(keypoints2, descriptors2))
    matched1 = find_match_pairs(points1, points2)
    if not cross_check_flag:
        return matched1
    matched2 = find_match_pairs(points2, points1)
    return cross_check(matched1, matched2)


def cross_check(matched1: list[
    ((Keypoint, np.ndarray), (Keypoint, np.ndarray))],
                matched2: list[
                    ((Keypoint, np.ndarray), (Keypoint, np.ndarray))]) -> list[
    ((Keypoint, np.ndarray), (Keypoint, np.ndarray))]:
    matched = []
    for (point1_a, point1_b), (point2_a, point2_b) in zip(matched1, matched2):
        _, desk1_a = point1_a
        _, desk1_b = point1_b
        _, desk2_a = point2_a
        _, desk2_b = point2_b
        if np.array_equal(desk1_a, desk2_b) and np.array_equal(desk1_b, desk2_a):
            matched.append((point1_a, point1_b))
    return matched


def find_match_pairs(points1: list[(Keypoint, np.ndarray), (Keypoint, np.ndarray)],
                     points2: list[(Keypoint, np.ndarray), (Keypoint, np.ndarray)]) -> list[
    ((Keypoint, np.ndarray), (Keypoint, np.ndarray))]:
    matched = []
    for point_1 in points1:
        point = points2[0]
        descriptor_1 = point_1[1]
        point_descriptor = point[1]
        dist = np.linalg.norm((descriptor_1 - point_descriptor))
        for point_2 in points2[1:]:
            descriptor_2 = point_2[1]
            new_dist = np.linalg.norm((descriptor_1 - descriptor_2))
            if new_dist <= dist:
                dist = new_dist
                point = point_2
        matched.append((point_1, point))
    return matched


def pick_matched_points(matched_points: list[((Keypoint, np.ndarray), (Keypoint, np.ndarray))],
                        picture1: np.ndarray, picture2: np.ndarray, show_keypoint_info: bool) -> np.ndarray:
    h1, w1, _ = picture1.shape
    union_picture = create_union_image(picture1, picture2)
    for matched in matched_points:
        point1, point2 = matched
        keypoint1, _ = point1
        keypoint2, _ = point2
        size1 = keypoint1.get_input_size()
        size2 = keypoint2.get_input_size()
        x1 = keypoint1.get_input_image_pos_x()
        y1 = keypoint1.get_input_image_pos_y()
        x2 = keypoint2.get_input_image_pos_x() + w1 + 1
        y2 = keypoint2.get_input_image_pos_y()
        radius1 = int(size1 / 2)
        radius2 = int(size2 / 2)
        draw_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        if show_keypoint_info:
            cv2.circle(union_picture, (x1, y1), radius1, draw_color, 1)
            cv2.circle(union_picture, (x2, y2), radius2, draw_color, 1)
            draw_arrow(union_picture, (x1, y1), (
                int(x1 + radius1 * np.cos(360 - keypoint1.angle)), int(y1 + radius1 * np.sin(360 - keypoint1.angle))),
                       draw_color)
            draw_arrow(union_picture, (x2, y2), (
                int(x2 + radius2 * np.cos(360 - keypoint1.angle)), int(y2 + radius2 * np.sin(360 - keypoint1.angle))),
                       draw_color)
        cv2.line(union_picture, (x1, y1), (x2, y2), draw_color, 1, 8, 0)
    return union_picture


def draw_arrow(image: np.ndarray, p: (int, int), q: (int, int), color: (np.uint8, np.uint8, np.uint8),
               arrow_magnitude: int = 9, thickness: int = 1, line_type: int = 8, shift: int = 0) -> None:
    cv2.line(image, p, q, color, thickness, line_type, shift)
    angle = np.arctan2(p[1] - q[1], p[0] - q[0])
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi / 4)),
         int(q[1] + arrow_magnitude * np.sin(angle + np.pi / 4)))
    cv2.line(image, p, q, color, thickness, line_type, shift)
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi / 4)),
         int(q[1] + arrow_magnitude * np.sin(angle - np.pi / 4)))
    cv2.line(image, p, q, color, thickness, line_type, shift)


def create_color_gray(picture: np.ndarray):
    color_gray = np.zeros((picture.shape[0], picture.shape[1], 3))
    for i in range(color_gray.shape[0]):
        for j in range(color_gray.shape[1]):
            color_gray[i][j] = [picture[i][j], picture[i][j], picture[i][j]]
    return color_gray


def create_union_image(picture1: np.ndarray, picture2: np.ndarray):
    h1, w1, channels1 = picture1.shape
    h2, w2, channels1 = picture2.shape
    h_diff = abs(h1 - h2)
    max_h = max(h1, h2)
    new_h = max_h + h_diff
    new_w = w1 + w2 + 1
    union_picture = np.zeros((new_h, new_w, channels1))
    union_picture[0:h1, 0:w1] = picture1
    union_picture[0:h2, w1 + 1:new_w] = picture2
    return union_picture.astype(np.uint8)


def main():
    picture1 = cv2.resize(cv2.imread('pictures/a.jpg', 0), (300, 300))
    picture2 = cv2.resize(cv2.imread('pictures/b.jpg', 0), (300, 300))
    keypoints1, descriptors1 = sift_calculation(picture1, contrast_threshold=0.01, octave_number=11, num_intervals=4,
                                                sigma=1.3)
    keypoints2, descriptors2 = sift_calculation(picture2, contrast_threshold=0.01, octave_number=11, num_intervals=4,
                                                sigma=1.3)
    print('keypoints1: ', len(descriptors1))
    print('keypoints2: ', len(descriptors2))
    color_gray1 = create_color_gray(picture1)
    color_gray2 = create_color_gray(picture2)
    matched_points = match_keypoints(keypoints1, descriptors1, keypoints2, descriptors2, cross_check_flag=True)
    union_picture_with_matching = pick_matched_points(matched_points, color_gray1, color_gray2, False)
    cv2.imshow('dafaq', union_picture_with_matching)
    cv2.waitKey(0)
    cv2.imwrite('examples/example5.jpg', union_picture_with_matching)


if __name__ == '__main__':
    print('time: ', timeit(lambda: main(), number=1))
