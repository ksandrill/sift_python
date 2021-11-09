import numpy as np


def picture_to_gray(picture: np.ndarray, is_bgr: bool = True):
    h, w, channels = picture.shape
    if channels != 3:
        raise Exception('channles should be')
    rgb_picture = picture.copy()
    if is_bgr:
        rgb_picture = rgb_picture[:, :, [2, 1, 0]]
    output = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            r, g, b = rgb_picture[i][j]
            output[i][j] = 0.299 * r + 0.587 * g + 0.114 * b
    return output.astype(np.uint8)


def convolution(image: np.ndarray, kernel: np.ndarray, average: bool = False, is_bgr: bool = True) -> np.ndarray:
    if len(image.shape) == 3:
        image = picture_to_gray(image, is_bgr)
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output


def normalize_image(x: np.ndarray, a: float, b: float):
    min_val = np.min(x)
    max_val = np.max(x)
    return a + (x - min_val) * (b - a) / (max_val - np.min(x))



def nearest_neighbor_sampling(img: np.ndarray, step: int = 2):
    return img[::step, ::step]
