import numpy as np
FLOAT_TOLERANCE = 1e-7


def calc_center_hessian(pixels: np.ndarray) -> np.ndarray:
    # f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # df/dxdy (x,y)  =(f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # x - third dim, y - second dim,  scale - first dim
    # hessian = [dxx, dxy, dxs]
    #           [dxy, dyy, dys]
    #           [dxs, dys, dss]
    dxx = pixels[1, 1, 2] - 2 * pixels[1, 1, 1] + pixels[1, 1, 0]
    dyy = pixels[1, 2, 1] - 2 * pixels[1, 1, 1] + pixels[1, 0, 1]
    dss = pixels[2, 1, 1] - 2 * pixels[1, 1, 1] + pixels[0, 1, 1]
    dxy = (pixels[1, 2, 2] - pixels[1, 2, 0] - pixels[1, 0, 2] + pixels[1, 0, 0]) * 0.25
    dxs = (pixels[2, 1, 2] - pixels[0, 1, 2] - pixels[2, 1, 0] + pixels[0, 1, 0]) * 0.25
    dys = (pixels[2, 2, 1] - pixels[2, 0, 1] - pixels[0, 2, 1] + pixels[0, 0, 1]) * 0.25
    return np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])


def calc_center_gradient(pixels: np.ndarray) -> np.ndarray:
    # f'(x) = (f(x + 1) - f(x - 1)) / 2
    # blur - change distance or zooming image so blur is about scale
    # calc gradient in [1,1,1], x - third dim, y - second dim,  scale - first dim
    dx = 0.5 * (pixels[1, 1, 2] - pixels[1, 1, 0])
    dy = 0.5 * (pixels[1, 2, 1] - pixels[1, 0, 1])
    ds = 0.5 * (pixels[2, 1, 1] - pixels[0, 1, 1])
    return np.array([dx, dy, ds])


def nearest_neighbor_sampling(img: np.ndarray, step: int = 2):
    return img[::step, ::step]
