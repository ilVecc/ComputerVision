import cv2.cv2 as cv
import numpy as np

from utils.algorithms import cylindrical_warp


def test_1():
    img = cv.imread('imgs/r.jpg')
    K = np.eye(3)
    K[0, 0] = 480
    K[1, 1] = 480
    K[0, 2] = 256
    K[1, 2] = 256
    return img, K


def test_2():
    img = cv.imread('../imgs/colosseum/low_res/20220104_113134.jpg')
    K = np.eye(3)
    K[0, 0] = 3215
    K[1, 1] = 3256
    K[0, 2] = 2175 // 2  # for low resolution, which is half the original
    K[1, 2] = 1725 // 2
    K[0, 1] = 10
    return img, K


def test_3():
    img = cv.imread('../imgs/stadium/low_res_1/20220104_160643.jpg')
    K = np.eye(3)
    K[0, 0] = 3215
    K[1, 1] = 3256
    K[0, 2] = 2175 // 2  # for low resolution, which is half the original
    K[1, 2] = 1725 // 2
    K[0, 1] = 10
    return img, K


if __name__ == '__main__':
    img, K = test_3()
    img_warp = cylindrical_warp(img, K)
    cv.imwrite('imgs/warp.png', img_warp)
