#!/usr/bin/python
import argparse
import sys

import cv2.cv2 as cv2
import numpy as np


def preprocess(img1, img2, overlap_w):
    if img1.shape[0] != img2.shape[0]:
        print("error: image dimension error")
        sys.exit()
    if overlap_w > img1.shape[1] or overlap_w > img2.shape[1]:
        print("error: overlapped area too large")
        sys.exit()
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # prepare final dimensions
    offset = 100
    shape = np.array(img1.shape)
    shape[0] += offset
    shape[1] = w1 + w2 - overlap_w
    
    # patch1_in_mosaic and patch1_mask_in_mosaic
    subA = np.zeros(shape)
    subA[offset:offset+h1, :w1, :] = img1
    
    # patch2_in_mosaic and patch2_mask_in_mosaic
    subB = np.zeros(shape)
    subB[:h2, w1 - overlap_w:, :] = img2
    
    maskA = np.zeros(shape)
    maskA[offset:offset+h1, :w1 - overlap_w // 2] = 1
    
    return subA, subB, maskA


def prepare_multi_band_blending(img1, img2, overlap_w, leveln=None):
    if overlap_w < 0:
        print("error: overlap_w should be a positive integer")
        sys.exit()
        
    max_leveln = int(np.floor(np.log2(min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1]))))
    if leveln is None:
        leveln = max_leveln
    if leveln < 1 or leveln > max_leveln:
        print("warning: inappropriate number of leveln")
        leveln = max_leveln
    
    subA, subB, mask = preprocess(img1, img2, overlap_w)
    return subA, subB, mask, leveln


def multi_band_blending(A, B, M, num_levels):
    
    num_levels = int(np.floor(np.log2(min(A.shape[0], B.shape[1]))))
    
    # gaussian pyramid
    gpA = [np.float32(A.copy())]
    gpB = [np.float32(B.copy())]
    gpM = [np.float32(M.copy())]
    for i in range(num_levels):
        gpA.append(cv2.pyrDown(gpA[i]))
        gpB.append(cv2.pyrDown(gpB[i]))
        gpM.append(cv2.pyrDown(gpM[i]))
    
    # laplacian pyramid
    gpMr = gpM[::-1]
    lpA = [gpA[num_levels]]
    lpB = [gpB[num_levels]]
    for i in range(num_levels, 0, -1):
        size = gpA[i - 1].shape[1::-1]
        LA = gpA[i - 1] - cv2.pyrUp(gpA[i], dstsize=size)
        LB = gpB[i - 1] - cv2.pyrUp(gpB[i], dstsize=size)
        lpA.append(LA)
        lpB.append(LB)
    
    # blend
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1 - gm)
        LS.append(ls)
    
    # reconstruct
    ls_ = LS[0]
    for lev_img in LS[1:]:
        ls_ = lev_img + cv2.pyrUp(ls_, dstsize=lev_img.shape[1::-1])
    ls_ = np.clip(ls_, 0, 255)
    
    return np.uint8(ls_)


if __name__ == '__main__':
    
    test = True
    
    if test:
        img1 = cv2.imread('imgs/test_fruit/l.jpg')
        img2 = cv2.imread('imgs/test_fruit/r.jpg')
        overlap_w = 256  # blend width
        imgs = prepare_multi_band_blending(img1, img2, overlap_w)
    else:
        img1 = cv2.imread('imgs/test_colosseum/img1.png')
        img2 = cv2.imread('imgs/test_colosseum/img2.png')
        mask1 = cv2.imread('imgs/test_colosseum/mask1.png') / 255
        # mask2 = cv2.imread('imgs/test_colosseum/mask2.png') / 255
        img1 = img1.astype(np.float64)  # [0, 255]
        img2 = img2.astype(np.float64)  # [0, 255]
        mask1 = mask1 == 1
        mask2 = np.bitwise_not(mask1)
        leveln = 8
        imgs = (img1, img2, mask1, mask2, 8)
    
    result = multi_band_blending(*imgs)
    cv2.imwrite('imgs/result.png', result)
