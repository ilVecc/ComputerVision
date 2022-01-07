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
    
    w1 = img1.shape[1]
    w2 = img2.shape[1]
    
    # prepare final dimensions
    shape = np.array(img1.shape)
    shape[1] = w1 + w2 - overlap_w
    
    # patch1_in_mosaic
    subA = np.zeros(shape)
    subA[:, :w1, :] = img1
    # patch2_in_mosaic
    subB = np.zeros(shape)
    subB[:, w1 - overlap_w:, :] = img2
    # patch1_non_shared_mask
    mask = np.zeros(shape)
    mask[:, :w1 - overlap_w // 2] = 1
    
    return subA, subB, mask


def GaussianPyramid(img, leveln):
    GP = [img]
    for i in range(leveln - 1):
        GP.append(cv2.pyrDown(GP[i]))
    return GP


def LaplacianPyramid(img, leveln):
    LP = []
    for i in range(leveln - 1):
        next_img = cv2.pyrDown(img)
        LP.append(img - cv2.pyrUp(next_img, dstsize=img.shape[1::-1]))
        img = next_img
    LP.append(img)
    return LP


def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended


def reconstruct(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.pyrUp(img, dstsize=lev_img.shape[1::-1])
        img += lev_img
    img = np.clip(img, 0, 255)
    return img


def multi_band_blending(img1, img2, overlap_w, leveln=None):
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
    
    # Get Gaussian pyramid and Laplacian pyramid
    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(subA, leveln)
    LPB = LaplacianPyramid(subB, leveln)
    
    # Blend two Laplacian pyramidspass
    blended_pyramids = blend_pyramid(LPA, LPB, MP)
    
    # Reconstruction process
    result = reconstruct(blended_pyramids)
    
    return result


if __name__ == '__main__':
    img1 = cv2.imread('imgs/l.jpg')
    img2 = cv2.imread('imgs/r.jpg')
    overlap_w = 256  # blend width
    
    result = multi_band_blending(img1, img2, overlap_w)
    cv2.imwrite('imgs/result.png', result)
