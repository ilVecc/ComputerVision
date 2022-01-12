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
    maskA = np.zeros(shape)
    maskA[offset:offset+h1, :w1 - overlap_w // 2] = 1
    
    # patch2_in_mosaic and patch2_mask_in_mosaic
    subB = np.zeros(shape)
    subB[:h2, w1 - overlap_w:, :] = img2
    maskB = np.zeros(shape)
    maskB[:h2, w1 - overlap_w // 2:] = 1
    
    return subA, subB, maskA, maskB


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


def blend_pyramid(LPA, LPB, MPA, MPB):
    blended = []
    for LA, LB, MA, MB in zip(LPA, LPB, MPA, MPB):
        blended.append(LA * MA + LB * MB)
    return blended


def reconstruct(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.pyrUp(img, dstsize=lev_img.shape[1::-1])
        img += lev_img
    img = np.clip(img, 0, 255)
    return img


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
    
    subA, subB, maskA, maskB = preprocess(img1, img2, overlap_w)
    return subA, subB, maskA, maskB, leveln


def multi_band_blending(imgA, imgB, maskA, maskB, leveln):
    
    # Get Gaussian pyramid and Laplacian pyramid
    MPA = GaussianPyramid(maskA, leveln)
    MPB = GaussianPyramid(maskB, leveln)
    LPA = LaplacianPyramid(imgA, leveln)
    LPB = LaplacianPyramid(imgB, leveln)
    
    # Blend two Laplacian pyramids
    blended_pyramids = blend_pyramid(LPA, LPB, MPA, MPB)
    
    # Reconstruction process
    result = reconstruct(blended_pyramids)
    mask = np.bitwise_or(np.bool_(maskA), np.bool_(maskB))
    result = result * mask
    
    return result


if __name__ == '__main__':
    
    test = False
    
    if test:
        img1 = cv2.imread('imgs/test_fruit/l.jpg')
        img2 = cv2.imread('imgs/test_fruit/r.jpg')
        overlap_w = 256  # blend width
        imgs = prepare_multi_band_blending(img1, img2, overlap_w)
    else:
        img1 = cv2.imread('imgs/test_colosseum/img1.png')
        img2 = cv2.imread('imgs/test_colosseum/img2.png')
        mask1 = cv2.imread('imgs/test_colosseum/mask1.png') / 255
        mask2 = cv2.imread('imgs/test_colosseum/mask2.png') / 255
        mask1 -= mask2
        img1 = img1.astype(np.float64)  # [0, 255]
        img2 = img2.astype(np.float64)  # [0, 255]
        mask1 = mask1.astype(np.float64)  # [0, 1]
        mask2 = mask2.astype(np.float64)  # [0, 1]
        leveln = 8
        imgs = (img1, img2, mask1, mask2, 8)
    
    result = multi_band_blending(*imgs)
    cv2.imwrite('imgs/result.png', result)
