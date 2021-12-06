import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from stitching.stitcher import ImagePatch

# https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

img_patch_1 = ImagePatch('../imgs/panorama/panorama_1.jpg', load=True)  # queryImage
img_patch_2 = ImagePatch('../imgs/panorama/panorama_2.jpg', load=True)  # trainImage

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(img_patch_1.descriptors, img_patch_2.descriptors, k=2)
# Need to draw only good matches, so create a mask with ratio test as per Lowe's paper
matchesMask = [[1, 0] if m.distance < 0.7 * n.distance else [0, 0] for m, n in matches]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img_patch_1.img, img_patch_1.keypoints, img_patch_2.img, img_patch_2.keypoints, matches, None, **draw_params)
plt.imshow(img3), plt.show()








good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

src_pts = np.float32([img_patch_1.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([img_patch_2.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()
h, w, d = img_patch_1.img.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv.perspectiveTransform(pts, M)
img2 = cv.polylines(img_patch_2.img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)
img3 = cv.drawMatches(img_patch_1.img, img_patch_1.keypoints, img_patch_2.img, img_patch_2.keypoints, good, None, **draw_params)
plt.imshow(img3, 'gray'), plt.show()
