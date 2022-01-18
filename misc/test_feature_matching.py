import cv2.cv2 as cv
import numpy as np

img1 = cv.imread("imgs/test_feature/img2.jpg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("imgs/test_feature/img1.jpg", cv.IMREAD_GRAYSCALE)

# -- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# -- Step 2: Matching descriptor vectors with a FLANN based matcher
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
knn_matches = flann.knnMatch(des1, des2, k=2)

# -- Filter matches using the Lowe's ratio test
ratio_thresh = 0.7
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
        
# -- Draw matches
img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# -- Save detected matches
cv.imwrite('imgs/matching.png', img_matches)
