import cv2.cv2 as cv
import numpy as np

src = cv.imread("imgs/test_feature/img2.jpg", cv.IMREAD_GRAYSCALE)

# -- Step 1: Detect the keypoints using SURF Detector
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(src, None)

# -- Draw keypoints
img_keypoints = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
cv.drawKeypoints(src, kp, img_keypoints, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# -- Show detected (drawn) keypoints
cv.imwrite("imgs/keypoints.png", img_keypoints)
