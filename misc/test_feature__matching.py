import cv2.cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils.homography import fit_homography, distance_homography, test_degenerate_samples
from utils.ransac import ransac

img1 = cv.imread('imgs/test_feature/img1.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('imgs/test_feature/img2.jpg', cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv.DrawMatchesFlags_DEFAULT)

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.imshow(img3)
plt.show()









matches = np.array(matches)
matchesMask = np.array(matchesMask, dtype=bool)[:, 0]

good_matches = matches[matchesMask, 0]

# pairs is (n, m, 3), with  n  2D homogeneous points for each of the  m  sets of selected matches
pairs = np.ones(shape=(len(good_matches), 2, 3))
pairs[:, :, 0:2] = [(kp1[match.queryIdx].pt, kp2[match.trainIdx].pt) for match in good_matches]

best_H, curr_iter, max_iter = None, 0, 10
k, th = 2000, 2
while best_H is None or curr_iter < max_iter:
    # samples = 4  because it's the minimum required to estimate an homography
    # providing more samples to RANSAC will most definitely result in problematic H outputs and glitches
    best_H, best_inliers, _ = ransac(
        pairs,
        max_iter=k, thresh=th, samples=4,
        fit_fun=fit_homography, dist_fun=distance_homography, test_samples=test_degenerate_samples
    )
    curr_iter += 1
    if best_H is not None:
        break
    print(f"Could not find an homography. Parameters have been relaxed.")
    k += 100
    th += 0.1





final_matches = tuple([(match, ) for match, keep in zip(good_matches, best_inliers) if keep])

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   flags=cv.DrawMatchesFlags_DEFAULT)

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, final_matches, None, **draw_params)

plt.imshow(img3)
plt.show()
