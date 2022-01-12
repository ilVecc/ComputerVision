# Standard imports
import cv2.cv2 as cv2
import numpy as np

# Read images
src = cv2.imread("imgs/test_poisson/airplane.jpg")
dst = cv2.imread("imgs/test_poisson/sky.jpg")

src = cv2.resize(src, (300, 194))
dst = cv2.resize(dst, (1000, 560))

# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)
poly = np.array([[4, 80], [30, 54], [151, 63], [254, 37], [298, 90], [272, 134], [43, 122]], np.int32)
cv2.fillPoly(src_mask, [poly], (255, 255, 255))

# This is where the CENTER of the airplane will be placed
center = (800, 100)

# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

# Save result
cv2.imshow("result", cv2.resize(output, (1366, 768)))
cv2.waitKey(0)
cv2.destroyAllWindows()
