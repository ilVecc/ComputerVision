import cv2.cv2 as cv
from pathlib import Path


folder = "imgs/library"
imgs = []

for img_path in Path(folder).iterdir():
    img = cv.imread(str(img_path))
    imgs.append(img)

stitcher = cv.Stitcher.create(cv.STITCHER_PANORAMA)
status, result = stitcher.stitch(imgs)

cv.imshow("mosaic", cv.resize(result, (1366, 768)))
cv.waitKey(0)
cv.destroyAllWindows()




