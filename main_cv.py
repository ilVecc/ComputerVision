import cv2.cv2 as cv
from pathlib import Path

# https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html

imgs_set = "venice"
folder = f"imgs/{imgs_set}/low_res"

imgs = []
for img_path in Path(folder).iterdir():
    img = cv.imread(str(img_path))
    imgs.append(img)

stitcher = cv.Stitcher.create(cv.STITCHER_PANORAMA)
status, result = stitcher.stitch(imgs)

cv.imwrite(f"imgs_results/cv_mosaic_{imgs_set}.png", result)
