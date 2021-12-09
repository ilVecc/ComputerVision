from stitching.stitcher import ImageStitching, CrossoverMethod, SeamMethod, HomographyMethod
import cv2.cv2 as cv

# TODO remove file info.md

if __name__ == '__main__':
    stitcher = ImageStitching(
        homography_method=HomographyMethod.MANUAL_IMPL,
        seam_method=SeamMethod.ENERGY_BASED,
        crossover_method=CrossoverMethod.AVERAGE)
    stitcher.process_folder("imgs/river")
    
    stitcher.stitch_all()
    
    cv.imshow("mosaic", stitcher.mosaic)
    cv.waitKey(0)
    cv.destroyAllWindows()
