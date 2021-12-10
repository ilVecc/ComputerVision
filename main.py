from stitching.stitcher import ImageStitching, StitchingMethod, SeamMethod, HomographyMethod
import cv2.cv2 as cv

# TODO remove file info.md

if __name__ == '__main__':
    stitcher = ImageStitching(
        homography_method=HomographyMethod.MANUAL_IMPL,
        seam_method=SeamMethod.ENERGY_BASED,
        stitching_method=StitchingMethod.AVERAGE)
    stitcher.process_folder("imgs/biennale")
    
    stitcher.stitch_all()
    
    cv.imshow("mosaic", stitcher.mosaic)
    cv.waitKey(0)
    cv.destroyAllWindows()
