from stitching.stitcher import ImageStitching, StitchingMethod, SeamMethod, HomographyMethod
import cv2.cv2 as cv

# TODO remove file info.md

if __name__ == '__main__':
    stitcher = ImageStitching(
        homography_method=HomographyMethod.CV_IMPL,
        seam_method=SeamMethod.ENERGY_BASED,
        stitching_method=StitchingMethod.AVERAGE,
        decimation_factor=0.5
    )
    stitcher.process_folder("imgs/lake")
    stitcher.stitch_all()
    
    cv.imshow("mosaic", cv.resize(stitcher.mosaic, (1366, 768)))
    cv.waitKey(0)
    cv.destroyAllWindows()
