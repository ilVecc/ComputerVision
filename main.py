from stitching.stitcher import ImageStitching, CrossoverMethod, SeamMethod
import cv2.cv2 as cv

if __name__ == '__main__':
    stitcher = ImageStitching(seam_method=SeamMethod.SIMPLE, crossover_method=CrossoverMethod.AVERAGE)
    stitcher.process_folder("imgs/roofs")
    
    stitcher.stitch_all()
    
    cv.imshow("mosaic", stitcher.mosaic)
    cv.waitKey(0)
    cv.destroyAllWindows()
