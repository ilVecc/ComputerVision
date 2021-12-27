from stitching.stitcher import ImageStitching, StitchingMethod, SeamMethod, HomographyMethod
import cv2.cv2 as cv

# TODO remove file info.md

if __name__ == '__main__':
    stitcher = ImageStitching(
        homography_method=HomographyMethod.MANUAL_IMPL,
        seam_method=SeamMethod.ENERGY_BASED,
        stitching_method=StitchingMethod.AVERAGE,
        decimation_factor=0.75,
        make_rectangle=False,
        trim_borders=False
    )
    # stitcher.process_folder("imgs/biennale/low_res")
    # stitcher.process_folder("imgs/roofs")
    stitcher.process_folder("imgs/library")
    stitcher.balance_warpings(use_central=False)
    # TODO
    #  Order the images using  H  so to avoid the case of a "double seam" necessity.
    #  e.g. the mosaic could be
    #                       ##  $$
    #                       %%**++
    #        and adding  €€  in
    #                       ##€€$$
    #                       %%**++
    #        would need one seam line for  ##  and one for  $$
    #  A simple order of the  t  component in  H  using left-to-right, top-to-bottom order would be enough
    stitcher.stitch_all()
    
    cv.imshow("mosaic", cv.resize(stitcher.mosaic, (1366, 768)))
    cv.waitKey(0)
    cv.destroyAllWindows()
