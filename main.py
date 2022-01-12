from stitching import *

# TODO remove file info.md

if __name__ == '__main__':
    stitcher = ImageStitching(
        warping_method=WarpingMethod.MANUAL_IMPL,
        seam_method=SeamMethod.ENERGY_BASED,
        exposure_compensation_method=ExposureCompensationMethod.GAIN,
        stitching_method=StitchingMethod.MULTI_CHANNEL_BLENDING, stitching_param=100,
        trimming_method=TrimmingMethod.NONE,
        decimation_factor=0,
        debug=True
    )
    # stitcher.process_folder("imgs/roofs")
    # stitcher.process_folder("imgs/river")
    # stitcher.process_folder("imgs/library")
    # stitcher.process_folder("imgs/biennale/low_res")
    stitcher.process_folder("imgs/colosseum/low_res")
    # stitcher.process_folder("imgs/stadium/low_res_1")

    stitcher.balance_warpings(use_translation=True)

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
    stitcher.save("mosaic.png")
