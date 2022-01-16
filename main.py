from stitching import *

if __name__ == '__main__':
    stitcher = ImageStitching(
        warping_method=WarpingMethod.CV_IMPL,
        seam_method=SeamMethod.ENERGY_MAP_BASED,
        exposure_compensation_method=ExposureCompensationMethod.GAIN,
        blending_method=BlendingMethod.POISSON,
        trimming_method=TrimmingMethod.NONE,
        decimation_factor=0,
        debug=True
    )
    # LEGACY
    # stitcher.add_folder("imgs/roofs")
    # stitcher.add_folder("imgs/river")
    # stitcher.add_folder("imgs/library")
    
    imgs_sets = [
        "arc",
        "biennale",  # BROKEN
        "certosa",
        "colosseum",
        "forum",
        "saint_peter",
        "saint_peter_church",
        "spagna_square",
        "passirio",  # WORKS in CV
        "venice"
    ]
    imgs_set = imgs_sets[0]
    stitcher.add_folder(f"imgs/{imgs_set}/low_res")
    
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
    stitcher.save(f"imgs_results/mosaic_{imgs_set}.png")
