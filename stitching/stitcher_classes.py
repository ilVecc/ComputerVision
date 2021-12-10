from enum import Enum, auto

import numpy as np
from cv2 import cv2 as cv

from utils.algorithms import energy_based_seam_line, biggest_shared_region_bb, fast_color_blending
from utils.homography import fit_homography, distance_homography
from utils.ransac import ransac


class HomographyMethod(Enum):
    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    MANUAL_IMPL = auto()
    SPHP_IMPL = auto()
    CV_IMPL = auto()
    
    def __call__(self, pairs):
        if self == HomographyMethod.MANUAL_IMPL:
            best_H, curr_iter, max_iter = None, 0, 10
            k, th = 2000, 3
            while best_H is None or curr_iter < max_iter:
                # samples = 4  because it's the minimum required to estimate an homography
                # providing more samples to RANSAC will most definitely result in problematic H outputs and glitches
                best_H, best_inliers, _ = ransac(pairs, max_iter=k, thresh=th, samples=4, fit_fun=fit_homography, dist_fun=distance_homography)
                curr_iter += 1
                if best_H is not None:
                    break
                print(f"Could not find an homography. Parameters have been relaxed.")
                k += 100
                th += 0.1
        
        # https://ieeexplore.ieee.org/document/6909812
        elif self == HomographyMethod.SPHP_IMPL:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
            # # TODO requires a new warping algorithm
            # theta = np.arctan2(-best_H[2, 1], -best_H[2, 0])
            # rot = np.array([[np.cos(theta), -np.sin(theta)],
            #                 [np.sin(theta),  np.cos(theta)]])
            # H_img_to_mosaic = best_H.copy()
            # H_img_to_mosaic[0:2, 0:2] = H_img_to_mosaic[0:2, 0:2] @ rot
            # H_img_to_mosaic[2, 0] = -np.sqrt(best_H[2, 1] ** 2 + best_H[2, 0] ** 2)
            # H_img_to_mosaic[2, 1] = 0
            # best_H = H_img_to_mosaic
        
        elif self == HomographyMethod.CV_IMPL:
            # the cv version refines the final homography with the Levenberg-Marquardt method
            # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gafd3ef89257e27d5235f4467cbb1b6a63
            # https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
            src_pts = pairs[:, 0, 0:2].reshape(-1, 1, 2)
            dst_pts = pairs[:, 1, 0:2].reshape(-1, 1, 2)
            best_H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        
        else:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
        
        return best_H
    
    def warp(self, img, H, new_size):
        if self == HomographyMethod.MANUAL_IMPL:
            patch = cv.warpPerspective(img, H, new_size, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
        
        elif self == HomographyMethod.SPHP_IMPL:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
        
        elif self == HomographyMethod.CV_IMPL:
            patch = cv.warpPerspective(img, H, new_size, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
        
        else:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
        
        return patch


class SeamMethod(Enum):
    SIMPLE = auto()
    ENERGY_BASED = auto()
    
    def __call__(self, mosaic, mosaic_mask, patch, patch_mask, patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic):
        
        # find the shared region in patch reference system
        ref__mosaic_mask_in_patch = mosaic_mask[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic]
        patch_mask_shared = np.bitwise_and(ref__mosaic_mask_in_patch, patch_mask)  # mask of the shared region w.r.t. patch
        
        if self == SeamMethod.SIMPLE:
            # stitch the warped patch according to its mask (this overwrites shared regions)
            mask = patch_mask
            seam_mask = cv.Scharr(patch_mask_shared.astype(np.uint8) * 255, ddepth=-1, dx=1, dy=0)
            seam = np.vstack(np.where(seam_mask))
        
        # https://ieeexplore.ieee.org/document/5304214/
        elif self == SeamMethod.ENERGY_BASED:
            
            # trace the shared region bounding box and use it as input for the algorithm
            x, y, w, h = biggest_shared_region_bb(patch_mask_shared)
            # if x is None then this is the first image and thus cannot have overlapping regions
            if x is None:
                # this is the first image
                mask, seam = SeamMethod.SIMPLE.__call__(mosaic, mosaic_mask, patch, patch_mask, patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic)
            else:
                mask_shared_in_bb = patch_mask_shared[y:y + h, x:x + w].copy()
                ref__patch_in_bb = patch[y:y + h, x:x + w]
                ref__mosaic_in_bb = mosaic[patch_y_range_wrt_mosaic.start + y:patch_y_range_wrt_mosaic.start + y + h,
                                           patch_x_range_wrt_mosaic.start + x:patch_x_range_wrt_mosaic.start + x + w]
                ####
                #   find optimal stitching seam
                ####
                theta = ref__mosaic_in_bb - ref__patch_in_bb
                # line.shape == (2, theta.shape[0]),  pixels are [y; x]
                seam_line = energy_based_seam_line(img=ref__mosaic_in_bb, theta=theta, theta_mask=mask_shared_in_bb)
                
                # # draw the seam line
                # ref__mosaic_in_bb[seam_line[0, :], seam_line[1, :]] = [255, 0, 0]
                # ref__patch_in_bb[seam_line[0, :], seam_line[1, :]] = [255, 0, 0]
                # import matplotlib.pyplot as pt
                # pt.imshow(ref__mosaic_in_bb), pt.show()
                # pt.imshow(ref__patch_in_bb), pt.show()
                
                ####
                #   find best half to stitch
                ####
                
                # fill the right half of the cut bb
                xv = np.repeat(np.arange(theta.shape[1])[np.newaxis, :], repeats=theta.shape[0], axis=0)
                y_mask = np.repeat(seam_line[1][:, np.newaxis], repeats=theta.shape[1], axis=1)
                mask_in_bb_right = xv >= y_mask
                
                # get the mask of all the new pixels added in the image
                patch_mask_new_pixels = np.bitwise_not(patch_mask.copy())
                patch_mask_new_pixels = np.bitwise_or(patch_mask_new_pixels, ref__mosaic_mask_in_patch)  # TODO could be optimized
                patch_mask_new_pixels = np.bitwise_not(patch_mask_new_pixels)
                
                # test right: if we get one connected component, then we have found the final mask, otherwise simply negate it
                #             we also remove the upper and lower bands
                seamed_patch_mask = patch_mask_new_pixels.copy()
                seamed_patch_mask[y:y + h, x:x + w] = mask_in_bb_right
                seamed_patch_mask[:y, x:x + seam_line[1, 0]] = False
                seamed_patch_mask[y + h:, x:x + seam_line[1, -1]] = False
                contours, _ = cv.findContours(seamed_patch_mask.astype(np.uint8) * 255, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 1:
                    # test left
                    seamed_patch_mask = patch_mask_new_pixels.copy()
                    seamed_patch_mask[y:y + h, x:x + w] = np.bitwise_not(mask_in_bb_right)
                    seamed_patch_mask[:y, x + seam_line[1, 0] - 1:] = False
                    seamed_patch_mask[y + h:, x + seam_line[1, -1] - 1:] = False
                
                # final re-masking using the patch mask
                mask = np.bitwise_and(seamed_patch_mask, patch_mask)
                seam = seam_line + np.array([[y, x]]).T
        
        else:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
        
        return mask, seam  # pixels are [y; x]


class StitchingMethod(Enum):
    NONE = auto()
    AVERAGE = auto()
    WEIGHTED = auto()
    SUPERPIXEL = auto()
    
    def __call__(self, mosaic, patch, patch_mask, seam_wrt_patch, patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic, t=0.5):
        
        if self == StitchingMethod.NONE:
            weights = [0.0, 1.0]
        
        elif self == StitchingMethod.AVERAGE:
            weights = [0.5, 0.5]
        
        elif self == StitchingMethod.WEIGHTED:
            weights = [t, 1 - t]
        
        # https://ieeexplore.ieee.org/document/8676030
        elif self == StitchingMethod.SUPERPIXEL:
            # check first image
            if seam_wrt_patch.shape[1] != 0:
                seam_in_mosaic = mosaic[patch_y_range_wrt_mosaic.start + seam_wrt_patch[0, :], patch_x_range_wrt_mosaic.start + seam_wrt_patch[1, :]]
                seam_in_patch = patch[seam_wrt_patch[0, :], seam_wrt_patch[1, :]]
                
                D = seam_in_mosaic - seam_in_patch
                w, segments, n_segments = fast_color_blending(patch, patch_mask, seam_in_patch, seam_wrt_patch)
                
                # calculate the color change and apply it to each superpixel, modifying the patch
                T = (w.T @ D).astype(np.uint8)
                for i in range(n_segments):
                    patch[segments == (i + 1)] += T[i, :]
            
            # finally, simply weight just the patch in the shared region
            weights = [0.0, 1.0]
        
        else:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
        
        return weights, patch
