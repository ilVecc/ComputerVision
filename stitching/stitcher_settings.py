from enum import Enum, auto

import numpy as np
from cv2 import cv2 as cv

from utils.algorithms import energy_based_seam_line, biggest_bb_in_mask, fast_color_blending, biggest_rectangle_in_mask, multi_band_blending, GainCompensator, \
    multi_band_blending_masked
from utils.homography import fit_homography, distance_homography, test_degenerate_samples
from utils.line import fit_line2D, distance_line2D
from utils.ransac import ransac

# useful pipeline
# https://docs.opencv.org/4.x/d1/d46/group__stitching.html
# https://github.com/opencv/opencv/blob/17234f82d025e3bbfbf611089637e5aa2038e7b8/samples/python/stitching_detailed.py
# http://matthewalunbrown.com/papers/ijcv2007.pdf


class WarpingMethod(Enum):
    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    MANUAL_IMPL = auto()
    SPHP_IMPL = auto()
    CV_IMPL = auto()
    
    def __call__(self, pairs):
        if self == WarpingMethod.MANUAL_IMPL:
            # samples = 4  because it's the minimum required to estimate an homography
            # providing more samples to RANSAC will most definitely result in problematic H outputs and glitches
            best_H, _, _ = ransac(
                pairs,
                max_iter=2000, thresh=3, samples=4,
                fit_fun=fit_homography, dist_fun=distance_homography, test_samples=test_degenerate_samples
            )
        
        # https://ieeexplore.ieee.org/document/6909812
        elif self == WarpingMethod.SPHP_IMPL:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
            # # TODO requires a new warping function
            # theta = np.arctan2(-best_H[2, 1], -best_H[2, 0])
            # rot = np.array([[np.cos(theta), -np.sin(theta)],
            #                 [np.sin(theta),  np.cos(theta)]])
            # H_img_to_mosaic = best_H.copy()
            # H_img_to_mosaic[0:2, 0:2] = H_img_to_mosaic[0:2, 0:2] @ rot
            # H_img_to_mosaic[2, 0] = -np.sqrt(best_H[2, 1] ** 2 + best_H[2, 0] ** 2)
            # H_img_to_mosaic[2, 1] = 0
            # best_H = H_img_to_mosaic
        
        elif self == WarpingMethod.CV_IMPL:
            # the cv version refines the final homography with the Levenberg-Marquardt method
            # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gafd3ef89257e27d5235f4467cbb1b6a63
            # https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
            src_pts = pairs[:, 0, 0:2].reshape(-1, 1, 2)
            dst_pts = pairs[:, 1, 0:2].reshape(-1, 1, 2)
            best_H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        
        else:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
        
        return best_H
    
    def warp(self, img, H, new_size, is_mask=False):
        
        mode = cv.BORDER_CONSTANT if is_mask else cv.BORDER_REFLECT
        
        if self == WarpingMethod.MANUAL_IMPL:
            patch = cv.warpPerspective(img, H, new_size, flags=cv.INTER_LINEAR, borderMode=mode, borderValue=0)
        
        elif self == WarpingMethod.SPHP_IMPL:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
        
        elif self == WarpingMethod.CV_IMPL:
            patch = cv.warpPerspective(img, H, new_size, flags=cv.INTER_LINEAR, borderMode=mode, borderValue=0)
        
        else:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
        
        return patch


class SeamMethod(Enum):
    DIRECT = auto()
    ENERGY_MAP_BASED = auto()
    
    def __call__(self, mosaic, mosaic_mask, patch, patch_mask, patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic):
        
        # find the shared region in patch reference system
        ref__mosaic_mask_in_patch = mosaic_mask[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic]
        patch_mask_shared = np.bitwise_and(ref__mosaic_mask_in_patch, patch_mask)  # mask of the shared region w.r.t. patch
        patch_mask_non_shared = np.bitwise_xor(patch_mask, patch_mask_shared)  # mask of the non-shared region w.r.t. patch
        
        if self == SeamMethod.DIRECT:
            # TODO broken
            # the seam is a simple vertical line
            seamed_patch_mask = patch_mask
            seam_mask = cv.Scharr(patch_mask_shared.astype(np.uint8) * 255, ddepth=-1, dx=1, dy=0)
            seam_wrt_patch = np.vstack(np.where(seam_mask))
        
        # https://ieeexplore.ieee.org/document/5304214/
        elif self == SeamMethod.ENERGY_MAP_BASED:
            
            # trace the shared region bounding box and use it as input for the algorithm
            x, y, w, h = biggest_bb_in_mask(patch_mask_shared)
            # if x is None then this is the first image and thus cannot have overlapping regions
            if x is None:
                # this is the first image
                seamed_patch_mask, seam_wrt_patch = SeamMethod.DIRECT.__call__(mosaic, mosaic_mask, patch, patch_mask, patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic)
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
                # import matplotlib.pyplot as pt; pt.imshow(ref__mosaic_in_bb), pt.show(); pt.imshow(ref__patch_in_bb), pt.show()
                
                ####
                #   find best half to stitch
                ####
                
                # fill the right half of the shared area
                xv = np.repeat(np.arange(w)[np.newaxis, :], repeats=theta.shape[0], axis=0)
                y_mask = np.repeat(seam_line[1][:, np.newaxis] - seam_line[1].min(), repeats=w, axis=1)
                seam_mask_in_bb = xv >= y_mask
                
                # test side: if the shared region is shifted on the left, then use the mask (right-filling), otherwise use its inverse (left-filling)
                is_right = x + w / 2 < patch_mask.shape[1] / 2
                seam_mask_in_bb = seam_mask_in_bb if is_right else np.bitwise_not(seam_mask_in_bb)
                
                # paste the seam mask onto the non-shared mask and re-mask with the original patch mask for safety
                seamed_patch_mask = patch_mask_non_shared.copy()
                ref__seamed_patch_mask_in_bb = seamed_patch_mask[y:y + h, x:x + w]
                seamed_patch_mask[y:y + h, x:x + w] = np.bitwise_or(ref__seamed_patch_mask_in_bb, seam_mask_in_bb)
                seamed_patch_mask = np.bitwise_and(seamed_patch_mask, patch_mask)  # just to be sure
                
                # add offset to the seam, in order to reference it w.r.t. patch
                seam_wrt_patch = seam_line + np.array([[y, x]]).T
        
        else:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
        
        return seamed_patch_mask, seam_wrt_patch  # pixels are [y; x]


class ExposureCompensationMethod(Enum):
    NO = auto()
    GAIN = auto()
    
    def __call__(self, patches):
        
        if self == ExposureCompensationMethod.NO:
            gains = np.ones(shape=(len(patches,)))
            
        elif self == ExposureCompensationMethod.GAIN:
            gc = GainCompensator()
            origins = [p.warped_bb_origin for p in patches]
            images = [p.warped_img for p in patches]
            masks = [p.warped_mask for p in patches]
            gains = gc.feed(origins, images, masks)
            
        else:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
        
        return gains


class BlendingMethod(Enum):
    DIRECT = auto()
    AVERAGE = auto()
    WEIGHTED = auto()
    ALPHA_GRADIENT = auto()
    SUPERPIXEL_BASED = auto()
    SUPERPIXEL_BASED_ALT = auto()
    POISSON = auto()
    MULTI_BAND_BLENDING = auto()
    
    def __call__(self, mosaic, mosaic_mask, patch, patch_mask, seam_wrt_patch, patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic, param=0.5):
        
        # get content of mosaic (and mosaic mask) in the patch region
        ref__mosaic_where_patch = mosaic[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic, :]
        ref__mosaic_mask_where_patch = mosaic_mask[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic]

        # find the shared region in patch reference system
        mask_shared = np.bitwise_and(ref__mosaic_mask_where_patch, patch_mask)  # mask of the shared region w.r.t. patch
        mask_non_shared = np.bitwise_xor(patch_mask, mask_shared)  # mask of the non-shared region w.r.t. patch

        # add new pixels without touching the shared region
        ref__mosaic_where_patch[mask_non_shared, :] = patch[mask_non_shared, :]

        # stitch the warped patch in the shared region
        if self == BlendingMethod.DIRECT:
            # simply use the patch content
            weights = [0.0, 1.0]
            ref__mosaic_where_patch[mask_shared] = weights[0] * ref__mosaic_where_patch[mask_shared] + weights[1] * patch[mask_shared]
        
        elif self == BlendingMethod.AVERAGE:
            weights = [0.5, 0.5]
            ref__mosaic_where_patch[mask_shared] = weights[0] * ref__mosaic_where_patch[mask_shared] + weights[1] * patch[mask_shared]
        
        elif self == BlendingMethod.WEIGHTED:
            weights = [param, 1 - param]
            ref__mosaic_where_patch[mask_shared] = weights[0] * ref__mosaic_where_patch[mask_shared] + weights[1] * patch[mask_shared]
        
        elif self == BlendingMethod.ALPHA_GRADIENT:

            kernel_size = int(2*np.floor(param/2)+1)
            gradient_mask = np.ones_like(patch_mask)

            if seam_wrt_patch.shape[1] != 0:
                # fit line on the seam
                # samples = 2  because it's the minimum required to estimate a line
                best_line, best_inliers, _ = ransac(seam_wrt_patch, max_iter=1000, thresh=3, samples=2, fit_fun=fit_line2D, dist_fun=distance_line2D)
                
                # calculate gradient orthogonal to the line
                m = -best_line[1]
                q = best_line[0]
                # q = kernel_size / np.cos(np.arctan(m)) - q  # new line
                # cut mask along the line
                yv, xv = np.meshgrid(np.arange(patch.shape[0]), np.arange(patch.shape[1]), indexing='ij')
                gradient_mask[-yv <= m * xv + q] = 0.0

            # blur line to get smooth transition
            gradient_mask = cv.GaussianBlur(np.uint8(gradient_mask) * 255, (kernel_size, kernel_size), 0)
            gradient_mask = gradient_mask.astype(np.float) / 255

            ref__mosaic_where_patch[mask_shared] \
                = (1.0 - gradient_mask[mask_shared, np.newaxis]) * ref__mosaic_where_patch[mask_shared] \
                  + gradient_mask[mask_shared, np.newaxis] * patch[mask_shared]
        
        # https://ieeexplore.ieee.org/document/9115682/  or  https://ieeexplore.ieee.org/document/8676030
        elif self == BlendingMethod.SUPERPIXEL_BASED_ALT or self == BlendingMethod.SUPERPIXEL_BASED:

            weights = [0.0, 1.0]

            # check first image
            if seam_wrt_patch.shape[1] != 0:
                use_distance = self == BlendingMethod.SUPERPIXEL_BASED
                seam_in_mosaic = mosaic[patch_y_range_wrt_mosaic.start + seam_wrt_patch[0, :], patch_x_range_wrt_mosaic.start + seam_wrt_patch[1, :]]
                seam_in_patch = patch[seam_wrt_patch[0, :], seam_wrt_patch[1, :]]
                
                D = seam_in_mosaic - seam_in_patch
                w, segments, n_segments = fast_color_blending(patch, patch_mask, seam_in_patch, seam_wrt_patch, use_distance)
                
                # calculate the color change and apply it to each superpixel, modifying the patch
                T = w.T @ D
                patch = np.float64(patch)
                for i in range(n_segments):
                    patch[segments == (i + 1)] += T[i, :]
                patch = np.clip(patch, 0, 255)
                patch = np.uint8(patch)

            ref__mosaic_where_patch[mask_shared] = weights[0] * ref__mosaic_where_patch[mask_shared] + weights[1] * patch[mask_shared]

        elif self == BlendingMethod.POISSON:
            # We need a little lateral thinking here: after having added the non-shared pixels, we Poisson-blend the patch onto mosaic (using seamed mask);
            # this way we blend the old pixels in the overlapping region (important) and the new pixels with the patch (useless, the content is already there).
            # This workaround effectively blends just the overlapping region and allows us to use the  seamlessClone()  function, which requires positioning
            # into an empty area of the mosaic, which would definitely ruin the result without these premises.
            
            # ref__mosaic_where_patch[mask_shared] = patch[mask_shared]
            
            # find center of mask
            patch_mask_ = np.uint8(patch_mask) * 255
            # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            # patch_mask_ = cv.erode(patch_mask_, kernel)
            contours, _ = cv.findContours(patch_mask_, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contour = np.vstack(contours)
            (x, y, w, h) = cv.boundingRect(contour)
            cx = w / 2 + x + patch_x_range_wrt_mosaic.start
            cy = h / 2 + y + patch_y_range_wrt_mosaic.start
            
            mosaic = cv.seamlessClone(patch, mosaic, patch_mask_, (int(cx), int(cy)), cv.MIXED_CLONE)
        
        elif self == BlendingMethod.MULTI_BAND_BLENDING:
            
            # TODO this is not the right solution, but it's good enough
            contours, _ = cv.findContours(np.uint8(mask_shared), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                # first image
                ref__mosaic_where_patch[patch_mask, :] = patch[patch_mask]

            for cont in contours:
                x, y, w, h = cv.boundingRect(cont)
                x_range = slice(x, x + w)
                y_range = slice(y, y + h)
                patch_where_shared = patch[y_range, x_range].copy()
                patch_mask_where_shared = np.repeat(patch_mask[y_range, x_range][..., np.newaxis], repeats=3, axis=2) * 1
                mosaic_where_shared = ref__mosaic_where_patch[y_range, x_range].copy()

                blending = multi_band_blending(patch_where_shared, mosaic_where_shared, patch_mask_where_shared)
                ref__mosaic_where_patch[y_range, x_range, :] = blending
            
            # # TODO this doesn't work
            # # create a patch-only mosaic and mask to be blended
            # patch_as_mosaic = np.zeros_like(mosaic)
            # patch_as_mosaic[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic, :][patch_mask] = patch[patch_mask]
            # patch_mask_as_mosaic = np.zeros_like(mosaic_mask, dtype=bool)
            # patch_mask_as_mosaic[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic] = patch_mask
            #
            # # masks must be separated
            # ref__mosaic_mask_where_patch[...] = np.bitwise_xor(ref__mosaic_mask_where_patch, mask_shared)
            #
            # # make 3-channels masks
            # mask_patch_ = np.repeat(patch_mask_as_mosaic[..., np.newaxis], repeats=3, axis=2)
            # mask_mosaic_ = np.repeat(mosaic_mask[..., np.newaxis], repeats=3, axis=2)
            #
            # blending = multi_band_blending_masked(patch_as_mosaic, mosaic, mask_patch_, mask_mosaic_)
            # ref__mosaic_where_patch[patch_mask, :] = blending[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic][patch_mask, :]
        
        else:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")

        # finally, update the total mask adding the patch mask
        mosaic_mask[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic] = np.bitwise_or(ref__mosaic_mask_where_patch, patch_mask)

        return mosaic, mosaic_mask
        

class TrimmingMethod(Enum):
    NONE = auto()
    TRIM = auto()
    TRIM_KEEP_BORDERS = auto()
    
    def __call__(self, image, mask):
        
        if self == TrimmingMethod.NONE:
            return image, mask
        
        # find the trimming area
        rect = biggest_rectangle_in_mask(mask)
        range_y = slice(rect[0], rect[2] + 1)
        range_x = slice(rect[1], rect[3] + 1)
        
        if self == TrimmingMethod.TRIM_KEEP_BORDERS:
            trimmed_mask = np.zeros_like(mask).astype(dtype=bool)
            trimmed_mask[range_y, range_x] = True
            trimmed_image = image.copy()
            trimmed_image[np.bitwise_not(trimmed_mask), :] = [0, 0, 0]
            
        elif self == TrimmingMethod.TRIM:
            trimmed_image = image[range_y, range_x, :]
            trimmed_mask = mask[range_y, range_x]
            
        else:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")

        return trimmed_image, trimmed_mask
