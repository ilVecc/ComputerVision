from enum import Enum, auto

import numpy as np
from cv2 import cv2 as cv

from utils.algorithms import energy_based_seam_line, biggest_shared_region_bb, fast_color_blending
from utils.homography import fit_homography, distance_homography
from utils.line import fit_line2D, distance_line2D
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
        patch_mask_non_shared = np.bitwise_xor(patch_mask, patch_mask_shared)  # mask of the non-shared region w.r.t. patch
        
        if self == SeamMethod.SIMPLE:
            # TODO broken
            # the seam is a simple vertical line
            seamed_patch_mask = patch_mask
            seam_mask = cv.Scharr(patch_mask_shared.astype(np.uint8) * 255, ddepth=-1, dx=1, dy=0)
            seam_wrt_patch = np.vstack(np.where(seam_mask))
        
        # https://ieeexplore.ieee.org/document/5304214/
        elif self == SeamMethod.ENERGY_BASED:
            
            # trace the shared region bounding box and use it as input for the algorithm
            x, y, w, h = biggest_shared_region_bb(patch_mask_shared)
            # if x is None then this is the first image and thus cannot have overlapping regions
            if x is None:
                # this is the first image
                seamed_patch_mask, seam_wrt_patch = SeamMethod.SIMPLE.__call__(mosaic, mosaic_mask, patch, patch_mask, patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic)
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


class StitchingMethod(Enum):
    DIRECT = auto()
    AVERAGE = auto()
    WEIGHTED = auto()
    ALPHA_GRADIENT = auto()
    SUPERPIXEL_BASED = auto()
    SUPERPIXEL_BASED_ALT = auto()
    POISSON = auto()
    
    def __call__(self, mosaic, mosaic_mask, patch, patch_mask, seam_wrt_patch, patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic, t=0.5):
        
        # get content of mosaic (and mosaic mask) in the patch region
        ref__mosaic_in_patch = mosaic[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic, :]
        ref__mosaic_mask_in_patch = mosaic_mask[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic]

        # find the shared region in patch reference system
        patch_mask_shared = np.bitwise_and(ref__mosaic_mask_in_patch, patch_mask)  # mask of the shared region w.r.t. patch
        patch_mask_non_shared = np.bitwise_xor(patch_mask, patch_mask_shared)  # mask of the non-shared region w.r.t. patch
        
        # add by default new pixels, without touching the shared region
        ref__mosaic_in_patch[patch_mask_non_shared, :] = patch[patch_mask_non_shared, :]

        # stitch the warped patch in the shared region
        if self == StitchingMethod.DIRECT:
            # simply use the patch content
            weights = [0.0, 1.0]
            ref__mosaic_in_patch[patch_mask_shared] = weights[0] * ref__mosaic_in_patch[patch_mask_shared] + weights[1] * patch[patch_mask_shared]
        
        elif self == StitchingMethod.AVERAGE:
            weights = [0.5, 0.5]
            ref__mosaic_in_patch[patch_mask_shared] = weights[0] * ref__mosaic_in_patch[patch_mask_shared] + weights[1] * patch[patch_mask_shared]
        
        elif self == StitchingMethod.WEIGHTED:
            weights = [t, 1 - t]
            ref__mosaic_in_patch[patch_mask_shared] = weights[0] * ref__mosaic_in_patch[patch_mask_shared] + weights[1] * patch[patch_mask_shared]
        
        elif self == StitchingMethod.ALPHA_GRADIENT:

            # TODO broken
            weights = [0.0, 1.0]

            if seam_wrt_patch.shape[1] != 0:
                # fit line on the seam
                best_line, curr_iter, max_iter = None, 0, 10
                k, th = 1000, 3
                while best_line is None or curr_iter < max_iter:
                    # samples = 4  because it's the minimum required to estimate an homography
                    # providing more samples to RANSAC will most definitely result in problematic H outputs and glitches
                    best_line, best_inliers, _ = ransac(seam_wrt_patch, max_iter=k, thresh=th, samples=2, fit_fun=fit_line2D, dist_fun=distance_line2D)
                    curr_iter += 1
                    if best_line is not None:
                        break
                    print(f"Could not find a line. Parameters have been relaxed.")
                    k += 20
                    th += 0.1
                # calculate gradient orthogonal to the line
                m = -best_line[0]
                q = best_line[1]
                q = t / np.cos(np.arctan(m)) - q  # new line
                xv, yv = np.meshgrid(np.arange(patch.shape[0]), np.arange(patch.shape[1]), indexing='ij')
                gradient_mask = np.zeros_like(patch_mask)
                gradient_mask[m * xv + q - yv > 0] = 1.0
                # blur line to get smooth transition
                kernel_size = t * np.cos(np.arctan(-1/m))
                gradient_mask = cv.GaussianBlur(gradient_mask, (kernel_size, kernel_size), 0)
                patch = patch * gradient_mask
            
            ref__mosaic_in_patch[patch_mask_shared] = weights[0] * ref__mosaic_in_patch[patch_mask_shared] + weights[1] * patch[patch_mask_shared]
        
        # https://ieeexplore.ieee.org/document/9115682/  or  https://ieeexplore.ieee.org/document/8676030
        elif self == StitchingMethod.SUPERPIXEL_BASED_ALT or self == StitchingMethod.SUPERPIXEL_BASED:

            weights = [0.0, 1.0]

            # check first image
            if seam_wrt_patch.shape[1] != 0:
                use_distance = self == StitchingMethod.SUPERPIXEL_BASED
                seam_in_mosaic = mosaic[patch_y_range_wrt_mosaic.start + seam_wrt_patch[0, :], patch_x_range_wrt_mosaic.start + seam_wrt_patch[1, :]]
                seam_in_patch = patch[seam_wrt_patch[0, :], seam_wrt_patch[1, :]]
                
                D = seam_in_mosaic - seam_in_patch
                w, segments, n_segments = fast_color_blending(patch, patch_mask, seam_in_patch, seam_wrt_patch, use_distance=use_distance)
                
                # calculate the color change and apply it to each superpixel, modifying the patch
                T = w.T @ D
                patch = np.float64(patch)
                for i in range(n_segments):
                    patch[segments == (i + 1)] += T[i, :]
                patch = np.clip(patch, 0, 255)
                patch = np.uint8(patch)

            ref__mosaic_in_patch[patch_mask_shared] = weights[0] * ref__mosaic_in_patch[patch_mask_shared] + weights[1] * patch[patch_mask_shared]

        elif self == StitchingMethod.POISSON:
            # We need a little lateral thinking here: after having added the non-shared pixels, we Poisson-blend the patch onto mosaic (using seamed mask);
            # this way we blend the old pixels in the overlapping region (important) and the new pixels with the patch (useless, the content is already there).
            # This workaround effectively blends just the overlapping region and allows us to use the  seamlessClone()  function, which requires positioning
            # into an empty area of the mosaic, which would definitely ruin the result without these premises.
            ref__mosaic_in_patch[patch_mask_shared] = patch[patch_mask_shared]
            cx = (patch_x_range_wrt_mosaic.stop + patch_x_range_wrt_mosaic.start)/2
            cy = (patch_y_range_wrt_mosaic.stop + patch_y_range_wrt_mosaic.start)/2
            mosaic = cv.seamlessClone(patch, mosaic, np.uint8(patch_mask), (int(cx), int(cy)), cv.MIXED_CLONE)
        
        else:
            raise NotImplementedError(f"{self.__class__.__name__} {self} is not implemented")
        
        # lastly, because matrix changes are by reference, update the total mask adding the patch mask
        mosaic_mask[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic] = np.bitwise_or(ref__mosaic_mask_in_patch, patch_mask)

        return mosaic, mosaic_mask
