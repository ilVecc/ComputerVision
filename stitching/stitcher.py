from enum import Enum, auto
from pathlib import Path

from cv2 import cv2 as cv
import numpy as np

from utils.homography import fit_homography, distance_homography, apply_homogeneous
from utils.ransac import ransac


class HomographyMethod(Enum):
    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    MANUAL_IMPL = auto()
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
    
        elif self == HomographyMethod.CV_IMPL:
            # the cv version refines the final homography with the Levenberg-Marquardt method
            # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gafd3ef89257e27d5235f4467cbb1b6a63
            # https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
            src_pts = pairs[:, 0, 0:2].reshape(-1, 1, 2)
            dst_pts = pairs[:, 1, 0:2].reshape(-1, 1, 2)
            best_H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        
        return best_H
    

class SeamMethod(Enum):
    SIMPLE = auto()
    ENERGY_BASED = auto()


class CrossoverMethod(Enum):
    AVERAGE = auto()
    BLEND = auto()
    GAUSSIAN = auto()
    
    def __call__(self, t=0.5):
        if self == CrossoverMethod.AVERAGE:
            weights = [0.5, 0.5]
        elif self == CrossoverMethod.BLEND:
            weights = [t, 1 - t]
        else:
            # TODO implement gaussian crossover
            weights = [0, 0]
        return weights


class ImagePatch(object):
    
    _sift = cv.SIFT_create()
    
    def __init__(self, path, load=False):
        self.path = path
        # loading
        self.loaded = False
        self.img = None
        self.keypoints = None
        self.descriptors = None
        self._gray = None
        # warping
        self.warped = False
        self.H = None
        self.warped_bb = None  # [origin_x, origin_y, ending_x, ending_y], w.r.t. H reference frame
        self.warped_bb_origin = None  # [origin_x, origin_y], w.r.t. H reference frame
        self.warped_bb_ending = None  # [ending_x, ending_y], w.r.t. H reference frame
        self.warped_center = None  # [x, y], w.r.t. H reference frame
        
        if load:
            self.load_and_sift()
    
    def load_and_sift(self):
        if self.loaded:
            print(f"Image {self.path} has been already loaded")
            return
        
        try:
            self.img = cv.imread(self.path)
        except Exception as ex:
            print(f"Failed to load image {self.path} due to exception:")
            print(ex)

        self.loaded = True
        self._gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.keypoints, self.descriptors = self._sift.detectAndCompute(self._gray, None)

    def assign_warping_matrix(self, H):
        self.warped = True
        self.H = H

    def warp_bounding_box(self, warp_H):
        h, w, d = self.img.shape
        # trace border warping
        border = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        border_warp = apply_homogeneous(border.T, warp_H).T.astype(int)
        # trace center warping
        center = np.array([w / 2, h / 2])
        center_warp = apply_homogeneous(center.T, warp_H).T
    
        # compute warped bounding box extrema wrt "world"
        bb_world_min_xy = np.min(border_warp, axis=0)
        bb_world_max_xy = np.max(border_warp, axis=0)
        bb_world_xy = np.hstack([bb_world_min_xy, bb_world_max_xy])  # [x_min, y_min, x_max, y_max]
    
        return bb_world_xy, center_warp
    
    def warp_bounding_box_self(self):
        self.warped_bb, self.warped_center = self.warp_bounding_box(self.H)
        self.warped_bb_origin = self.warped_bb[0:2]
        self.warped_bb_ending = self.warped_bb[2:4]
    
    def __hash__(self):
        return self.path.__hash__()
    
    def __eq__(self, other):
        if not isinstance(other, ImagePatch):
            return False
        if other.path != self.path:
            return False
        return True


class ImageStitching(object):
    
    def __init__(self, folder: str = None,
                 homography_method=HomographyMethod.MANUAL_IMPL,
                 seam_method=SeamMethod.SIMPLE,
                 crossover_method=CrossoverMethod.AVERAGE, crossover_param=None,
                 decimation_factor=0.1):
        # parameters
        self.homography_method = homography_method
        self.seam_method = seam_method
        self.crossover_method = crossover_method
        self.crossover_param = crossover_param
        self.decimation_factor = decimation_factor
        
        self._images = []
        self._i = 0
        # SIFT-related
        self._train_kp = np.array([])
        self._train_desc = np.empty(shape=(0, 128), dtype=np.float32)
        # matcher-related
        # https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or empty dictionary
        self._flann_matcher = cv.FlannBasedMatcher(index_params, search_params)
        
        self.mosaic = None
        self.mosaic_mask = None
        
        if folder is not None:
            self.process_folder(folder)
        
    # https://ieeexplore.ieee.org/document/4604308
    def _find_matches(self, desc_new, desc_ref, k=2):
        # find the best  k  matches for the  desc_new  descriptors searching in between the ones in  desc_ref
        # basically this returns an array of  |desc_new|  tuples of  k  elements, which are the best  k  matches in  desc_ref
        return self._flann_matcher.knnMatch(desc_new, desc_ref, k)
    
    def _find_good_matches(self, desc_new, desc_ref):
        # get the best 2 matches as per Lowe's paper
        matches = self._find_matches(desc_new, desc_ref, k=2)
        # now we need only good matches, so we create a mask using the ratio test as per Lowe's paper
        # given the two best matches, we take the best one (m1) if   m1.distance < 0.7 * m2.distance
        return [m1 for m1, m2 in matches if m1.distance < 0.7 * m2.distance]
        
    def add_folder(self, folder):
        ret = 0
        folder = Path(folder)
        if folder.is_dir():
            # init images
            for img_path in folder.iterdir():
                ret += self.add_image(str(img_path))
        else:
            print(f"Provided path {folder} is not a directory")
        return ret
    
    def add_image(self, path):
        if not Path(path).is_file():
            print(f"Provided path {path} is not a file")
            return False
        
        img_patch = ImagePatch(path)
        self._images.append(img_patch)
        return True
    
    # FIXME i don't take into account pending images
    def process_folder(self, folder):
        self.add_folder(folder)
        while self._i < len(self._images):
            self.process_next()

    # FIXME i don't take into account pending images
    def process_image(self, path):
        if self.add_image(path):
            self.process_next()

    def process_next(self):
        ###
        #  1) get SIFT keypoints and descriptors
        ###
        image_patch = self._images[self._i]
        self._i += 1
        image_patch.load_and_sift()
        # cv.imshow("keypoints", cv.resize(cv.drawKeypoints(image_patch._gray, image_patch.keypoints, image_patch.img), (1366, 768))); cv.waitKey(0)
        
        ###
        #  2) estimate homography
        ###
        if self._i == 1:
            # this is the first image, no transform is needed
            best_H = np.eye(3)
        else:
            ###
            #  2.1) find matches for descriptors
            ###
            good_matches = self._find_good_matches(image_patch.descriptors, self._train_desc)
            
            ###
            #  2.2) filter valid matches
            ###
            # pairs is (n, m, 3), with  n  2D homogeneous points for each of the  m  sets of selected matches
            pairs = np.ones(shape=(len(good_matches), 2, 3))
            pairs[:, :, 0:2] = [(image_patch.keypoints[match.queryIdx].pt, self._train_kp[match.trainIdx].pt) for match in good_matches]
            
            ###
            #  2.3) actually estimate homography using lines from matches via RANSAC
            ###
            best_H = self.homography_method(pairs)
            if best_H is None:
                print(f"Something went very bad with the homography estimation... removing this patch from list")
                self._images.pop(self._i - 1)
                return
        
        # # TODO requires a new warping algorithm
        # # https://ieeexplore.ieee.org/document/6909812
        # theta = np.arctan2(-best_H[2, 1], -best_H[2, 0])
        # rot = np.array([[np.cos(theta), -np.sin(theta)],
        #                 [np.sin(theta),  np.cos(theta)]])
        # H_img_to_mosaic = best_H.copy()
        # H_img_to_mosaic[0:2, 0:2] = H_img_to_mosaic[0:2, 0:2] @ rot
        # H_img_to_mosaic[2, 0] = -np.sqrt(best_H[2, 1] ** 2 + best_H[2, 0] ** 2)
        # H_img_to_mosaic[2, 1] = 0
        
        H_img_to_mosaic = best_H
        image_patch.assign_warping_matrix(H_img_to_mosaic)
        
        ###
        #  3) add keypoints and descriptors to training set
        ###
        self._train_kp = np.append(self._train_kp, image_patch.keypoints)
        self._train_desc = np.vstack([self._train_desc, image_patch.descriptors])
        
        train_size = len(self._train_kp)
        train_size_new = int(self.decimation_factor * train_size)
        filter_idxs = np.random.choice(np.arange(train_size), size=train_size_new, replace=False)
        
        self._train_kp = self._train_kp[filter_idxs]
        self._train_desc = self._train_desc[filter_idxs]
        
        ###
        #  4) precompute patch bounding box (not yet the mask though)
        ###
        image_patch.warp_bounding_box_self()
    
    def stitch_all(self):
        
        ###
        #  apply homography and offset via cv.warpPerspective
        #
        #     Viewbox precomputation
        #     ----------------------
        #     Basically we precompute the warped bounding box of each patch, so to have the size of the image that completely contains
        #     the warped patch; this will be our patch's viewbox. Then, we calculate an offset matrix, so to put the warped patch inside the viewbox, thus
        #     minimizing the size of the warped patch in memory (kinda). Then, we expand all the viewboxes to find the viewbox of the whole mosaic, and use the
        #     previously computed offsets to put each warped patch in its correct location inside the mosaic viewbox.
        ###
        
        # bounding box of the whole mosaic wrt world
        bb_mosaic_min = np.min(np.vstack([image_patch.warped_bb_origin for image_patch in self._images]), axis=0)  # [min_x, min_y] wrt world frame
        bb_mosaic_max = np.max(np.vstack([image_patch.warped_bb_ending for image_patch in self._images]), axis=0)  # [max_x, max_y] wrt world frame
        
        # initialize empty image
        mosaic_size = (bb_mosaic_max - bb_mosaic_min + 1).astype(int)
        self.mosaic = np.zeros(shape=(mosaic_size[1], mosaic_size[0], 3), dtype=int)
        self.mosaic_mask = np.zeros(shape=(mosaic_size[1], mosaic_size[0])).astype(bool)
        
        for i, other_image_patch in enumerate(self._images):
            origin = other_image_patch.warped_bb_origin
            ending = other_image_patch.warped_bb_ending
            
            # compute warped image size wrt world
            new_size = (ending - origin + 1).astype(int)
            
            # create and apply offset transform (so we minimize the warped image size, keeping it completely inside the viewbox)
            offset_H = np.array([[1, 0, -origin[0]],
                                 [0, 1, -origin[1]],
                                 [0, 0, 1]])
            H = offset_H @ other_image_patch.H
            
            # image color info
            patch = cv.warpPerspective(other_image_patch.img, H, new_size, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
            
            # image shape info
            img_mask = np.ones(shape=other_image_patch.img.shape[0:2], dtype=np.uint8) * 255
            patch_mask = cv.warpPerspective(img_mask, H, new_size, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0) == 255
            
            ####
            # stitch patch onto mosaic
            ####
            
            # calculate patch range wrt mosaic
            patch_origin_wrt_mosaic = (origin - bb_mosaic_min).astype(int)
            patch_ending_wrt_mosaic = (ending - bb_mosaic_min + 1).astype(int)
            patch_y_range_wrt_mosaic = slice(patch_origin_wrt_mosaic[1], patch_ending_wrt_mosaic[1])
            patch_x_range_wrt_mosaic = slice(patch_origin_wrt_mosaic[0], patch_ending_wrt_mosaic[0])
            
            # find the shared region in patch reference system and in mosaic reference system
            ref__mosaic_mask_of_patch = self.mosaic_mask[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic]
            patch_mask_shared = np.bitwise_and(ref__mosaic_mask_of_patch, patch_mask)
            mosaic_mask_shared = np.zeros_like(self.mosaic_mask)
            mosaic_mask_shared[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic] = patch_mask_shared
            # stitch the warped patch according to its mask (this overwrites shared regions)
            ref__mosaic_of_patch = self.mosaic[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic, :]
            ref__mosaic_of_patch[patch_mask != 0, :] = patch[patch_mask != 0, :]
            
            ####
            #   deal with overlapping region
            ####
            
            
            
            # find optimal stitching seam
            # https://ieeexplore.ieee.org/document/5304214/
            theta = self.mosaic[mosaic_mask_shared != 0, :] - patch[patch_mask_shared != 0, :]  # this is not (x,y) though!
            
            
            
            # use shared mask to blend the colors in the shared region
            weights = self.crossover_method(self.crossover_param)
            self.mosaic[mosaic_mask_shared != 0, :] \
                = weights[0] * self.mosaic[mosaic_mask_shared != 0, :] \
                + weights[1] * patch[patch_mask_shared != 0, :]
            
            # lastly, because matrix changes are by reference, update the total mask adding the new patch mask
            self.mosaic_mask[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic] = np.bitwise_or(ref__mosaic_mask_of_patch, patch_mask)
        
        # final image still contains float values
        self.mosaic = self.mosaic.astype(np.uint8)
    
    def balance_warpings(self):
        # calculate image center and recalculate every H so to balance the image
        pass

