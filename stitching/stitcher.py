from pathlib import Path

from cv2 import cv2 as cv
import numpy as np

from stitching.stitcher_classes import HomographyMethod, SeamMethod, StitchingMethod
from utils.homography import apply_homogeneous


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
        self.offset = None
        self.H_with_offset = None
        self.warped_bb = None  # [origin_x, origin_y, ending_x, ending_y], w.r.t. H reference frame
        self.warped_bb_origin = None  # [origin_x, origin_y], w.r.t. H reference frame
        self.warped_bb_ending = None  # [ending_x, ending_y], w.r.t. H reference frame
        self.warped_bb_size = None  # size of the bb after warping
        self.warped_bb_center = None  # [x, y], w.r.t. H reference frame
        
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
        self.warped_bb, self.warped_bb_center = self.warp_bounding_box(self.H)
        self.warped_bb_origin = self.warped_bb[0:2]
        self.warped_bb_ending = self.warped_bb[2:4]

        # compute warped image size wrt world
        self.warped_bb_size = (self.warped_bb_ending - self.warped_bb_origin + 1).astype(int)

        # create and apply offset transform (so we minimize the warped image size, keeping it completely inside the viewbox)
        self.offset = np.array([[1, 0, -self.warped_bb_origin[0]],
                                [0, 1, -self.warped_bb_origin[1]],
                                [0, 0, 1]])
        self.H_with_offset = self.offset @ self.H
    
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
                 stitching_method=StitchingMethod.AVERAGE, crossover_param=None,
                 decimation_factor=0.1):
        # parameters
        self.homography_method = homography_method
        self.seam_method = seam_method
        self.stitching_method = stitching_method
        self.crossover_param = crossover_param
        self.decimation_factor = np.clip(decimation_factor, 0, 1)
        
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
    
    # FIXME we don't take into account pending images
    def process_folder(self, folder):
        self.add_folder(folder)
        while self._i < len(self._images):
            self.process_next()

    # FIXME we don't take into account pending images
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
            H_img_to_mosaic = np.eye(3)
        else:
            ###
            #  2.1) find matches for descriptors
            ###
            good_matches = self._find_good_matches(image_patch.descriptors, self._train_desc)
            
            ###
            #  2.2) filter valid matches
            ###
            # TODO
            #  Keypoints are in local coordinate system! This means that when we add them to  self._train_kp  they refer to the original image,
            #  and not to the patch in the mosaic. Before adding them to self._train_kp, we must transform their  .pt  with the  H_img_to_mosaic
            #  we obtained after the matching process!
            # pairs is (n, m, 3), with  n  2D homogeneous points for each of the  m  sets of selected matches
            pairs = np.ones(shape=(len(good_matches), 2, 3))
            pairs[:, :, 0:2] = [(image_patch.keypoints[match.queryIdx].pt, self._train_kp[match.trainIdx].pt) for match in good_matches]
            
            ###
            #  2.3) actually estimate homography using lines from matches
            ###
            H_img_to_mosaic = self.homography_method(pairs)
            if H_img_to_mosaic is None:
                print(f"Something went very bad with the homography estimation... removing this patch from list")
                self._images.pop(self._i - 1)
                return
        
        image_patch.assign_warping_matrix(H_img_to_mosaic)
        
        ###
        #  3) add keypoints and descriptors to training set
        ###
        self._train_kp = np.append(self._train_kp, image_patch.keypoints)
        self._train_desc = np.vstack([self._train_desc, image_patch.descriptors])
        
        if self.decimation_factor < 1.0:
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
        
        for image_patch in self._images:
            
            ####
            #   warp the patch accordingly to the homography method used
            ####
            
            # image color info
            patch = self.homography_method.warp(image_patch.img, image_patch.H_with_offset, image_patch.warped_bb_size)
            # image shape info
            img_mask = np.ones(shape=image_patch.img.shape[0:2], dtype=np.uint8) * 255
            patch_mask = self.homography_method.warp(img_mask, image_patch.H_with_offset, image_patch.warped_bb_size) == 255
            
            ####
            #   create patch to stitch onto mosaic
            ####
            
            # calculate patch range wrt mosaic
            patch_origin_wrt_mosaic = (image_patch.warped_bb_origin - bb_mosaic_min).astype(int)
            patch_ending_wrt_mosaic = (image_patch.warped_bb_ending - bb_mosaic_min + 1).astype(int)
            patch_y_range_wrt_mosaic = slice(patch_origin_wrt_mosaic[1], patch_ending_wrt_mosaic[1])
            patch_x_range_wrt_mosaic = slice(patch_origin_wrt_mosaic[0], patch_ending_wrt_mosaic[0])
            
            # calculate the seam using the given method
            patch_mask_seam, seam_wrt_patch = self.seam_method(
                self.mosaic, self.mosaic_mask, patch, patch_mask, patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic)
            
            ####
            #   handle color blending
            ####
            # get the weights for the overlapping regions and eventually an updated patch
            weights, patch = self.stitching_method(
                self.mosaic, patch, patch_mask, seam_wrt_patch, patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic, self.crossover_param)
            
            # stitch the warped patch according to the obtained seamed mask (this overwrites the shared regions)
            ref__mosaic_in_patch = self.mosaic[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic, :]
            ref__mosaic_in_patch[patch_mask_seam, :] = patch[patch_mask_seam, :]

            # find the shared region in mosaic reference system
            ref__mosaic_mask_in_patch = self.mosaic_mask[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic]
            patch_mask_shared = np.bitwise_and(ref__mosaic_mask_in_patch, patch_mask_seam)  # mask of the shared region w.r.t. patch
            mosaic_mask_shared = np.zeros_like(self.mosaic_mask)  # mask of the shared region w.r.t. mosaic
            mosaic_mask_shared[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic] = patch_mask_shared

            # use shared mask to blend the colors in the shared region with the weights
            self.mosaic[mosaic_mask_shared, :] \
                = weights[0] * self.mosaic[mosaic_mask_shared, :] \
                + weights[1] * patch[patch_mask_shared, :]
            
            # lastly, because matrix changes are by reference, update the total mask adding the patch mask
            self.mosaic_mask[patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic] = np.bitwise_or(ref__mosaic_mask_in_patch, patch_mask_seam)
        
        # final image still contains float values, so we convert it back
        self.mosaic = self.mosaic.astype(np.uint8)
    
    def balance_warpings(self):
        # calculate image center and recalculate every H so to balance the image
        pass

