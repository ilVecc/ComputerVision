from pathlib import Path

from cv2 import cv2 as cv
import numpy as np

from stitching.stitcher_classes import HomographyMethod, SeamMethod, StitchingMethod, TrimmingMethod
from utils.algorithms import cylindrical_warp
from utils.homography import apply_homogeneous


class ImagePatch(object):
    
    _sift = cv.SIFT_create()
    
    def __init__(self, path, load=False):
        self.path = path
        # loading
        self.is_loaded = False
        self.img = None
        self.mask = None
        self.is_described = False
        self.keypoints = None  # keypoints position is updated after assigning the warping
        self.descriptors = None
        self._gray = None
        # warping
        self.has_warp = False
        self.H = None
        self.offset = None
        self.H_without_offset = None
        self.warped_img = None
        self.warped_mask = None
        self.warped_bb = None  # [origin_x, origin_y, ending_x, ending_y], w.r.t. H reference frame
        self.warped_bb_origin = None  # [origin_x, origin_y], w.r.t. H reference frame
        self.warped_bb_ending = None  # [ending_x, ending_y], w.r.t. H reference frame
        self.warped_bb_size = None  # size of the bb after warping
        self.warped_bb_center = None  # [x, y], w.r.t. H reference frame
        
        if load:
            self.load_and_sift()
    
    def load_and_sift(self):
        if self.is_loaded:
            print(f"Image {self.path} has been already loaded")
            return
        
        try:
            self.img = cv.imread(self.path)
            self.mask = np.ones(shape=self.img.shape[0:2], dtype=bool)
            # TODO think again this
            # self.img[:, :, 0] = cv.equalizeHist(self.img[:, :, 0])
            # self.img[:, :, 1] = cv.equalizeHist(self.img[:, :, 1])
            # self.img[:, :, 2] = cv.equalizeHist(self.img[:, :, 2])
        except Exception as ex:
            print(f"Failed to load image {self.path} due to exception:")
            print(ex)

        K = np.eye(3)
        K[0, 0] = 3215
        K[1, 1] = 3256
        K[0, 2] = 2175 // 2
        K[1, 2] = 1725 // 2
        K[0, 1] = 10
        
        img_rgba = cylindrical_warp(self.img, K)
        self.img = img_rgba[:, :, 0:3]
        self.mask = img_rgba[:, :, 3] != 0

        self.is_loaded = True
        self._gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.keypoints, self.descriptors = self._sift.detectAndCompute(self._gray, np.uint8(self.mask) * 255)
        self.is_described = True

    def assign_warping_matrix(self, H):
        self.has_warp = True
        self.H = H
        # transform the keypoints'  .pt  with the  H  homography, so to have the new keypoints' position after the warping
        kp_pt = np.array([kp.pt for kp in self.keypoints])
        warped_kp_pt = apply_homogeneous(kp_pt.T, self.H)
        for i in range(len(self.keypoints)):
            self.keypoints[i].pt = tuple(warped_kp_pt[:, i])

    def preprocess_warping(self, warp_H=None):
        
        if warp_H is None:
            if self.has_warp:
                warp_H = self.H
            else:
                raise RuntimeError("No warping transform provided!")

        h, w, d = self.img.shape
        # trace border warping
        border = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        border_warp = apply_homogeneous(border.T, warp_H).T.astype(int)
        # trace center warping
        center = np.array([w / 2, h / 2])
        self.warped_bb_center = apply_homogeneous(center.T, warp_H).T

        # compute warped bounding box extrema wrt "world"
        bb_world_min_xy = np.min(border_warp, axis=0)
        bb_world_max_xy = np.max(border_warp, axis=0)
        self.warped_bb = np.hstack([bb_world_min_xy, bb_world_max_xy])  # [x_min, y_min, x_max, y_max]
        
        # set useful values and compute warped image size wrt world
        self.warped_bb_origin = self.warped_bb[0:2]
        self.warped_bb_ending = self.warped_bb[2:4]
        self.warped_bb_size = (self.warped_bb_ending - self.warped_bb_origin + 1).astype(int)

        # add an anti-offset transform
        # Since OpenCV warps the image without any other processing (so most of the time the image is painted outside the viewbox of given size),
        # this procedure is three-fold:
        #   1) we completely fill the viewbox (by setting the final dimensions of the world image to  self.warped_bb_size )
        #   2) we limit the world dimensions (which is then simply the warped image and nothing else)
        #   3) we can actually show the whole image (since it's completely inside the viewbox)
        # Note that in order to put the warped image in the world reference system we must use  self.warped_bb_origin  and  self.warped_bb_size
        self.offset = np.array([[1, 0, -self.warped_bb_origin[0]],
                                [0, 1, -self.warped_bb_origin[1]],
                                [0, 0, 1]])
        self.H_without_offset = self.offset @ self.H
    
    def apply_warping(self, homography_method):

        ###
        #  apply homography and offset via  cv.warpPerspective  (most of the time)
        #
        #     Viewbox precomputation
        #     ----------------------
        #     Since we precomputed the warped bounding box of each patch, we have the size of the image that completely contains the warped patch; this is our
        #     patch's viewbox. Now, we expand all the viewboxes to find the viewbox of the whole mosaic, which contains all of them, and use the previously
        #     computed offsets to put each warped patch in its correct location inside the mosaic viewbox, simply called  self.mosaic .
        ###

        # Here we use  image_patch.H_without_offset  because otherwise (using  image_patch.H ) the warped image
        # would not be inside the given bounding box, which is centered in (0, 0).
        # The relationship of this image and the world image is still described by  H .
        
        self.warped_img = homography_method.warp(self.img, self.H_without_offset, self.warped_bb_size)
        self.warped_mask = homography_method.warp(np.uint8(self.mask) * 255, self.H_without_offset, self.warped_bb_size) == 255
    
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
                 seam_method=SeamMethod.NONE,
                 stitching_method=StitchingMethod.AVERAGE, stitching_param=None,
                 trimming_method=TrimmingMethod.NONE,
                 decimation_factor=0.1):
        # parameters
        self.homography_method = homography_method
        self.seam_method = seam_method
        self.stitching_method = stitching_method
        self.stitching_param = stitching_param
        self.trimming_method = trimming_method
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

        ###
        #  3) assign homography to patch and precompute warped patch bounding box
        ###
        image_patch.assign_warping_matrix(H_img_to_mosaic)
        image_patch.preprocess_warping()
        
        ###
        #  4) add keypoints and descriptors to training set
        ###
        self._update_training_set(image_patch)
    
    def _update_training_set(self, image_patch):
        self._train_kp = np.append(self._train_kp, image_patch.keypoints)
        self._train_desc = np.vstack([self._train_desc, image_patch.descriptors])
        
        if self.decimation_factor > 0.0:
            train_size = len(self._train_kp)
            train_size_new = int((1 - self.decimation_factor) * train_size)
            filter_idxs = np.random.choice(np.arange(train_size), size=train_size_new, replace=False)
            
            self._train_kp = self._train_kp[filter_idxs]
            self._train_desc = self._train_desc[filter_idxs]
        
    def stitch_all(self):
        
        # bounding box of the whole mosaic wrt world
        bb_mosaic_min = np.min(np.vstack([image_patch.warped_bb_origin for image_patch in self._images]), axis=0)  # [min_x, min_y] wrt world frame
        bb_mosaic_max = np.max(np.vstack([image_patch.warped_bb_ending for image_patch in self._images]), axis=0)  # [max_x, max_y] wrt world frame
        
        # initialize empty mosaic image
        mosaic_size = (bb_mosaic_max - bb_mosaic_min + 1).astype(int)
        self.mosaic = np.zeros(shape=(mosaic_size[1], mosaic_size[0], 3), dtype=np.uint8)
        self.mosaic_mask = np.zeros(shape=(mosaic_size[1], mosaic_size[0]), dtype=np.bool)
        
        # stitch each image one by one
        for image_patch in self._images:
            
            ####
            #   warp the patch accordingly to the homography method used
            ####
            
            # image color and mask info
            image_patch.apply_warping(self.homography_method)
            
            ####
            #   create patch to stitch onto mosaic
            ####
            
            # calculate patch range wrt mosaic
            patch_origin_wrt_mosaic = (image_patch.warped_bb_origin - bb_mosaic_min).astype(int)
            patch_ending_wrt_mosaic = (image_patch.warped_bb_ending - bb_mosaic_min + 1).astype(int)  # +1 is for  [origin, ending)
            patch_y_range_wrt_mosaic = slice(patch_origin_wrt_mosaic[1], patch_ending_wrt_mosaic[1])
            patch_x_range_wrt_mosaic = slice(patch_origin_wrt_mosaic[0], patch_ending_wrt_mosaic[0])
            
            # calculate the seam using the given method and the patch mask accounting the seam
            patch_mask_seam, seam_wrt_patch = self.seam_method(
                self.mosaic, self.mosaic_mask,
                image_patch.warped_img, image_patch.warped_mask,
                patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic
            )
            
            ####
            #   handle color blending
            ####
            self.mosaic, self.mosaic_mask = self.stitching_method(
                self.mosaic, self.mosaic_mask,
                image_patch.warped_img, patch_mask_seam,
                seam_wrt_patch,
                patch_y_range_wrt_mosaic, patch_x_range_wrt_mosaic, self.stitching_param
            )

        # cut the image
        self.mosaic, self.mosaic_mask = self.trimming_method(self.mosaic, self.mosaic_mask)
    
    def balance_warpings(self, use_translation=False, rigorous=False):
        # calculate image center and recalculate every H so to balance the image

        Hs = np.vstack([image_patch.H[np.newaxis, ...] for image_patch in self._images])

        # we should do something like this
        # 1) average of the translations in the homographies
        # 2) average of the rotations in the homographies
        # 3) average of the projections in the homographies
        # but the approaches below are fine enough
        
        if use_translation:
            # TODO both these approaches feel very wrong, but somehow are working
            if rigorous:
                # classical eigenvectors approach
                w, v = np.linalg.eig(Hs)  # val, vec
                w_avg = w.mean(axis=0)
                v_avg = v.mean(axis=0)
                H_avg = v_avg @ np.diag(w_avg) @ np.linalg.inv(v_avg)
                H_avg /= H_avg[2, 2]
            else:
                # simple mean of the homographies
                H_avg = Hs.mean(axis=0)
        
        else:
            # just use the mean of the translation to find the homography "in the barycenter" of the mosaic
            ts = Hs[:, 0:2, 2]
            t_avg = ts.mean(axis=0)
            
            dist = np.linalg.norm(ts - t_avg.T, axis=1)
            H_avg = Hs[np.argmin(dist), ...]
        
        # apply the inverse of the average homography to center all the homographies
        H_avg_inv = np.linalg.inv(H_avg)
        for image_patch in self._images:
            H_new = H_avg_inv @ image_patch.H
            # re-preprocess everything
            image_patch.assign_warping_matrix(H_new)
            image_patch.preprocess_warping()
        
        return H_avg
