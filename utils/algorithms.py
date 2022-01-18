import cv2.cv2 as cv
import numpy as np


# https://ieeexplore.ieee.org/document/5304214
def energy_based_seam_line(img, theta, theta_mask, fast_method=True):
    # The  theta  image derivative is required, and we could use the Sobel operator.
    # OpenCV tells us that it's better to use the Scharr operator, so we do so.
    # https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    img = img.astype(np.uint8)
    theta = theta.astype(np.uint8)

    # construct the Energy function
    e = 0.5 * cv.convertScaleAbs(cv.Scharr(theta, ddepth=-1, dx=1, dy=0)).astype(np.double) \
        + 0.5 * cv.convertScaleAbs(cv.Scharr(theta, ddepth=-1, dx=0, dy=1)).astype(np.double)

    # TODO this  .sum()  is not specified in the original paper, but I see no other way around this...
    img = img.sum(axis=2).astype(np.double)
    e = e.sum(axis=2)

    # construct the Interactive Penalty Factor function
    n, m, c = theta.shape
    if fast_method:
        # We replaced the 2-for monstrosity with a specifically designed kernel and a convolution for better computational cost.
        # It doesn't yield to the same F matrix, but the result has equivalent meaning and thus e_ipf works quite as well.
        k = np.array([[0, 0, 0],
                      [0, 3, 0],
                      [-1, -1, -1]])
        diff_color = cv.filter2D(img, ddepth=cv.CV_64F, kernel=k, borderType=cv.BORDER_ISOLATED)
        diff_error = cv.filter2D(e, ddepth=cv.CV_64F, kernel=k, borderType=cv.BORDER_ISOLATED)
        F = 0.1 * diff_color + 0.9 * diff_error
    else:
        F = np.zeros(shape=(n, m))
        for i in range(n):  # y-axis
            for j in range(m):  # x-axis
                if theta_mask[i, j]:
                    diff_color, diff_error = 0, 0
                    # left neighbor
                    if j - 1 >= 0 and theta_mask[i, j - 1]:
                        diff_color += img[i, j] - img[i, j - 1]
                        diff_error += e[i, j] - e[i, j - 1]
                    # right neighbor
                    if j + 1 < m and theta_mask[i, j + 1]:
                        diff_color += img[i, j] - img[i, j + 1]
                        diff_error += e[i, j] - e[i, j + 1]
                    # bottom neighbor
                    if i + 1 < n and theta_mask[i + 1, j]:
                        diff_color += img[i, j] - img[i + 1, j]
                        diff_error += e[i, j] - e[i + 1, j]
                    F[i, j] = 0.1 * diff_color + 0.9 * diff_error

    # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    e_ipf = cv.filter2D(e, ddepth=cv.CV_64F, kernel=F, borderType=cv.BORDER_ISOLATED)  # performs cross correlation!

    # GREEDY IMPLEMENTATION
    best_score = float('inf')
    best_cols = []
    for j in range(m):  # x-axis
        # check that j-column in first row is inside mask
        if not theta_mask[0, j]:
            continue
        cols = [j]
        score = e_ipf[0, j]
        # for each next i-row
        for i in range(1, n):  # y-axis
            best_next_score = float('inf')
            best_next_col = None
            # check the three underlying neighbors (must be inside bb and inside mask)
            next_cols = [cols[-1] - 1, cols[-1], cols[-1] + 1]
            for next_col in next_cols:
                if 0 <= next_col < m and theta_mask[i, next_col]:
                    next_score = e_ipf[i, next_col]
                    # use as best next column
                    if next_score < best_next_score:
                        best_next_score = next_score
                        best_next_col = next_col
            # if no next column was found (because bb and mask cut off every possibility),
            # simply trace a straight line going down and repeatedly add the last available score,
            # ending the j-column search and moving on to the next one
            if best_next_col is None:
                remaining_rows = n - len(cols)
                score += remaining_rows * e_ipf[i - 1, cols[-1]]
                cols.extend([cols[-1]] * remaining_rows)
                break

            # if a next column was found, add it and its score, then move on to the next i-row
            cols.append(best_next_col)
            score += best_next_score

        # update the best line if the score is better
        if score < best_score:
            best_score = score
            best_cols = cols

    best_pixels = np.vstack([np.arange(len(best_cols)), best_cols])  # pixels are [y; x]
    return best_pixels


# https://ieeexplore.ieee.org/document/8676030  or  https://ieeexplore.ieee.org/document/8676030
def fast_color_blending(patch, patch_mask, seam_color_in_patch, seam_coords_wrt_patch, use_distance=True):
    # use SLIC to compute superpixels and reduce the computational cost
    from skimage.segmentation import slic
    from skimage.measure import regionprops

    # find superpixels as per paper
    # we should want each superpixel to have around 200 pixels inside it, but this is very slow on large images,
    # so we simply fix a number of desired segments to speed everything up
    n_segments = np.sum(patch_mask) // 300
    segments = slic(patch, n_segments=n_segments, compactness=7, sigma=5, mask=patch_mask)
    segments_centroids_coords = np.array([props.centroid for props in regionprops(segments)]).T
    # get actual number of segments
    n_segments = segments_centroids_coords.shape[1]
    # labels start from 1 (0 is outside the mask)
    segments_centroids_color_in_patch = np.array([patch[segments == (i + 1)].mean(axis=0) for i in range(n_segments)])

    diff_color = seam_color_in_patch[:, np.newaxis, :] - segments_centroids_color_in_patch[np.newaxis, :, :]
    if use_distance:
        # https://ieeexplore.ieee.org/document/8676030
        diff_color /= 255
        diff_coord = seam_coords_wrt_patch[:, :, np.newaxis] - segments_centroids_coords[:, np.newaxis, :]

        # # cool distance visualization
        # patch_copy = patch.copy()
        # for i in range(657):
        #     x1 = int(segments_centroids_coords[1, i])
        #     y1 = int(segments_centroids_coords[0, i])
        #     x2 = int(x1 + diff_coord[1, 240, i])
        #     y2 = int(y1 + diff_coord[0, 240, i])
        #     cv.line(patch_copy, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)
        # import matplotlib.pyplot as pt; pt.imshow(patch_copy); pt.show()

        # apply the color interpolation "as per paper"
        sigma_1, sigma_2, W = 0.05, 0.5, patch.shape[1]

        # TODO
        #  Here the paper uses a  *  without specifying whether this is a multiplication or a convolution.
        #  Since the values are scalars, I supposed the first one and thus simplified applying exponential rules.
        w = np.exp(-(np.linalg.norm(diff_color, ord=2, axis=2) / sigma_1) ** 2 - (np.linalg.norm(diff_coord, ord=2, axis=0) / (W * sigma_2)) ** 2)

        # exp_color = np.exp(-(np.linalg.norm(diff_color, ord=2, axis=2) / sigma_1) ** 2)
        # exp_coord = np.exp(-(np.linalg.norm(diff_coord, ord=2, axis=0) / (W * sigma_2)) ** 2)
        # w = cv.filter2D(exp_color, ddepth=cv.CV_64F, kernel=exp_coord, borderType=cv.BORDER_ISOLATED)  # performs cross correlation!

        # normalize weights
        w = (w - w.min(axis=0)) / (w.max(axis=0) - w.min(axis=0))
        w[w == float('nan')] = 0  # when the range is 0 we have a division by zero, so we fix the weight to 0

    else:
        # https://ieeexplore.ieee.org/document/9115682/
        sigma = 50
        w = np.exp(-0.5 * (np.linalg.norm(diff_color, ord=2, axis=2) / sigma) ** 2)

    return w, segments, n_segments


# def multi_channel_blending(img1, img2, mask1, mask2):
#     def GaussianPyramid(img, leveln):
#         GP = [img]
#         for i in range(leveln - 1):
#             GP.append(cv.pyrDown(GP[i]))
#         return GP
#
#     def LaplacianPyramid(img, leveln):
#         LP = []
#         for i in range(leveln - 1):
#             next_img = cv.pyrDown(img)
#             LP.append(img - cv.pyrUp(next_img, dstsize=img.shape[1::-1]))
#             img = next_img
#         LP.append(img)
#         return LP
#
#     def blend_pyramid(LPA, LPB, MPA, MPB):
#         blended = []
#         for LA, LB, MA, MB in zip(LPA, LPB, MPA, MPB):
#             blended.append(LA * MA + LB * MB)
#         return blended
#
#     def reconstruct(LS):
#         img = LS[-1]
#         for lev_img in LS[-2::-1]:
#             img = cv.pyrUp(img, dstsize=lev_img.shape[1::-1])
#             img += lev_img
#         img = np.clip(img, 0, 255)
#         return img
#
#     # Get Gaussian pyramid and Laplacian pyramid
#     leveln = int(np.floor(np.log2(min(img1.shape[0], img2.shape[1]))))
#
#     mask = np.bitwise_or(mask1, mask2)
#     img1 = img1.astype(np.float64)  # [0, 255]
#     img2 = img2.astype(np.float64)  # [0, 255]
#     mask1 = mask1.astype(np.float64)  # [0, 1]
#     mask2 = mask2.astype(np.float64)  # [0, 1]
#     LP1 = LaplacianPyramid(img1, leveln)
#     LP2 = LaplacianPyramid(img2, leveln)
#     MP1 = GaussianPyramid(mask1, leveln)
#     MP2 = GaussianPyramid(mask2, leveln)
#
#     # Blend two Laplacian pyramids
#     blended_pyramids = blend_pyramid(LP1, LP2, MP1, MP2)
#
#     # Reconstruction process
#     blended_img = reconstruct(blended_pyramids)
#     blended_img = np.uint8(blended_img)
#     blended_img = blended_img * mask
#
#     return blended_img


def multi_band_blending(A, B, M):
    
    num_levels = int(np.floor(np.log2(min(*A.shape[:2], *B.shape[:2]))))
    
    # gaussian pyramid
    gpA = [np.float32(A.copy())]
    gpB = [np.float32(B.copy())]
    gpM = [np.float32(M.copy())]
    for i in range(num_levels):
        gpA.append(cv.pyrDown(gpA[i]))
        gpB.append(cv.pyrDown(gpB[i]))
        gpM.append(cv.pyrDown(gpM[i]))
    
    # laplacian pyramid
    gpMr = gpM[::-1]
    lpA = [gpA[num_levels]]
    lpB = [gpB[num_levels]]
    for i in range(num_levels, 0, -1):
        size = gpA[i - 1].shape[1::-1]
        LA = gpA[i - 1] - cv.pyrUp(gpA[i], dstsize=size)
        LB = gpB[i - 1] - cv.pyrUp(gpB[i], dstsize=size)
        lpA.append(LA)
        lpB.append(LB)
    
    # blend
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1 - gm)
        LS.append(ls)
    
    # reconstruct
    ls_ = LS[0]
    for lev_img in LS[1:]:
        ls_ = lev_img + cv.pyrUp(ls_, dstsize=lev_img.shape[1::-1])
    ls_ = np.clip(ls_, 0, 255)
    
    return np.uint8(ls_)


def multi_band_blending_masked(A, B, MA, MB):
    num_levels = int(np.floor(np.log2(min(*A.shape[:2], *B.shape[:2]))))
    
    # global mask
    M = np.clip(MA + MB, 0, 1)
    
    # gaussian pyramid
    gpA = [np.float32(A.copy())]
    gpB = [np.float32(B.copy())]
    gpMA = [np.float32(MA.copy())]
    gpMB = [np.float32(MB.copy())]
    gpM = [np.float32(M.copy())]
    for i in range(num_levels):
        gpA.append(cv.pyrDown(gpA[i]))
        gpB.append(cv.pyrDown(gpB[i]))
        gpMA.append(cv.pyrDown(gpM[i]))
        gpMB.append(cv.pyrDown(gpM[i]))
        gpM.append(cv.pyrDown(gpM[i]))
    
    # laplacian pyramid
    gpMAr = gpMA[::-1]
    gpMBr = gpMB[::-1]
    gpMr = gpM[::-1]
    lpA = [gpA[num_levels]]
    lpB = [gpB[num_levels]]
    for i in range(num_levels, 0, -1):
        size = gpA[i - 1].shape[1::-1]
        LA = gpA[i - 1] - cv.pyrUp(gpA[i], dstsize=size)
        LB = gpB[i - 1] - cv.pyrUp(gpB[i], dstsize=size)
        lpA.append(LA)
        lpB.append(LB)
    
    # blend
    LS = []
    for la, lb, gma, gmb, gm in zip(lpA, lpB, gpMAr, gpMBr, gpMr):
        ls = (la * gma + lb * gmb) * gm
        LS.append(ls)
    
    # reconstruct
    ls_ = LS[0]
    for lev_img in LS[1:]:
        ls_ = lev_img + cv.pyrUp(ls_, dstsize=lev_img.shape[1::-1])
    ls_ = np.clip(ls_, 0, 255)
    
    return np.uint8(ls_)


# https://docs.opencv.org/3.4/d3/d05/tutorial_py_table_of_contents_contours.html
def show_blobs(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # draw BBs of detected contours in green
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # draw detected contours in red
    img = cv.drawContours(img, contours, -1, (0, 0, 255), 2)
    cv.imshow("Contours and BBs", img)
    cv.waitKey(0)


def biggest_bb_in_mask(img):
    img = np.uint8(img) * 255
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None
    # find biggest contour
    areas = [cv.contourArea(c) for c in contours]
    return cv.boundingRect(contours[np.argmax(areas)])


# https://stackoverflow.com/a/30136404/7742892
def biggest_rectangle_in_mask(mask):
    # set up initial parameters
    r, c = mask.shape
    ul_r = 0    # upper-left row        (param #0)
    ul_c = 0    # upper-left column     (param #1)
    br_r = r-1  # bottom-right row      (param #2)
    br_c = c-1  # bottom-right column   (param #3)

    parameters = [0, 1, 2, 3]  # parameters left to be updated
    pidx = 0  # index of parameter currently being updated

    # shrink region until acceptable
    while len(parameters) > 0:  # update until all parameters reach bounds

        # 1. update parameter number
        pidx %= len(parameters)
        p = parameters[pidx]  # current parameter number

        # 2. update current parameter
        # 3. grab newest part of region (row or column)
        if p == 0:
            ul_r += 1
            border = mask[ul_r, ul_c:(br_c+1)]
        elif p == 1:
            ul_c += 1
            border = mask[ul_r:(br_r+1), ul_c]
        elif p == 2:
            br_r -= 1
            border = mask[br_r, ul_c:(br_c+1)]
        else:
            br_c -= 1
            border = mask[ul_r:(br_r+1), br_c]

        # 4. if the new region has only zeros, stop shrinking the current parameter
        if np.count_nonzero(border) == border.size:
            del parameters[pidx]

        pidx += 1

    return ul_r, ul_c, br_r, br_c


def cylindrical_warp(img, K):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_, w_ = img.shape[:2]

    # pixel coordinates
    y_i, x_i = np.indices((h_, w_))
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
    X = X @ np.linalg.inv(K).T  # normalized coords  [ (np.linalg.inv(K) @ X.T).T ]

    # calculate cylindrical coords (sin(θ), h, cos(θ)
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
    B = A @ K.T  # project back to image-pixels plane  [ (K @ A.T).T ]
    B = B[:, :-1] / B[:, [-1]]  # back from homog coords
    B = B.reshape(h_, w_, -1)
    B = B.astype(np.float32)
    
    # make sure warp coords only within image bounds
    C = B.copy()
    C[(C[:, :, 0] < 0) | (C[:, :, 0] >= w_) | (C[:, :, 1] < 0) | (C[:, :, 1] >= h_)] = -1
    # the line above breaks reflection! the following implements it just for the color part
    B = abs(B)
    B[B[:, :, 0] >= w_] = w_ - (B[B[:, :, 0] >= w_] - w_)
    B[B[:, :, 1] >= h_] = h_ - (B[B[:, :, 1] >= h_] - h_)

    # warp the image according to cylindrical coords
    img_rgba = cv.cvtColor(img, cv.COLOR_BGR2BGRA)  # for transparent borders...
    img_rgba_warped = cv.remap(
        img_rgba,
        map1=B[:, :, 0],  # map_x
        map2=B[:, :, 1],  # map_y
        interpolation=cv.INTER_AREA,
        borderMode=cv.BORDER_REFLECT  # this is useless, everything has been mapped
    )
    # https://answers.opencv.org/question/89028/blending-artifacts-in-opencv-image-stitching/
    img_rgba_warped[:, :, 3] = cv.remap(
        img_rgba[:, :, 3],
        map1=C[:, :, 0],
        map2=C[:, :, 1],
        interpolation=cv.INTER_AREA,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0
    )
    return img_rgba_warped

    # img_warped = img_rgba_warped[:, :, 0:3]
    # mask_warped = img_rgba_warped[:, :, 3] == 0


def histeq_color(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    cv.equalizeHist(src=channels[0], dst=channels[0])  # equalize intensity channel
    cv.merge(channels, dst=ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, dst=img)
    return img


def gain_intensity(img, gain):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    y = np.float64(channels[0]) * gain  # equalize intensity channel
    y = np.clip(y, 0, 255)
    y = np.uint8(np.round(y))
    channels = (y, channels[1], channels[2])
    cv.merge(channels, dst=ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, dst=img)
    return img


def gain_rbg(img, gain):
    img = np.float64(img) * gain  # equalize channels
    img = np.clip(img, 0, 255)
    img = np.uint8(np.round(img))
    return img


def overlap_mask(origin1, origin2, mask1, mask2):
    y_min = min(origin1[1], origin2[1])
    y_max = max(origin1[1] + mask1.shape[0], origin2[1] + mask2.shape[0])
    x_min = min(origin1[0], origin2[0])
    x_max = max(origin1[0] + mask1.shape[1], origin2[0] + mask2.shape[1])
    mask = np.zeros(shape=(y_max - y_min, x_max - x_min), dtype=bool)
    mask[
        origin1[1] - y_min: origin1[1] - y_min + mask1.shape[0],
        origin1[0] - x_min: origin1[0] - x_min + mask1.shape[1]
    ] = mask1
    ref__mask_in_mask2_region = mask[
        origin2[1] - y_min: origin2[1] - y_min + mask2.shape[0],
        origin2[0] - x_min: origin2[0] - x_min + mask2.shape[1]
    ]
    overlap_wrt_mask2 = np.bitwise_and(ref__mask_in_mask2_region, mask2)
    x, y, w, h = biggest_bb_in_mask(overlap_wrt_mask2)
    if x is None:
        return (None, None), np.empty(shape=(), dtype=bool)
    overlap = overlap_wrt_mask2[y:y + h, x:x + w]
    return (x + origin2[0], y + origin2[1]), overlap


class GainCompensator(object):
    
    def __init__(self):
        self.similarity_thresh = 1.0
        self._similarities = []
        self._gains = np.empty(shape=(0,))
        self._nr_feeds = 1
    
    def apply(self, index, corner, image, mask):
        return image * self._gains[index]
    
    # noinspection PyUnboundLocalVariable
    def feed(self, corners, images, masks):
        num_images = len(images)
        self.prepare_similarity_mask(corners, images, masks)
        
        for n in range(self._nr_feeds):
            if n > 0:
                # Apply previous iteration gains
                for i in range(num_images):
                    images[i] = self.apply(i, corners[i], images[i], masks[i])
            
            self.single_feed(corners, images, masks)
            accumulated_gains = self._gains.copy() if n == 0 else accumulated_gains * self._gains
            
        self._gains = accumulated_gains
        return self._gains
    
    def single_feed(self, origins, images, masks):
        
        assert len(origins) == len(images) and len(images) == len(masks)
        
        if len(images) == 0:
            return
        
        num_channels = 1 if len(images[0].shape) < 2 else images[0].shape[2]
        assert all([(1 if len(images[0].shape) < 2 else images[0].shape[2]) == num_channels for image in images])
        if num_channels == 1:
            images = [image[..., np.newaxis] for image in images]
        assert num_channels == 1 or num_channels == 3
        
        num_images = len(images)
        N = np.zeros(shape=(num_images, num_images), dtype=np.int32)
        I = np.zeros(shape=(num_images, num_images))
        skip = np.ones(shape=(num_images,), dtype=bool)
        
        similarity_it = 0
        for i in range(num_images):
            for j in range(i, num_images):
                overlap_origin, overlap = overlap_mask(origins[i], origins[j], masks[i], masks[j])
                if overlap_origin[0] is not None:
                    subimg1 = images[i][
                              overlap_origin[1] - origins[i][1]: overlap_origin[1] - origins[i][1] + overlap.shape[0],
                              overlap_origin[0] - origins[i][0]: overlap_origin[0] - origins[i][0] + overlap.shape[1]]
                    subimg2 = images[j][
                              overlap_origin[1] - origins[j][1]: overlap_origin[1] - origins[j][1] + overlap.shape[0],
                              overlap_origin[0] - origins[j][0]: overlap_origin[0] - origins[j][0] + overlap.shape[1]]
                    submask1 = masks[i][
                               overlap_origin[1] - origins[i][1]: overlap_origin[1] - origins[i][1] + overlap.shape[0],
                               overlap_origin[0] - origins[i][0]: overlap_origin[0] - origins[i][0] + overlap.shape[1]]
                    submask2 = masks[j][
                               overlap_origin[1] - origins[j][1]: overlap_origin[1] - origins[j][1] + overlap.shape[0],
                               overlap_origin[0] - origins[j][0]: overlap_origin[0] - origins[j][0] + overlap.shape[1]]
                    intersect = np.bitwise_and(submask1, submask2)
                    
                    # if similarities have been set, use them
                    if len(self._similarities) > 1:
                        intersect = np.bitwise_and(intersect, self._similarities[similarity_it])
                        similarity_it += 1
                    
                    intersect_count = np.count_nonzero(intersect)
                    N[i, j] = N[j, i] = max(1, intersect_count)
                    
                    # Don't compute Isums if subimages do not intersect anyway
                    if intersect_count == 0:
                        continue
                    
                    # Don't skip images that intersect with at least one other image
                    if i != j:
                        skip[i] = False
                        skip[j] = False
                    
                    I[i, j] = np.sum(np.linalg.norm(subimg1, axis=2) * intersect) / N[i, j]
                    I[j, i] = np.sum(np.linalg.norm(subimg2, axis=2) * intersect) / N[i, j]
        
        if len(self._gains) != num_images:
            alpha = 0.01
            beta = 100.0
            num_eq = num_images - np.count_nonzero(skip)
            self._gains = np.ones(shape=(num_images,))
            
            # No image process, gains are all set to one, stop here
            if num_eq == 0:
                return
            
            A = np.zeros(shape=(num_eq, num_eq))
            b = np.zeros(shape=(num_eq,))
            ki = 0
            for i in range(num_images):
                if skip[i]:
                    continue
                
                kj = 0
                for j in range(num_images):
                    if skip[j]:
                        continue
                    
                    b[ki] += beta * N[i, j]
                    A[ki, ki] += beta * N[i, j]
                    if j != i:
                        A[ki, ki] += 2 * alpha * I[i, j] * I[i, j] * N[i, j]
                        A[ki, kj] -= 2 * alpha * I[i, j] * I[j, i] * N[i, j]
                    kj += 1
                ki += 1
            
            l_gains = np.linalg.solve(A, b)
            
            j = 0
            for i in range(num_images):
                # Only assign non-skipped gains. Other gains are already set to 1
                if not skip[i]:
                    self._gains[i] = l_gains[j]
                    j += 1
        return self._gains
    
    def prepare_similarity_mask(self, origins, images, masks):
        
        if self.similarity_thresh >= 1:
            print("  skipping similarity mask: disabled")
            return
        
        if len(self._similarities) != 0:
            print("  skipping similarity mask: already set")
            return
        
        print("  calculating similarity mask")
        num_images = len(images)
        for i in range(num_images):
            for j in range(i, num_images):
                overlap_origin, overlap = overlap_mask(origins[i], origins[j], masks[i], masks[j])
                if overlap_origin[0] is not None:
                    subimg1 = images[i][
                              overlap_origin[1] - origins[i][1]: overlap_origin[1] - origins[i][1] + overlap.shape[0],
                              overlap_origin[0] - origins[i][0]: overlap_origin[0] - origins[i][0] + overlap.shape[1]]
                    subimg2 = images[j][
                              overlap_origin[1] - origins[j][1]: overlap_origin[1] - origins[j][1] + overlap.shape[0],
                              overlap_origin[0] - origins[j][0]: overlap_origin[0] - origins[j][0] + overlap.shape[1]]
                    similarity = self.build_similarity_mask(subimg1, subimg2)
                    self._similarities.append(similarity)
    
    def build_similarity_mask(self, src1, src2):
        
        assert src1.shape[0] == src2.shape[0] and src1.shape[1] == src2.shape[1]
        assert src1.dtype == src2.dtype == np.uint8
        assert len(src1.shape) < 3 or src1.shape[2] == 1 or src1.shape[2] == 3
        
        # similarity = np.zeros(shape=(src1.rows, src1.cols), dtype=cv.CV_8UC1)
        # if src1.channels() == 3:
        #     for y in range(similarity.shape[0]):
        #         for x in range(similarity.shape[1]):
        #             diff = np.linalg.norm((src1[y, x, :] - src2[y, x, :]) / 255.0)
        #             similarity[y, x] = 255 if diff <= similarity_threshold_ else 0
        
        similarity = (np.linalg.norm((src1 - src2) / 255.0, axis=2) <= self.similarity_thresh) * 255
        similarity = np.uint8(similarity)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        similarity = cv.erode(similarity, kernel)
        similarity = cv.dilate(similarity, kernel)
        return similarity
