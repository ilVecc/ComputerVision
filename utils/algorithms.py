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
        k = np.array([[ 0,  0,  0],
                      [ 0,  3,  0],
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


# https://ieeexplore.ieee.org/document/8676030
def fast_color_blending(patch, patch_mask, seam_in_patch, seam_wrt_patch):
    # use SLIC to compute superpixels and reduce the computational cost
    from skimage.segmentation import slic
    from skimage.measure import regionprops
    
    # find superpixels as per paper
    n_segments = patch_mask.sum() // 400  # we want each superpixel to have around 200 pixels inside it
    segments = slic(patch, n_segments=n_segments, compactness=5, sigma=5, mask=patch_mask)
    segments_centroids_position = np.array([props.centroid for props in regionprops(segments)]).T
    # get actual number of segments
    n_segments = segments_centroids_position.shape[1]
    # labels start from 1 (0 is outside the mask)
    segments_centroids_color = np.array([patch[segments == (i + 1)].mean(axis=0) for i in range(n_segments)])
    
    # apply the color interpolation "as per paper"
    sigma_1, sigma_2 = (0.05 + 0.5) / 2, (0.5 + 5) / 2
    W = patch.shape[1]
    color = seam_in_patch[:, np.newaxis, :] - segments_centroids_color[np.newaxis, :, :]
    distance = seam_wrt_patch[:, :, np.newaxis] - segments_centroids_position[:, np.newaxis, :]
    # TODO
    #  Here the paper uses a  *  without specifying whether this is a multiplication or a convolution.
    #  Since the values are scalars, I supposed the first one and thus simplified applying exponential rules.
    w = np.exp(-(np.linalg.norm(color, ord=2, axis=2) / sigma_1) ** 2 - (np.linalg.norm(distance, ord=2, axis=0) / (W * sigma_2)) ** 2)
    w = (w - w.min()) / (w.max() - w.min())
    
    return w, segments, n_segments


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


def biggest_shared_region_bb(img):
    img = img.astype(np.uint8) * 255
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # find biggest contour
    areas = [cv.contourArea(c) for c in contours]
    return cv.boundingRect(contours[np.argmax(areas)]) if areas else (None, None, None, None)
