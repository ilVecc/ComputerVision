import cv2.cv2 as cv
import numpy as np

WEIGHT_EPS = 1e-5

# Ptr<Blender> Blender::createDefault(int type, bool try_gpu):
#     if (type == NO)
#         return makePtr<Blender>()
#     if (type == FEATHER)
#         return makePtr<FeatherBlender>(try_gpu)
#     if (type == MULTI_BAND)
#         return makePtr<MultiBandBlender>(try_gpu)
#     CV_Error(Error::StsBadArg, "unsupported blending method")
#
#
# void Blender::prepare(const std::vector<Point> &corners, const std::vector<Size> &sizes):
#     prepare(resultRoi(corners, sizes))
#
# void Blender::prepare(Rect dst_roi):
#     dst_.create(dst_roi.size(), CV_16SC3)
#     dst_.setTo(Scalar::all(0))
#     dst_mask_.create(dst_roi.size(), CV_8U)
#     dst_mask_.setTo(Scalar::all(0))
#     dst_roi_ = dst_roi
#
# void Blender::feed(InputArray _img, InputArray _mask, Point tl):
#     Mat img = _img.getMat()
#     Mat mask = _mask.getMat()
#     Mat dst = dst_.getMat(ACCESS_RW)
#     Mat dst_mask = dst_mask_.getMat(ACCESS_RW)
#
#     CV_Assert(img.type() == CV_16SC3)
#     CV_Assert(mask.type() == CV_8U)
#     int dx = tl[0] - dst_roi_[0]
#     int dy = tl[1] - dst_roi_[1]
#
#     for (int y = 0; y < img.rows; ++y):
#         const Point3_<short> *src_row = img.ptr<Point3_<short> >(y)
#         Point3_<short> *dst_row = dst.ptr<Point3_<short> >(dy + y)
#         const uchar *mask_row = mask.ptr<uchar>(y)
#         uchar *dst_mask_row = dst_mask.ptr<uchar>(dy + y)
#
#         for (int x = 0; x < img.cols; ++x):
#             if (mask_row[x])
#                 dst_row[dx + x] = src_row[x]
#             dst_mask_row[dx + x] |= mask_row[x]
#
# void Blender::blend(InputOutputArray dst, InputOutputArray dst_mask):
#     UMat mask
#     compare(dst_mask_, 0, mask, CMP_EQ)
#     dst_.setTo(Scalar::all(0), mask)
#     dst.assign(dst_)
#     dst_mask.assign(dst_mask_)
#     dst_.release()
#     dst_mask_.release()


class MultiBandBlender(object):
    
    def __init__(self, num_bands, weight_type):
        self.num_bands_ = num_bands
    
        assert weight_type == cv.CV_32F or weight_type == cv.CV_16S
        self.weight_type_ = weight_type
        self.dst_roi_final_ = None
    
    def prepare(self, dst_roi):
        self.dst_roi_final_ = dst_roi
    
        # Crop unnecessary bands
        max_len = max(dst_roi.width, dst_roi.height)
        self.num_bands_ = min(self.actual_num_bands_, np.int(np.ceil(np.log(max_len) / np.log(2.0))))
    
        # Add border to the final image, to ensure sizes are divided by (1 << num_bands_)
        dst_roi.width += ((1 << self.num_bands_) - dst_roi.width % (1 << self.num_bands_)) % (1 << self.num_bands_)
        dst_roi.height += ((1 << self.num_bands_) - dst_roi.height % (1 << self.num_bands_)) % (1 << self.num_bands_)
    
        Blender_prepare(dst_roi)
        dst_pyr_laplace_.resize(num_bands_ + 1)
        dst_pyr_laplace_[0] = dst_
    
        dst_band_weights_.resize(num_bands_ + 1)
        dst_band_weights_[0].create(dst_roi.shape, weight_type_)
        dst_band_weights_[0, :] = 0
    
        for i in range(1, self.num_bands_ + 1):
            dst_pyr_laplace_[i].create((dst_pyr_laplace_[i - 1].rows + 1) / 2, (dst_pyr_laplace_[i - 1].cols + 1) / 2, cv.CV_16SC3)
            dst_band_weights_[i].create((dst_band_weights_[i - 1].rows + 1) / 2, (dst_band_weights_[i - 1].cols + 1) / 2, weight_type_)
            dst_pyr_laplace_[i].setTo(Scalar::all(0))
            dst_band_weights_[i, :] = 0
    
    def feed(self, img, mask, tl):
        #
        # assert img.dtype == cv.CV_16SC3 or img.dtype == cv.CV_8UC3
        # assert mask.dtype == cv.CV_8U
        #
        # # Keep source image in memory with small border
        # gap = 3 * (1 << self.num_bands_)
        # tl_new = [max(dst_roi_[0], tl[0] - gap), max(dst_roi_[1], tl[1] - gap)]
        # br_new = [min(dst_roi_.br()[0], tl[0] + img.cols + gap), min(dst_roi_.br()[1], tl[1] + img.rows + gap)]
        #
        # # Ensure coordinates of top-left, bottom-right corners are divided by (1 << num_bands_).
        # # After that scale between layers is exactly 2.
        # #
        # # We do it to avoid interpolation problems when keeping sub-images only. There is no such problem when
        # # image is bordered to have size equal to the final image size, but this is too memory hungry approach.
        # tl_new[0] = dst_roi_[0] + (((tl_new[0] - dst_roi_[0]) >> num_bands_) << num_bands_)
        # tl_new[1] = dst_roi_[1] + (((tl_new[1] - dst_roi_[1]) >> num_bands_) << num_bands_)
        # width = br_new[0] - tl_new[0]
        # height = br_new[1] - tl_new[1]
        # width += ((1 << num_bands_) - width % (1 << num_bands_)) % (1 << num_bands_)
        # height += ((1 << num_bands_) - height % (1 << num_bands_)) % (1 << num_bands_)
        # br_new[0] = tl_new[0] + width
        # br_new[1] = tl_new[1] + height
        # dy = max(br_new[1] - dst_roi_.br()[1], 0)
        # dx = max(br_new[0] - dst_roi_.br()[0], 0)
        # tl_new[0] -= dx
        # br_new[0] -= dx
        # tl_new[1] -= dy
        # br_new[1] -= dy
        #
        # top = tl[1] - tl_new[1]
        # left = tl[0] - tl_new[0]
        # bottom = br_new[1] - tl[1] - img.rows
        # right = br_new[0] - tl[0] - img.cols
        #
        # # Create the source image Laplacian pyramid
        # img_with_border = cv.copyMakeBorder(img, top, bottom, left, right, borderType=cv.BORDER_REFLECT)
        # src_pyr_laplace = createLaplacePyr(img_with_border, self.num_bands_)
        #
        # # Create the weight map Gaussian pyramid
        # if self.weight_type_ == cv.CV_32F:
        #     weight_map = np.float64(mask) / 255
        # else: # weight_type_ == CV_16S
        #     weight_map = np.float64(mask)
        #     add_mask = mask != 0
        #     weight_map[add_mask] += 1
        # weight_pyr_gauss = [cv.copyMakeBorder(weight_map, top, bottom, left, right, borderType=cv.BORDER_CONSTANT)]
        # for i in range(self.num_bands_):
        #     weight_pyr_gauss.append(cv.pyrDown(weight_pyr_gauss[i]))
        #
        # y_tl = tl_new[1] - dst_roi_[1]
        # y_br = br_new[1] - dst_roi_[1]
        # x_tl = tl_new[0] - dst_roi_[0]
        # x_br = br_new[0] - dst_roi_[0]
        # # Add weighted layer of the source image to the final Laplacian pyramid layer
        # for i in range(self.num_bands_ + 1):
        #     Rect rc(x_tl, y_tl, x_br - x_tl, y_br - y_tl)
        #     Mat _src_pyr_laplace = src_pyr_laplace[i]
        #     Mat _dst_pyr_laplace = dst_pyr_laplace_[i](rc)
        #     Mat _weight_pyr_gauss = weight_pyr_gauss[i]
        #     Mat _dst_band_weights = dst_band_weights_[i](rc)
        #     if self.weight_type_ == cv.CV_32F:
        #         for y in range(rc.height):
        #             src_row = _src_pyr_laplace[y, :]
        #             dst_row = _dst_pyr_laplace[y, :]
        #             weight_row = _weight_pyr_gauss[y, :]
        #             dst_weight_row = _dst_band_weights[y, :]
        #
        #             dst_row[:, :] += src_row[:, :] * weight_row
        #             dst_weight_row += weight_row
        #
        #     else:  # weight_type_ == cv.CV_16S
        #         for y in range(y_br - y_tl):
        #             src_row = _src_pyr_laplace[y, :]
        #             dst_row = _dst_pyr_laplace[y, :]
        #             weight_row = _weight_pyr_gauss[y, :]
        #             dst_weight_row = _dst_band_weights[y, :]
        #
        #             dst_row[:, :] += (src_row[:, :] * weight_row) >> 8
        #             dst_weight_row += weight_row
        #
        #     x_tl /= 2
        #     y_tl /= 2
        #     x_br /= 2
        #     y_br /= 2
        #
        
        # make gaussian pyramid
        # make laplacian pyramid
        # multiply them
        pass
    
    def MultiBandBlender_blend(self, dst, dst_mask):
        Rect dst_rc(0, 0, dst_roi_final_.width, dst_roi_final_.height)
        
        for i in range(num_bands_ + 1):
            normalizeUsingWeightMap(dst_band_weights_[i], dst_pyr_laplace_[i])
    
        blended = restoreImageFromLaplacePyr(dst_pyr_laplace_)
    
        self.dst_ = blended(dst_rc)
        self.dst_mask_ = dst_band_weights_[0](dst_rc) > WEIGHT_EPS
    
        Blender_blend(dst, dst_mask)


#######################################
# Auxiliary functions

def normalizeUsingWeightMap(weight, src):
    src[:, :, :] = src[:, :, :] / (weight[:, :] + WEIGHT_EPS)


def createWeightMap(mask, sharpness):
    weight = cv.distanceTransform(mask, cv.DIST_L1, 3) * sharpness
    weight = cv.threshold(weight, 1.0, 1.0, cv.THRESH_TRUNC)
    return weight


def createLaplacePyr(img, num_levels):
    pyr = []
    for i in range(num_levels - 1):
        next_img = cv.pyrDown(img)
        pyr.append(img - cv.pyrUp(next_img, dstsize=img.shape[1::-1]))
        img = next_img
    pyr.append(img)
    return pyr


def restoreImageFromLaplacePyr(pyr):
    if len(pyr) == 0:
        return
    for i in range(len(pyr) - 1, 0, -1):
        pyr[i - 1] += cv.pyrUp(pyr[i], dstsize=pyr[i - 1].size())
    return pyr[0]
