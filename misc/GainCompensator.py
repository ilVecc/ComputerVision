import numpy as np

def single_feed(corners, images, masks):

    assert corners.size() == images.size() and images.size() == masks.size()

    if images.size() == 0:
        return

    num_channels = images[0].channels()
    assert all([image.channels() == num_channels for image in images])
    assert num_channels == 1 or num_channels == 3

    num_images = len(images)
    N = np.zeros(shape=(num_images, num_images), dtype=np.int32)
    I = np.zeros(shape=(num_images, num_images))
    skip = np.ones(shape=(num_images, 1), dtype=bool)

    Mat subimg1, subimg2
    Mat_<uchar> submask1, submask2, intersect

    std::vector<UMat>::iterator similarity_it = similarities_.begin()

    for i in range(num_images):
        for j in range(i, num_images):
            Rect roi
            if overlapRoi(corners[i], corners[j], images[i].size(), images[j].size(), roi):
                subimg1 = images[i](Rect(roi.tl() - corners[i], roi.br() - corners[i])).getMat(ACCESS_READ)
                subimg2 = images[j](Rect(roi.tl() - corners[j], roi.br() - corners[j])).getMat(ACCESS_READ)

                submask1 = masks[i].first(Rect(roi.tl() - corners[i], roi.br() - corners[i])).getMat(ACCESS_READ)
                submask2 = masks[j].first(Rect(roi.tl() - corners[j], roi.br() - corners[j])).getMat(ACCESS_READ)
                intersect = (submask1 == masks[i].second) & (submask2 == masks[j].second)

                if not similarities_.empty():
                    assert similarity_it != similarities_.end()
                    UMat similarity = *similarity_it++
                    # in-place operation has an issue. don't remove the swap
                    # detail https://github.com/opencv/opencv/issues/19184
                    Mat_<uchar> intersect_updated
                    bitwise_and(intersect, similarity, intersect_updated)
                    std::swap(intersect, intersect_updated)

                intersect_count = np.count_nonzero(intersect)
                N[i, j] = N[j, i] = max(1, intersect_count)

                # Don't compute Isums if subimages do not intersect anyway
                if intersect_count == 0:
                    continue

                # Don't skip images that intersect with at least one other image
                if i != j:
                    skip[i, 0] = False
                    skip[j, 0] = False

                Isum1, Isum2 = 0, 0
                for y in range(roi.height):
                    r1 = subimg1.ptr<Vec<uchar, 3>>(y)
                    r2 = subimg2.ptr<Vec<uchar, 3>>(y)
                    for x in range(roi.width):
                        if intersect[y, x]:
                            Isum1 += np.linalg.norm(r1[x])
                            Isum2 += np.linalg.norm(r2[x])
                I[i, j] = Isum1 / N[i, j]
                I[j, i] = Isum2 / N[i, j]

    if getUpdateGain() or gains_.rows != num_images:
        alpha = 0.01
        beta = 100.0
        num_eq = num_images - np.count_nonzero(skip)
        gains_.create(num_images, 1)
        gains_.setTo(1)

        # No image process, gains are all set to one, stop here
        if num_eq == 0:
            return

        A = np.zeros(shape=(num_eq, num_eq))
        b = np.zeros(shape=(num_eq, 1))
        ki = 0
        for i in range(num_images):
            if skip[i, 0]:
                continue

            kj = 0
            for j in range(num_images):
                if skip[j, 0]:
                    continue

                b[ki, 0] += beta * N[i, j]
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
            if not skip[i, 0]:
                gains_.at<double>(i, 0) = l_gains[j, 0]
                j += 1