void GainCompensator::singleFeed(const std::vector<Point> &corners, const std::vector<UMat> &images,
                                 const std::vector<std::pair<UMat,uchar> > &masks)
{
    CV_Assert(corners.size() == images.size() && images.size() == masks.size());

    if (images.size() == 0)
        return;

    const int num_channels = images[0].channels();
    CV_Assert(std::all_of(images.begin(), images.end(),
        [num_channels](const UMat& image) { return image.channels() == num_channels; }));
    CV_Assert(num_channels == 1 || num_channels == 3);

    const int num_images = static_cast<int>(images.size());
    Mat_<int> N(num_images, num_images); N.setTo(0);
    Mat_<double> I(num_images, num_images); I.setTo(0);
    Mat_<bool> skip(num_images, 1); skip.setTo(true);

    Mat subimg1, subimg2;
    Mat_<uchar> submask1, submask2, intersect;

    std::vector<UMat>::iterator similarity_it = similarities_.begin();

    for (int i = 0; i < num_images; ++i)
    {
        for (int j = i; j < num_images; ++j)
        {
            Rect roi;
            if (overlapRoi(corners[i], corners[j], images[i].size(), images[j].size(), roi))
            {
                subimg1 = images[i](Rect(roi.tl() - corners[i], roi.br() - corners[i])).getMat(ACCESS_READ);
                subimg2 = images[j](Rect(roi.tl() - corners[j], roi.br() - corners[j])).getMat(ACCESS_READ);

                submask1 = masks[i].first(Rect(roi.tl() - corners[i], roi.br() - corners[i])).getMat(ACCESS_READ);
                submask2 = masks[j].first(Rect(roi.tl() - corners[j], roi.br() - corners[j])).getMat(ACCESS_READ);
                intersect = (submask1 == masks[i].second) & (submask2 == masks[j].second);

                if (!similarities_.empty())
                {
                    CV_Assert(similarity_it != similarities_.end());
                    UMat similarity = *similarity_it++;
                    // in-place operation has an issue. don't remove the swap
                    // detail https://github.com/opencv/opencv/issues/19184
                    Mat_<uchar> intersect_updated;
                    bitwise_and(intersect, similarity, intersect_updated);
                    std::swap(intersect, intersect_updated);
                }

                int intersect_count = countNonZero(intersect);
                N(i, j) = N(j, i) = std::max(1, intersect_count);

                // Don't compute Isums if subimages do not intersect anyway
                if (intersect_count == 0)
                    continue;

                // Don't skip images that intersect with at least one other image
                if (i != j)
                {
                    skip(i, 0) = false;
                    skip(j, 0) = false;
                }

                double Isum1 = 0, Isum2 = 0;
                for (int y = 0; y < roi.height; ++y)
                {
                    if (num_channels == 3)
                    {
                        const Vec<uchar, 3>* r1 = subimg1.ptr<Vec<uchar, 3> >(y);
                        const Vec<uchar, 3>* r2 = subimg2.ptr<Vec<uchar, 3> >(y);
                        for (int x = 0; x < roi.width; ++x)
                        {
                            if (intersect(y, x))
                            {
                                Isum1 += norm(r1[x]);
                                Isum2 += norm(r2[x]);
                            }
                        }
                    }
                    else // if (num_channels == 1)
                    {
                        const uchar* r1 = subimg1.ptr<uchar>(y);
                        const uchar* r2 = subimg2.ptr<uchar>(y);
                        for (int x = 0; x < roi.width; ++x)
                        {
                            if (intersect(y, x))
                            {
                                Isum1 += r1[x];
                                Isum2 += r2[x];
                            }
                        }
                    }
                }
                I(i, j) = Isum1 / N(i, j);
                I(j, i) = Isum2 / N(i, j);
            }
        }
    }
    if (getUpdateGain() || gains_.rows != num_images)
    {
        double alpha = 0.01;
        double beta = 100;
        int num_eq = num_images - countNonZero(skip);
        gains_.create(num_images, 1);
        gains_.setTo(1);

        // No image process, gains are all set to one, stop here
        if (num_eq == 0)
            return;

        Mat_<double> A(num_eq, num_eq); A.setTo(0);
        Mat_<double> b(num_eq, 1); b.setTo(0);
        for (int i = 0, ki = 0; i < num_images; ++i)
        {
            if (skip(i, 0))
                continue;

            for (int j = 0, kj = 0; j < num_images; ++j)
            {
                if (skip(j, 0))
                    continue;

                b(ki, 0) += beta * N(i, j);
                A(ki, ki) += beta * N(i, j);
                if (j != i)
                {
                    A(ki, ki) += 2 * alpha * I(i, j) * I(i, j) * N(i, j);
                    A(ki, kj) -= 2 * alpha * I(i, j) * I(j, i) * N(i, j);
                }
                ++kj;
            }
            ++ki;
        }

        Mat_<double> l_gains;

#ifdef HAVE_EIGEN
        Eigen::MatrixXf eigen_A, eigen_b, eigen_x;
        cv2eigen(A, eigen_A);
        cv2eigen(b, eigen_b);

        Eigen::LLT<Eigen::MatrixXf> solver(eigen_A);
#if ENABLE_LOG
        if (solver.info() != Eigen::ComputationInfo::Success)
            LOGLN("Failed to solve exposure compensation system");
#endif
        eigen_x = solver.solve(eigen_b);

        Mat_<float> l_gains_float;
        eigen2cv(eigen_x, l_gains_float);
        l_gains_float.convertTo(l_gains, CV_64FC1);
#else
        solve(A, b, l_gains);
#endif
        CV_CheckTypeEQ(l_gains.type(), CV_64FC1, "");

        for (int i = 0, j = 0; i < num_images; ++i)
        {
            // Only assign non-skipped gains. Other gains are already set to 1
            if (!skip(i, 0))
                gains_.at<double>(i, 0) = l_gains(j++, 0);
        }
    }
}