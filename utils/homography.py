import numpy as np


def skew(vector):
    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])


# cv.findHomography()
# https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gafd3ef89257e27d5235f4467cbb1b6a63
def homography(m1, m2):
    n = m1.shape[1]
    if n < 4:
        raise ArithmeticError(f"Homography estimation requires at least 4 pairs ({n} provided)")
    A = np.zeros(shape=(2*n, 9))
    for i in range(n):
        k = np.kron(m1[:, i].T, skew(m2[:, i]))
        # we keep just the first two lines since the 3rd one
        # is linearly dependent from the others
        A[2*i:2*i+2, :] = k[0:2, :]
    _, _, VT = np.linalg.svd(A)
    # last row contains a vector from the kernel
    H = np.reshape(VT[-1, :], newshape=(3, 3)).T
    H /= H[2, 2]
    
    return H


def fit_homography(points, idxs):
    src_points, dst_points = points[idxs, 0, :], points[idxs, 1, :]
    return homography(src_points.T, dst_points.T)


def distance_homography(pairs, H):
    # reprojection error:  m2 ~= H m1
    m1, m2 = pairs[:, 0, :].T, pairs[:, 1, :].T
    # m1 = np.hstack([m1, np.ones(shape=(1, m1.shape[0]))])
    # m2 = np.hstack([m1, np.ones(shape=(1, m1.shape[0]))])
    
    # classic test
    m2_reproj = H @ m1
    # FIXME sometimes here we have a divide by zero error
    m2_reproj /= m2_reproj[2, :]
    dist2 = np.linalg.norm(m2 - m2_reproj, axis=0)
    
    # symmetric transfer function test
    m1_reproj = np.linalg.inv(H) @ m2
    # FIXME sometimes here we have a divide by zero error
    m1_reproj /= m1_reproj[2, :]
    dist1 = np.linalg.norm(m1 - m1_reproj, axis=0)
    
    # dist = dist2  # classic test
    dist = np.linalg.norm([dist1, dist2], axis=0)  # symmetric transfer function test
    
    return dist, m2_reproj


def test_degenerate_samples(points, idxs):
    src_points, dst_points = points[idxs, 0, :], points[idxs, 1, :]
    
    # https://doi.org/10.1007/978-3-642-17691-3_19
    def process(pt):
        pt = np.vstack([pt, pt[0:2, :]])
        vec = pt[1:] - pt[:-1]
        cross = np.cross(vec[:-1], vec[1:])[:, -1]
        return np.sign(cross), np.any(np.abs(cross) < tol)

    tol = 1e-2
    src_sign, src_collinearity = process(src_points)
    if src_collinearity:
        return False
    dst_sign, dst_collinearity = process(src_points)
    if dst_collinearity:
        return False
    
    return np.alltrue(src_sign == dst_sign)


def apply_homogeneous(data, T):
    """
    
    :param data:    np.ndarray of dimensions (2, n, m) or (3, n, m), where  m  is the number of sets (default  m = 1)
    :param T:       transformation to be applied
    :return:        transformed points
    """
    assert len(data.shape) <= 3, "Data can have at most  m  sets or  n  rows, either 2D or 3D"
    homo_shape =  [1] + list(data.shape[1:])
    homo_data = np.concatenate([data, np.ones(shape=homo_shape)], axis=0)
    assert T.shape[1] == homo_data.shape[0], "Transform and data must have same dimension"
    T_data = T @ homo_data
    T_data /= T_data[-1, ...]
    return T_data[:-1, ...]
