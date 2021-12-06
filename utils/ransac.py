import numpy as np


class FittingFunctions(object):
    
    @staticmethod
    def fit_line_PCA(points, idxs):
        points = points[idxs, :]
        assert points.shape[1] >= 3, 'This fitting function works with at least 3 points'
        
        # 3D line: C + d*L
        c = np.mean(points, axis=0)  # centroid
        w, v = np.linalg.eig(np.cov(points))  # just a simple PCA actually
        i = np.argmax(w)  # the biggest eigenvector is the direction
        d = v[:, i]  # direction
        return c, d

    @staticmethod
    def fit_line_points(points, idxs):
        points = points[idxs, :]
        if points.shape != (2, 3):
            print('This fitting function only works with 2 points (taking the first two)')
        
        c = np.mean(points, axis=0)
        p1 = points[0, :]
        p2 = points[1, :]
        d = (p2 - p1)
        d /= np.linalg.norm(d)
        return c, d

    @staticmethod
    def fit_plane(points, idxs):
        points = points[idxs, :]
        # 3D plane: ax + by + cz + d = 0  ==  (X - c) * n = 0
        c = np.mean(points, axis=0)  # centroid [d = -c*n]
        w, v = np.linalg.eig(np.cov(points))  # just a simple PCA actually
        i = np.argmin(w)  # the smallest eigenvector is the normal
        n = v[:, i]  # normal [a b c]
        return c, n


class DistanceFunctions(object):
    
    @staticmethod
    def distance_point_to_line(points, line):
        c, d = line
        h = np.cross(np.repeat(d.T, (points.shape[0], 1)), (points - c), 2)
        d2H = np.linalg.norm(h, ord=2, axis=1)  # distance
        H = points + np.cross(np.repeat(d.T, (points.shape[0], 1)), h, 2)  # projection
        return d2H, H

    @staticmethod
    def distance_point_to_plane(points, plane):
        c, n = plane
        d2H = (points - c) @ n  # distance
        H = points - d2H @ n  # projection
        return d2H, H

    
def ransac(data, k, th, s, fitFun, distFun, desired_score=0):
    """
    RANSAC implementation
    :param data:     numpy array
    :param k:        number of iterations
    :param th:       distance threshold
    :param s:        samples to draw in each iteration
    :param fitFun:   fitting function
    :param distFun:  distance function
    :param desired_score:
    :return:
    """
    best_score = 0
    best_model = None
    best_inliers = None
    for i in range(k):
        idxs = np.random.choice(data.shape[0], size=s, replace=False)
        model = fitFun(data, idxs)
        d, _ = distFun(data, model)
        inliers_set = d < th
        model_score = np.sum(inliers_set)
        if model_score > best_score:
            best_score = model_score
            best_model = model
            best_inliers = inliers_set
        if model_score <= desired_score:
            break

    return best_model, best_inliers, best_score

