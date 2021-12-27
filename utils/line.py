import numpy as np


def line2D(pa, pb):
    m = (pb[1] - pa[1]) / (pb[0] - pa[0])
    q = -m*pa[0] + pa[1]
    return m, q


def fit_line2D(points, idxs):
    if idxs.size > 2:
        raise RuntimeError("Line must be fitted with maximum 2 points")
    selected_points = points[idxs, :]
    return line2D(selected_points[0, :], selected_points[1, :])


def distance_line2D(points, line):
    m = line[0]
    q = line[1]
    points = np.vstack([points, np.zeros(shape=(1, points.shape[1]))]).T
    line_d = np.array([[np.cos(m), np.sin(m), 0]]).T
    line_c = np.array([[0, q, 0]]).T
    return distance_points_to_line(points, line_d, line_c)


def distance_points_to_line(P, line_d, line_c):
  h = np.cross(np.tile(line_d.T, reps=(P.shape[0], 1)), P - line_c.T, axis=1)
  d2H = np.linalg.norm(h, ord=2, axis=1)  # distance
  H = P + np.cross(np.tile(line_d.T, reps=(P.shape[0], 1)), h, axis=1)  # projection
  return d2H, H
