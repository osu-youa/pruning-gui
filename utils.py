import numpy as np


def line_plane_intersection(plane_normal, plane_pt, ray_direction, ray_point=None, epsilon=1e-6):
    # https://gist.github.com/TimSC/8c25ca941d614bf48ebba6b473747d72

    if ray_point is None:
        ray_point = np.zeros(3)

    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = ray_point - plane_pt
    si = -plane_normal.dot(w) / ndotu
    Psi = w + si * ray_direction + plane_pt
    return Psi
