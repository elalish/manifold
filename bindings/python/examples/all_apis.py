from manifold3d import *
import numpy as np


def root_level():
    set_min_circular_angle(10)
    set_min_circular_edge_length(1)
    set_circular_segments(22)
    n = get_circular_segments(1)
    assert n == 22
    poly = [[0, 0], [1, 0], [1, 1]]
    tris = triangulate([poly])
    tris = triangulate([np.array(poly)])


def cross_section():
    poly = [[0, 0], [1, 0], [1, 1]]
    c = CrossSection([np.array(poly)])
    c = CrossSection([poly])
    c = c.offset(1, JoinType.Round)


if __name__ == "__main__":
    root_level()
    cross_section()
