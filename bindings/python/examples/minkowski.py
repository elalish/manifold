import numpy as np
from manifold3d import Manifold


def run():
    small_cube = Manifold.cube([0.1, 0.1, 0.1], True)
    cube_vertices = small_cube.to_mesh().vert_properties[:, :3]
    star = Manifold.as_original(small_cube)
    for offset in [
        [[0.2, 0.0, 0.0]],
        [[-0.2, 0.0, 0.0]],
        [[0.0, 0.2, 0.0]],
        [[0.0, -0.2, 0.0]],
        [[0.0, 0.0, 0.2]],
        [[0.0, 0.0, -0.2]],
    ]:
        star += Manifold.hull_points(np.concatenate((cube_vertices, offset), axis=0))

    sphere = Manifold.sphere(0.6, 15)
    cube = Manifold.cube([1.0, 1.0, 1.0], True)
    sphereless_cube = cube - sphere
    return sphereless_cube.minkowski_sum(star)
