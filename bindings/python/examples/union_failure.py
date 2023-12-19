from manifold3d import Manifold


def run():
    # for some reason this causes collider error
    obj = Manifold.cube()
    obj += Manifold.cube().rotate([0, 0, 45 + 180])
    return obj
