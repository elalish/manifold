from manifold3d import Manifold


def fractal(holes, hole, w, position, depth, maxDepth):
    w /= 3
    holes.append(hole.scale([w, w, 1.0]).translate([position[0], position[1], 0.0]))
    if depth == maxDepth:
        return
    offsets = [
        [-w, -w],
        [-w, 0.0],
        [-w, w],
        [0.0, w],
        [w, w],
        [w, 0.0],
        [w, -w],
        [0.0, -w],
    ]
    for offset in offsets:
        fractal(holes, hole, w, position + offset, depth + 1, maxDepth)


def posColors(newProp, pos):
    for i in [0, 1, 2]:
        newProp[i] = (1 - pos[i]) / 2


def run(n=1):
    result = Manifold.cube([1, 1, 1], True)
    holes = []
    fractal(holes, result, 1.0, [0.0, 0.0], 1, n)

    hole = Manifold.compose(holes)

    result -= hole
    result -= hole.rotate([90, 0, 0])
    result -= hole.rotate([0, 90, 0])

    return result.trim_by_plane([1, 1, 1], 0).setProperties(3, posColors).scale(100)
