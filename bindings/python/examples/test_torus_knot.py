from manifold3d import *
import numpy as np
import pytest

# Creates a classic torus knot, defined as a string wrapping periodically
# around the surface of an imaginary donut. If p and q have a common
# factor then you will get multiple separate, interwoven knots. This is
# an example of using the warp() method, thus avoiding any direct
# handling of triangles.


def run(warp_single=False):
    # The number of times the thread passes through the donut hole.
    p = 1
    # The number of times the thread circles the donut.
    q = 3
    # Radius of the interior of the imaginary donut.
    majorRadius = 25
    # Radius of the small cross-section of the imaginary donut.
    minorRadius = 10
    # Radius of the small cross-section of the actual object.
    threadRadius = 3.75
    # Number of linear segments making up the threadRadius circle. Default is
    # getCircularSegments(threadRadius).
    circularSegments = -1
    # Number of segments along the length of the knot. Default makes roughly
    # square facets.
    linearSegments = -1

    # These default values recreate Matlab Knot by Emmett Lalish:
    # https://www.thingiverse.com/thing:7080

    kLoops = np.gcd(p, q)
    pk = p / kLoops
    qk = q / kLoops
    n = (
        circularSegments
        if circularSegments > 2
        else get_circular_segments(threadRadius)
    )
    m = linearSegments if linearSegments > 2 else n * qk * majorRadius / threadRadius

    offset = 2
    circle = CrossSection.circle(1, n).translate([offset, 0])

    def ax_rotate(x, theta):
        a, b = (x + 1) % 3, (x + 2) % 3
        s, c = np.sin(theta), np.cos(theta)
        m = np.zeros((len(theta), 4, 4), dtype=np.float32)
        m[:, a, a], m[:, a, b] = c, s
        m[:, b, a], m[:, b, b] = -s, c
        m[:, x, x], m[:, 3, 3] = 1, 1
        return m

    def func(pts):
        npts = pts.shape[0]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        psi = qk * np.arctan2(x, y)
        theta = psi * pk / qk
        x1 = np.sqrt(x * x + y * y)
        phi = np.arctan2(x1 - offset, z)

        v = np.zeros((npts, 4), dtype=np.float32)
        v[:, 0] = threadRadius * np.cos(phi)
        v[:, 2] = threadRadius * np.sin(phi)
        v[:, 3] = 1
        r = majorRadius + minorRadius * np.cos(theta)

        m1 = ax_rotate(0, -np.arctan2(pk * minorRadius, qk * r))
        m1[:, 3, 0] = minorRadius
        m2 = ax_rotate(1, theta)
        m2[:, 3, 0] = majorRadius
        m3 = ax_rotate(2, psi)

        v = v[:, None, :] @ m1 @ m2 @ m3
        return v[:, 0, :3]

    def func_single(v):
        pts = np.array([v])
        return func(pts)[0]

    if warp_single:
        return Manifold.revolve(circle, int(m)).warp(func_single)
    else:
        return Manifold.revolve(circle, int(m)).warp_batch(func)


@pytest.mark.parametrize("warp_single", [True, False])
def test_warp(warp_single):
    m = run(warp_single=warp_single)
    assert m.volume() == pytest.approx(20785.76)
    assert m.surface_area() == pytest.approx(11176.8)
    assert m.genus() == 1


if __name__ == "__main__":
    run()
