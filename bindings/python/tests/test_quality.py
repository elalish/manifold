import threading

import pytest

import manifold3d as md
from manifold3d import CircularQuality, CrossSection, JoinType, Manifold


def test_circular_quality_defaults():
    q = CircularQuality()
    assert q.segments == 0
    assert q.min_angle == 10
    assert q.min_edge_length == 1
    assert q.segments_for(1) == 8
    assert CircularQuality(min_angle=22.5, min_edge_length=0.125).segments_for(2) == 16
    assert CircularQuality(12).segments_for(2) == 12


def test_factories_take_int_or_quality():
    q = CircularQuality(min_angle=30, min_edge_length=0.1)
    n = q.segments_for(5)
    assert n == 12
    assert len(CrossSection.circle(5, q).to_polygons()[0]) == n
    assert len(CrossSection.circle(5, quality=q).to_polygons()[0]) == n
    assert len(CrossSection.circle(5, n).to_polygons()[0]) == n
    assert len(CrossSection.circle(5, quality=n).to_polygons()[0]) == n
    assert len(CrossSection.circle(5, circular_segments=n).to_polygons()[0]) == n
    assert Manifold.cylinder(2, 5, quality=q).num_vert() == 2 * n
    assert Manifold.cylinder(2, 5, -1, n, True).num_vert() == 2 * n
    assert Manifold.cylinder(3, 0.3, circular_segments=48).num_vert() == 96
    assert Manifold.sphere(5, q).num_tri() == Manifold.sphere(5, n).num_tri()
    ring = CrossSection.square((1, 1)).translate((4.5, 0))
    assert ring.revolve(q).num_vert() == ring.revolve(circular_segments=n).num_vert()
    assert (
        Manifold.revolve(ring, quality=q).num_vert()
        == Manifold.revolve(ring, n).num_vert()
    )
    tiny = CrossSection.square((1e-3, 1e-3), True)
    assert (
        tiny.offset(5, JoinType.Round, 2, q).num_vert()
        == tiny.offset(5, JoinType.Round, 2, circular_segments=n).num_vert()
    )
    with pytest.raises(TypeError):
        Manifold.sphere(5, q, circular_segments=n)


def test_circular_setters_removed():
    for name in [
        "set_min_circular_angle",
        "set_min_circular_edge_length",
        "set_circular_segments",
        "get_circular_segments",
        "get_min_circular_angle",
        "get_min_circular_edge_length",
    ]:
        assert not hasattr(md, name)


def test_quality_per_thread():
    coarse = CircularQuality(min_angle=90)
    fine = CircularQuality(min_angle=5, min_edge_length=0.01)
    expected = {
        id(coarse): Manifold.sphere(10, coarse).num_tri(),
        id(fine): Manifold.sphere(10, fine).num_tri(),
    }
    assert expected[id(coarse)] != expected[id(fine)]
    wrong = []

    def work(q):
        for _ in range(20):
            if Manifold.sphere(10, q).num_tri() != expected[id(q)]:
                wrong.append(q)

    threads = [threading.Thread(target=work, args=(q,)) for q in [coarse, fine] * 2]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not wrong


def test_for_divisions():
    ring = CrossSection.square((1, 1)).translate((4.5, 0))
    # A partial revolve of this ring has 4 vertices per slice.
    assert ring.revolve(30, 90).num_vert() == 4 * (7 + 1)
    for degrees in [0.7, 360 / 19, 90, 239]:
        for n in [1, 5, 21]:
            q = CircularQuality.for_divisions(n, degrees)
            assert ring.revolve(q, degrees).num_vert() == 4 * (n + 1)
