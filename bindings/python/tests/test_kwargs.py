import pytest

from manifold3d import CrossSection, ExecutionContext, JoinType, Manifold


def sdf(x, y, z):
    return 1 - (x * x + y * y + z * z) ** 0.5


BOUNDS = [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]


def test_snake_case_keywords():
    m = Manifold.cube(size=(1, 1, 1))
    cs = CrossSection.square(size=(1, 1))
    assert m.translate(v=(1, 0, 0)).bounding_box()[0] == 1
    assert m.scale(v=(2, 1, 1)).bounding_box()[3] == 2
    assert m.mirror(normal=(1, 0, 0)).bounding_box()[0] == -1
    assert m.rotate(degrees=(0, 0, 180)).bounding_box()[0] == pytest.approx(-1)
    assert cs.translate(v=(1, 0)).bounds()[0] == 1
    assert cs.scale(v=(2, 1)).bounds()[2] == 2
    assert cs.mirror(normal=(1, 0)).bounds()[0] == -1
    assert cs.offset(delta=0.1, join_type=JoinType.Miter).area() > 1
    assert Manifold.extrude(cross_section=cs, height=1).volume() == 1
    assert Manifold.revolve(cross_section=cs.translate((1, 0))).volume() > 0
    assert Manifold.hull_points(points=m.to_mesh().vert_properties).volume() == 1
    assert CrossSection.hull_points(points=[(0, 0), (1, 0), (0, 1)]).area() == 0.5
    assert not Manifold.level_set(sdf=sdf, bounds=BOUNDS, edge_length=0.5).is_empty()
    ctx = ExecutionContext()
    assert not ctx.level_set(sdf=sdf, bounds=BOUNDS, edge_length=0.5).is_empty()
    props = m.set_properties(num_prop=1, prop_func=lambda pos, old: [1.0])
    assert props.num_prop() == 1


@pytest.mark.parametrize(
    "call",
    [
        lambda: Manifold.cube().translate(t=(1, 0, 0)),
        lambda: Manifold.cube().mirror(v=(1, 0, 0)),
        lambda: Manifold.cube().rotate(v=(0, 0, 90)),
        lambda: CrossSection.square((1, 1)).mirror(ax=(1, 0)),
        lambda: CrossSection.square((1, 1)).scale(scale=(2, 1)),
        lambda: Manifold.extrude(crossSection=CrossSection.square((1, 1)), height=1),
        lambda: Manifold.revolve(crossSection=CrossSection.square((1, 1))),
        lambda: Manifold.hull_points(pts=[(0, 0, 0)]),
        lambda: CrossSection.hull_points(pts=[(0, 0)]),
        lambda: Manifold.level_set(f=sdf, bounds=BOUNDS, edge_length=0.5),
        lambda: Manifold.level_set(sdf, BOUNDS, edgeLength=0.5),
        lambda: Manifold.cube().set_properties(new_num_prop=3, prop_func=None),
        lambda: Manifold.cube().set_properties(num_prop=3, f=lambda p, o: p),
    ],
)
def test_old_keywords_rejected(call):
    with pytest.raises(TypeError):
        call()
