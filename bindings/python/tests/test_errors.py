import numpy as np
import pytest

import manifold3d as m3d
from manifold3d import (
    CircularQuality,
    CrossSection,
    Error,
    ExecutionContext,
    Manifold,
    ManifoldCancelled,
    ManifoldError,
    Mesh,
    Mesh64,
    OpType,
)


def test_error_names_and_messages():
    errors = list(Error)
    assert [e.value for e in errors] == list(range(len(errors)))
    names = [e.name for e in errors]
    messages = [str(e) for e in errors]
    assert all(names) and all(messages)
    assert "Unknown error" not in messages
    assert len(set(names)) == len(names)
    assert len(set(messages)) == len(messages)
    for e in errors:
        assert Error[e.name] is e
        assert str(e) != repr(e)


def test_exception_types():
    assert issubclass(ManifoldError, Exception)
    assert not issubclass(ManifoldError, RuntimeError)
    assert issubclass(ManifoldCancelled, ManifoldError)
    with pytest.raises(ManifoldError) as e:
        Manifold.cube((-1, 1, 1))
    assert type(e.value) is ManifoldError
    assert e.value.code == Error.InvalidConstruction
    assert str(e.value) == str(Error.InvalidConstruction)
    ctx = ExecutionContext()
    ctx.cancel()
    with pytest.raises(ManifoldCancelled) as e:
        ctx.from_mesh(Manifold.cube().to_mesh())
    assert e.value.code == Error.Cancelled
    assert str(e.value) == str(Error.Cancelled)


NAN = float("nan")
NFV = Error.NonFiniteVertex
IC = Error.InvalidConstruction
NM = Error.NotManifold
A, B, C, D = "A", "B", "C", "D"


def bad():
    """A lazily errored Manifold: making it doesn't raise, evaluating it gives
    NonFiniteVertex."""
    return Manifold.cube().translate((NAN, 0, 0))


def open_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    return Mesh(verts, np.array([[0, 1, 2]], np.uint32))


def open_mesh64():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64)
    return Mesh64(verts, np.array([[0, 1, 2]], np.uint64))


def sphere_sdf(x, y, z):
    return 1 - (x * x + y * y + z * z) ** 0.5


BOUNDS = [-1.2, -1.2, -1.2, 1.2, 1.2, 1.2]
SQUARE = CrossSection.square((1, 1))
CUBE = Manifold.cube()

# Every public binding name and its category from #2089. A and B entries list
# (code, call) fixtures that must raise ManifoldError with that code. C entries
# list calls on errored inputs that must not raise, and no calls when they take
# no Manifold. D entries are like A and also have a cancel fixture in CANCEL.
CATEGORIES = {
    "Manifold.__init__": (
        A,
        [
            (NM, lambda: Manifold(open_mesh())),
            (NM, lambda: Manifold(open_mesh64())),
        ],
    ),
    "Manifold.tetrahedron": (A, []),
    "Manifold.cube": (A, [(IC, lambda: Manifold.cube((-1, 1, 1)))]),
    "Manifold.cylinder": (A, [(IC, lambda: Manifold.cylinder(-1, 1))]),
    "Manifold.sphere": (A, [(IC, lambda: Manifold.sphere(-1))]),
    "Manifold.extrude": (A, [(IC, lambda: Manifold.extrude(SQUARE, 0))]),
    "Manifold.revolve": (
        A,
        [
            (IC, lambda: Manifold.revolve(SQUARE, revolve_degrees=NAN)),
        ],
    ),
    "Manifold.level_set": (
        A,
        [
            (IC, lambda: Manifold.level_set(sphere_sdf, BOUNDS, 0)),
            (IC, lambda: Manifold.level_set(sphere_sdf, BOUNDS[3:] + BOUNDS[:3], 0.5)),
        ],
    ),
    "Manifold.hull_points": (A, []),
    "Manifold.batch_hull": (A, [(NFV, lambda: Manifold.batch_hull([CUBE, bad()]))]),
    "Manifold.reserve_ids": (C, []),
    "Manifold.__add__": (C, [lambda: bad() + CUBE, lambda: CUBE + bad()]),
    "Manifold.__sub__": (C, [lambda: bad() - CUBE, lambda: CUBE - bad()]),
    "Manifold.__xor__": (C, [lambda: bad() ^ CUBE, lambda: CUBE ^ bad()]),
    "Manifold.batch_boolean": (
        C,
        [lambda op=op: Manifold.batch_boolean([bad(), CUBE], op) for op in OpType],
    ),
    "Manifold.translate": (C, [lambda: bad().translate((1, 0, 0))]),
    "Manifold.rotate": (C, [lambda: bad().rotate((0, 0, 90))]),
    "Manifold.scale": (C, [lambda: bad().scale((1, 2, 3)), lambda: bad().scale(2)]),
    "Manifold.mirror": (
        C,
        [
            lambda: bad().mirror((0, 0, 1)),
            lambda: bad().mirror((0, 0, 0)),
        ],
    ),
    "Manifold.transform": (
        C,
        [lambda: bad().transform([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])],
    ),
    "Manifold.trim_by_plane": (C, [lambda: bad().trim_by_plane((0, 0, 1), 0.5)]),
    "Manifold.with_context": (C, [lambda: bad().with_context(ExecutionContext())]),
    "Manifold.status": (C, [lambda: bad().status()]),
    "Manifold.warp": (A, [(NFV, lambda: bad().warp(lambda v: v))]),
    "Manifold.warp_batch": (A, [(NFV, lambda: bad().warp_batch(lambda v: v))]),
    "Manifold.set_properties": (
        A,
        [
            (NFV, lambda: bad().set_properties(3, lambda p, old: p)),
            (IC, lambda: CUBE.set_properties(-1, lambda p, old: [])),
        ],
    ),
    "Manifold.calculate_curvature": (
        A,
        [
            (NFV, lambda: bad().calculate_curvature(0, 1)),
        ],
    ),
    "Manifold.calculate_normals": (A, [(NFV, lambda: bad().calculate_normals())]),
    "Manifold.smooth_by_normals": (A, [(NFV, lambda: bad().smooth_by_normals())]),
    "Manifold.simplify": (A, [(NFV, lambda: bad().simplify())]),
    "Manifold.as_original": (A, [(NFV, lambda: bad().as_original())]),
    "Manifold.decompose": (A, [(NFV, lambda: bad().decompose())]),
    "Manifold.split": (
        A,
        [
            (NFV, lambda: bad().split(CUBE)),
            (NFV, lambda: CUBE.split(bad())),
        ],
    ),
    "Manifold.split_by_plane": (
        A,
        [
            (NFV, lambda: bad().split_by_plane((0, 0, 1), 0.5)),
        ],
    ),
    "Manifold.refine": (D, [(NFV, lambda: bad().refine(2))]),
    "Manifold.refine_to_length": (
        D,
        [
            (NFV, lambda: bad().refine_to_length(0.5)),
            (IC, lambda: CUBE.refine_to_length(0)),
        ],
    ),
    "Manifold.refine_to_tolerance": (
        D,
        [
            (NFV, lambda: bad().refine_to_tolerance(0.1)),
            (IC, lambda: CUBE.refine_to_tolerance(0)),
        ],
    ),
    "Manifold.hull": (D, [(NFV, lambda: bad().hull())]),
    "Manifold.minkowski_sum": (
        D,
        [
            (NFV, lambda: bad().minkowski_sum(CUBE)),
            (NFV, lambda: CUBE.minkowski_sum(bad())),
        ],
    ),
    "Manifold.minkowski_difference": (
        D,
        [
            (NFV, lambda: bad().minkowski_difference(CUBE)),
            (NFV, lambda: CUBE.minkowski_difference(bad())),
        ],
    ),
    "Manifold.with_precision": (D, [(NFV, lambda: bad().with_precision(1e-3))]),
    "Manifold.get_precision": (D, [(NFV, lambda: bad().get_precision())]),
    "Manifold.to_mesh": (B, [(NFV, lambda: bad().to_mesh())]),
    "Manifold.to_mesh64": (B, [(NFV, lambda: bad().to_mesh64())]),
    "Manifold.is_empty": (B, [(NFV, lambda: bad().is_empty())]),
    "Manifold.num_vert": (B, [(NFV, lambda: bad().num_vert())]),
    "Manifold.num_edge": (B, [(NFV, lambda: bad().num_edge())]),
    "Manifold.num_tri": (B, [(NFV, lambda: bad().num_tri())]),
    "Manifold.num_prop": (B, [(NFV, lambda: bad().num_prop())]),
    "Manifold.num_prop_vert": (B, [(NFV, lambda: bad().num_prop_vert())]),
    "Manifold.bounding_box": (B, [(NFV, lambda: bad().bounding_box())]),
    "Manifold.genus": (B, [(NFV, lambda: bad().genus())]),
    "Manifold.volume": (B, [(NFV, lambda: bad().volume())]),
    "Manifold.surface_area": (B, [(NFV, lambda: bad().surface_area())]),
    "Manifold.original_id": (B, [(NFV, lambda: bad().original_id())]),
    "Manifold.min_gap": (
        B,
        [
            (NFV, lambda: bad().min_gap(CUBE, 1)),
            (NFV, lambda: CUBE.min_gap(bad(), 1)),
        ],
    ),
    "Manifold.ray_cast": (B, [(NFV, lambda: bad().ray_cast((0, 0, -1), (0, 0, 1)))]),
    "Manifold.winding_number": (B, [(NFV, lambda: bad().winding_number([(0, 0, 0)]))]),
    "Manifold.slice": (B, [(NFV, lambda: bad().slice(0.5))]),
    "Manifold.project": (B, [(NFV, lambda: bad().project())]),
    "CrossSection.__init__": (C, []),
    "CrossSection.even_odd": (C, []),
    "CrossSection.non_zero": (C, []),
    "CrossSection.square": (C, []),
    "CrossSection.circle": (C, []),
    "CrossSection.area": (C, []),
    "CrossSection.num_vert": (C, []),
    "CrossSection.num_contour": (C, []),
    "CrossSection.is_empty": (C, []),
    "CrossSection.bounds": (C, []),
    "CrossSection.translate": (C, []),
    "CrossSection.rotate": (C, []),
    "CrossSection.scale": (C, []),
    "CrossSection.mirror": (C, []),
    "CrossSection.transform": (C, []),
    "CrossSection.warp": (C, []),
    "CrossSection.warp_batch": (C, []),
    "CrossSection.simplify": (C, []),
    "CrossSection.offset": (C, []),
    "CrossSection.__add__": (C, []),
    "CrossSection.__sub__": (C, []),
    "CrossSection.__xor__": (C, []),
    "CrossSection.batch_boolean": (C, []),
    "CrossSection.hull": (C, []),
    "CrossSection.batch_hull": (C, []),
    "CrossSection.hull_points": (C, []),
    "CrossSection.decompose": (C, []),
    "CrossSection.to_polygons": (C, []),
    "CrossSection.extrude": (A, [(IC, lambda: SQUARE.extrude(0))]),
    "CrossSection.revolve": (A, [(IC, lambda: SQUARE.revolve(revolve_degrees=NAN))]),
    "ExecutionContext.__init__": (C, []),
    "ExecutionContext.cancel": (C, []),
    "ExecutionContext.cancelled": (C, []),
    "ExecutionContext.progress": (C, []),
    "ExecutionContext.from_mesh": (
        D,
        [
            (NM, lambda: ExecutionContext().from_mesh(open_mesh())),
            (NM, lambda: ExecutionContext().from_mesh(open_mesh64())),
        ],
    ),
    "ExecutionContext.level_set": (
        D,
        [
            (IC, lambda: ExecutionContext().level_set(sphere_sdf, BOUNDS, 0)),
            (
                IC,
                lambda: ExecutionContext().level_set(
                    sphere_sdf, BOUNDS[3:] + BOUNDS[:3], 0.5
                ),
            ),
        ],
    ),
    "CircularQuality.segments_for": (C, []),
    "CircularQuality.for_divisions": (C, []),
    "triangulate": (C, []),
}

# Calls that cannot give an error today. They are checked anyway.
CANNOT_FAIL = {"Manifold.tetrahedron", "Manifold.hull_points"}


def cancelled_tree(ctx):
    return (Manifold.cube() + Manifold.sphere(1)).with_context(ctx)


CANCEL = {
    "Manifold.refine": lambda ctx: cancelled_tree(ctx).refine(2),
    "Manifold.refine_to_length": lambda ctx: cancelled_tree(ctx).refine_to_length(0.5),
    "Manifold.refine_to_tolerance": lambda ctx: cancelled_tree(ctx).refine_to_tolerance(
        0.1
    ),
    "Manifold.hull": lambda ctx: cancelled_tree(ctx).hull(),
    "Manifold.minkowski_sum": lambda ctx: cancelled_tree(ctx).minkowski_sum(CUBE),
    "Manifold.minkowski_difference": lambda ctx: cancelled_tree(
        ctx
    ).minkowski_difference(CUBE),
    "Manifold.with_precision": lambda ctx: cancelled_tree(ctx).with_precision(1e-3),
    "Manifold.get_precision": lambda ctx: cancelled_tree(ctx).get_precision(),
    "ExecutionContext.from_mesh": lambda ctx: ctx.from_mesh(CUBE.to_mesh64()),
    "ExecutionContext.level_set": lambda ctx: ctx.level_set(sphere_sdf, BOUNDS, 0.5),
}

CLASSES = [Manifold, CrossSection, ExecutionContext, CircularQuality]
# Classes that never touch a Manifold. A new class must go in one list.
EXEMPT_CLASSES = {
    "Mesh",
    "Mesh64",
    "RayHit",
    "Error",
    "JoinType",
    "OpType",
    "ManifoldError",
    "ManifoldCancelled",
}
DUNDERS = {"__init__", "__add__", "__sub__", "__xor__"}
# CircularQuality's constructor and fields never touch a Manifold.
NOT_LISTED = {"CircularQuality.__init__"}


def public_names():
    names = set()
    for cls in CLASSES:
        for n, v in vars(cls).items():
            if isinstance(v, property):
                continue
            if not n.startswith("_") or n in DUNDERS:
                names.add(f"{cls.__name__}.{n}")
    for n, v in vars(m3d).items():
        if n.startswith("_"):
            continue
        if isinstance(v, type):
            assert v in CLASSES or n in EXEMPT_CLASSES, n
        elif callable(v):
            names.add(n)
    return names - NOT_LISTED


def test_every_method_is_categorized():
    assert public_names() == set(CATEGORIES)


def test_every_checked_method_has_a_fixture():
    missing = {
        name
        for name, (cat, fixtures) in CATEGORIES.items()
        if cat != C and not fixtures
    }
    assert missing == CANNOT_FAIL
    assert set(CANCEL) == {n for n, (cat, _) in CATEGORIES.items() if cat == D}


@pytest.mark.parametrize(
    "name", [n for n, (cat, fixtures) in CATEGORIES.items() if cat != C and fixtures]
)
def test_checked_methods_raise(name):
    for code, call in CATEGORIES[name][1]:
        with pytest.raises(ManifoldError) as e:
            call()
        assert type(e.value) is ManifoldError
        assert e.value.code == code


@pytest.mark.parametrize(
    "name", [n for n, (cat, fixtures) in CATEGORIES.items() if cat == C and fixtures]
)
def test_lazy_methods_keep_the_error(name):
    for call in CATEGORIES[name][1]:
        result = call()
        status = result if isinstance(result, Error) else result.status()
        assert status == NFV


@pytest.mark.parametrize("name", sorted(CANCEL))
def test_cancel_raises(name):
    ctx = ExecutionContext()
    ctx.cancel()
    with pytest.raises(ManifoldCancelled) as e:
        CANCEL[name](ctx)
    assert e.value.code == Error.Cancelled


def test_cancelled_status_does_not_raise():
    ctx = ExecutionContext()
    ctx.cancel()
    assert cancelled_tree(ctx).status() == Error.Cancelled


def test_empty_input_does_not_raise():
    assert Manifold().is_empty()
    assert len(Manifold().to_mesh().tri_verts) == 0
    assert CrossSection().extrude(1).is_empty()
    assert CrossSection().revolve().is_empty()
    assert Manifold.extrude(CrossSection(), 1).is_empty()
    assert Manifold.hull_points([(0, 0, 0), (1, 0, 0), (0, 1, 0)]).is_empty()
    assert Manifold.batch_hull([]).is_empty()
    assert Manifold().trim_by_plane((0, 0, 1), 0).is_empty()
    assert (CUBE ^ CUBE.translate((5, 0, 0))).is_empty()
    empty = list(Manifold().bounding_box())
    assert Manifold.level_set(sphere_sdf, empty, 0.5).is_empty()
    assert ExecutionContext().level_set(sphere_sdf, empty, 0.5).is_empty()
