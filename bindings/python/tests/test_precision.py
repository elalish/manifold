import numpy as np
import pytest

import manifold3d as md


def test_precision():
    float_eps = float(np.finfo(np.float32).eps)
    assert md.MIN_RELATIVE_PRECISION == 1e-12
    mesh = md.Manifold.sphere(1, 64).to_mesh()
    mesh64 = md.Manifold.sphere(1, 64).to_mesh64()
    assert md.Manifold(mesh).get_precision() == float_eps
    assert md.Manifold(mesh64).get_precision() == 1e-12
    ctx = md.ExecutionContext()
    assert ctx.from_mesh(mesh).get_precision() == float_eps
    assert ctx.from_mesh(mesh64).get_precision() == 1e-12

    coarse = md.Manifold(mesh).with_precision(1e-3)
    assert coarse.get_precision() == 1e-3
    assert coarse.translate((1, 0, 0)).get_precision() == 1e-3
    # The mesh does not store the precision.
    assert md.Manifold(coarse.to_mesh()).get_precision() == float_eps
    assert md.Manifold(coarse.to_mesh64()).get_precision() == 1e-12

    cube = md.Manifold.cube((1, 1, 1))
    stamped = cube.with_precision(1e-2)
    assert stamped.get_precision() == 1e-2
    assert stamped.num_tri() == cube.num_tri()
    assert cube.get_precision() == float_eps
    fine = cube.with_precision(md.MIN_RELATIVE_PRECISION)
    assert fine.get_precision() == 1e-12
    assert cube.with_precision(1e-15).get_precision() == 1e-12
    # 0, negative values, NaN and infinity give float epsilon, even for a
    # Manifold made from a Mesh64.
    for unset in (0, -1, float("nan"), float("inf")):
        assert md.Manifold(mesh64).with_precision(unset).get_precision() == float_eps
        assert fine.with_precision(unset).get_precision() == float_eps
    assert (stamped + coarse).get_precision() == 1e-3
    assert (coarse + stamped).get_precision() == 1e-3

    # Built shapes have float epsilon.
    square = md.CrossSection.square((1, 1))
    built = [
        md.Manifold.extrude(square, 1),
        md.Manifold.revolve(square.translate((1, 0))),
        md.Manifold.level_set(
            lambda x, y, z: 1 - (x * x + y * y + z * z) ** 0.5,
            [-2, -2, -2, 2, 2, 2],
            0.5,
        ),
        md.Manifold.hull_points([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
    ]
    for shape in built:
        assert shape.get_precision() == float_eps
    assert "kDefault" not in md.Manifold.get_precision.__doc__
    assert "float epsilon" in md.Manifold.with_precision.__doc__


def test_merge_floor():
    # The merge floor is the ingest precision of the mesh type times the
    # bounding box scale, which is 1 for this cube.
    float_eps = float(np.finfo(np.float32).eps)
    cube = md.Manifold.cube((1, 1, 1)).to_mesh64()
    tri_verts = np.asarray(cube.tri_verts).reshape(-1)
    soup = np.asarray(cube.vert_properties)[tri_verts]
    tris = np.arange(len(soup)).reshape(-1, 3)

    def merged64(offset, tolerance=0):
        verts = soup.copy()
        verts[0, 0] += offset
        mesh = md.Mesh64(vert_properties=verts, tri_verts=tris.astype(np.uint64))
        return mesh.merge(tolerance)

    with pytest.raises(md.ManifoldError) as e:
        md.Manifold(merged64(0.5 * float_eps))
    assert e.value.code == md.Error.NotManifold
    assert not md.Manifold(merged64(0.5 * float_eps, float_eps)).is_empty()
    assert not md.Manifold(merged64(0.5e-12)).is_empty()

    verts = soup.astype(np.float32)
    verts[0, 0] += 1e-4
    mesh = md.Mesh(vert_properties=verts, tri_verts=tris.astype(np.uint32))
    with pytest.raises(md.ManifoldError) as e:
        md.Manifold(mesh.merge())
    assert e.value.code == md.Error.NotManifold
    assert not md.Manifold(mesh.merge(1e-3)).is_empty()


def test_no_precision_argument():
    mesh = md.Manifold.cube().to_mesh()
    with pytest.raises(TypeError):
        md.Manifold(mesh, 1e-3)
    with pytest.raises(TypeError):
        mesh.merge(0, 1e-3)
