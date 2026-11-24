import numpy as np

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


def test_merge_precision():
    cube = md.Manifold.cube((1, 1, 1)).to_mesh()
    tri_verts = np.asarray(cube.tri_verts).reshape(-1)
    verts = np.asarray(cube.vert_properties)[tri_verts].astype(np.float32)
    verts[0, 0] += 1e-4
    mesh = md.Mesh(
        vert_properties=verts,
        tri_verts=np.arange(len(verts), dtype=np.uint32).reshape(-1, 3),
    )
    assert md.Manifold(mesh.merge()).status() == md.Error.NotManifold
    merged = mesh.merge(tolerance=0, rel_precision=1e-3)
    assert md.Manifold(merged).status() == md.Error.NoError


def test_merge64_precision():
    cube = md.Manifold.cube((1, 1, 1)).to_mesh64()
    tri_verts = np.asarray(cube.tri_verts).reshape(-1)
    verts = np.asarray(cube.vert_properties)[tri_verts].astype(np.float64)
    verts[0, 0] += 1e-9
    mesh = md.Mesh64(
        vert_properties=verts,
        tri_verts=np.arange(len(verts), dtype=np.uint64).reshape(-1, 3),
    )
    assert md.Manifold(mesh.merge()).status() == md.Error.NotManifold
    merged = mesh.merge(tolerance=0, rel_precision=1e-6)
    assert md.Manifold(merged).status() == md.Error.NoError
