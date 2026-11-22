import numpy as np

import manifold3d as md


def test_precision():
    mesh = md.Manifold.sphere(1, 64).to_mesh()
    assert md.Manifold(mesh).get_precision() == 1e-12
    coarse = md.Manifold(mesh, rel_precision=1e-3)
    assert coarse.get_precision() == 1e-3
    assert coarse.translate((1, 0, 0)).get_precision() == 1e-3

    cube = md.Manifold.cube((1, 1, 1))
    stamped = cube.with_precision(1e-2)
    assert stamped.get_precision() == 1e-2
    assert stamped.num_tri() == cube.num_tri()
    assert cube.get_precision() == 1e-12
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
