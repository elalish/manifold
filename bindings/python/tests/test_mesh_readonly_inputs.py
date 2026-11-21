"""Mesh constructors only read their input buffers and own the copied data."""

import numpy as np
import pytest

from manifold3d import Manifold, Mesh, Mesh64


@pytest.mark.parametrize("mesh_type,export", [(Mesh, "to_mesh"), (Mesh64, "to_mesh64")])
def test_mesh_accepts_its_own_readonly_exports(mesh_type, export):
    source = getattr(Manifold.cube(), export)()
    assert not source.vert_properties.flags.writeable
    assert not source.tri_verts.flags.writeable
    copied = mesh_type(source.vert_properties, source.tri_verts)
    np.testing.assert_array_equal(copied.vert_properties, source.vert_properties)
    np.testing.assert_array_equal(copied.tri_verts, source.tri_verts)
    assert Manifold(copied).volume() == 1


@pytest.mark.parametrize("mesh_type,export", [(Mesh, "to_mesh"), (Mesh64, "to_mesh64")])
@pytest.mark.parametrize("readonly", [False, True])
def test_mesh_copies_all_input_buffers(mesh_type, export, readonly):
    source = getattr(Manifold.cube(), export)()
    float_type = source.vert_properties.dtype
    index_type = source.tri_verts.dtype
    buffers = {
        "vert_properties": source.vert_properties.copy(),
        "tri_verts": source.tri_verts.copy(),
        "merge_from_vert": np.array([1], dtype=index_type),
        "merge_to_vert": np.array([0], dtype=index_type),
        "run_index": np.array([0, source.tri_verts.size], dtype=index_type),
        "run_original_id": np.array([7], dtype=np.uint32),
        "run_transform": np.arange(12, dtype=float_type).reshape(1, 4, 3),
        "run_flags": np.array([1], dtype=np.uint8),
        "face_id": np.arange(len(source.tri_verts), dtype=index_type),
        "halfedge_tangent": np.arange(
            len(source.tri_verts) * 12, dtype=float_type
        ).reshape(-1, 3, 4),
    }
    expected = {name: value.copy() for name, value in buffers.items()}
    for value in buffers.values():
        value.flags.writeable = not readonly
    copied = mesh_type(**buffers)
    for name, value in buffers.items():
        np.testing.assert_array_equal(value, expected[name])
        value.flags.writeable = True
        value[...] = 0
        np.testing.assert_array_equal(getattr(copied, name), expected[name])


@pytest.mark.parametrize("mesh_type,dtype,index", [(Mesh, np.float32, np.uint32), (Mesh64, np.float64, np.uint64)])
def test_readonly_vertices_still_require_three_properties(mesh_type, dtype, index):
    vertices = np.zeros((4, 2), dtype=dtype)
    triangles = np.zeros((0, 3), dtype=index)
    vertices.flags.writeable = False
    triangles.flags.writeable = False
    with pytest.raises(RuntimeError, match="at least 3 columns"):
        mesh_type(vertices, triangles)
