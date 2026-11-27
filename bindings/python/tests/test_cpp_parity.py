"""Mesh run transforms and project() behave as they do in C++."""

import numpy as np
import pytest

from manifold3d import Manifold, Mesh, Mesh64

EXPORTS = [(Mesh, "to_mesh"), (Mesh64, "to_mesh64")]


def run_positions(mesh, run):
    start, end = mesh.run_index[run] // 3, mesh.run_index[run + 1] // 3
    return mesh.vert_properties[mesh.tri_verts[start:end].ravel(), :3]


def assert_on_vertices(points, vertices, tol):
    distance = np.linalg.norm(points[:, None, :] - vertices[None, :, :], axis=2)
    assert distance.min(axis=1).max() < tol


@pytest.mark.parametrize("mesh_type,export", EXPORTS)
def test_run_transform_reproduces_each_run(mesh_type, export):
    original = Manifold.tetrahedron().scale((1, 2, 3)).as_original()
    moved = original.rotate((10, 20, 30)).translate((5, -1, 2))
    other = original.rotate((0, 45, 0)).translate((-5, 3, 1))
    mesh = getattr(moved + other, export)()
    assert mesh.run_transform.shape == (2, 3, 4)
    for run in range(2):
        transform = mesh.run_transform[run]
        np.testing.assert_array_equal(transform, mesh.get_run_transform(run))
        expected = getattr(original.transform(transform), export)()
        assert_on_vertices(
            run_positions(mesh, run), expected.vert_properties[:, :3], 1e-5
        )


@pytest.mark.parametrize("mesh_type,export", EXPORTS)
def test_run_transform_input_uses_the_output_layout(mesh_type, export):
    source = getattr(Manifold.cube(), export)()
    transform = np.arange(24, dtype=source.vert_properties.dtype).reshape(2, 3, 4)
    size = source.tri_verts.size
    mesh = mesh_type(
        source.vert_properties,
        source.tri_verts,
        run_index=np.array([0, size // 2, size], dtype=source.tri_verts.dtype),
        run_original_id=np.array([1, 2], dtype=np.uint32),
        run_transform=transform,
    )
    np.testing.assert_array_equal(mesh.run_transform, transform)
    for run in range(2):
        np.testing.assert_array_equal(mesh.get_run_transform(run), transform[run])
    copy = mesh_type(
        mesh.vert_properties,
        mesh.tri_verts,
        run_index=np.array(mesh.run_index, dtype=source.tri_verts.dtype),
        run_original_id=np.array(mesh.run_original_id, dtype=np.uint32),
        run_transform=mesh.run_transform,
    )
    np.testing.assert_array_equal(copy.run_transform, transform)


@pytest.mark.parametrize("mesh_type,export", EXPORTS)
def test_unmodified_run_has_identity_transform(mesh_type, export):
    cube = Manifold.cube((1, 2, 3))
    mesh = getattr(cube, export)()
    transform = mesh.get_run_transform(0)
    np.testing.assert_array_equal(transform, np.eye(3, 4))
    applied = getattr(cube.transform(transform), export)()
    np.testing.assert_array_equal(applied.vert_properties, mesh.vert_properties)


@pytest.mark.parametrize("mesh_type,export", EXPORTS)
@pytest.mark.parametrize("run", [1, 2**64 // 12, 2**64 - 1])
def test_out_of_range_run_has_identity_transform(mesh_type, export, run):
    mesh = getattr(Manifold.cube().as_original().translate((1, 2, 3)), export)()
    assert mesh.run_transform.shape == (1, 3, 4)
    np.testing.assert_array_equal(mesh.get_run_transform(run), np.eye(3, 4))


@pytest.mark.parametrize("n", [2, 4])
def test_project_keeps_outline_vertices(n):
    # Each bottom edge of the refined cube is split into n collinear segments.
    section = Manifold.cube().refine(n).project()
    assert section.num_vert() == 4 * n
    assert section.area() == pytest.approx(1)
