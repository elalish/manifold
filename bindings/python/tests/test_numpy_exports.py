"""Array ownership and geometry at the vector-to-NumPy boundary."""

import gc

import numpy as np
import pytest

from manifold3d import CrossSection, Manifold, triangulate


@pytest.mark.parametrize("kind", ["polygon", "triangles", "empty_triangles"])
def test_vector_export_owns_writeable_resizable_storage(kind):
    if kind == "polygon":
        array = CrossSection.circle(1, 32).to_polygons()[0]
        dtype, columns = np.float64, 2
    else:
        contours = [] if kind == "empty_triangles" else [[(0, 0), (1, 0), (0, 1)]]
        array = triangulate(contours)
        dtype, columns = np.int32, 3

    assert isinstance(array, np.ndarray)
    assert array.dtype == dtype
    assert array.shape[1] == columns
    assert array.flags.c_contiguous
    assert array.flags.owndata
    assert array.flags.writeable
    original = array.copy()
    old_rows = len(array)
    array.resize((old_rows + 4, columns), refcheck=False)
    np.testing.assert_array_equal(array[:old_rows], original)
    np.testing.assert_array_equal(array[old_rows:], 0)
    array[-1] = 7
    np.testing.assert_array_equal(array[-1], 7)


@pytest.mark.parametrize("geometry", ["section", "slice", "projection"])
def test_polygon_exports_are_independent_of_source_and_each_other(geometry):
    if geometry == "section":
        section = CrossSection.square((4, 4)) - CrossSection.square((2, 2)).translate((1, 1))
    elif geometry == "slice":
        section = Manifold.cube((4, 4, 2)).slice(1)
    else:
        section = Manifold.cube((4, 4, 2)).project()
    first = section.to_polygons()
    expected = [array.copy() for array in first]
    for array in first:
        array[:] = -100
    second = section.to_polygons()
    for old, new, reference in zip(first, second, expected):
        assert not np.shares_memory(old, new)
        np.testing.assert_array_equal(new, reference)

    del section, first
    gc.collect()
    # Reuse native-sized allocations after all geometry and source vectors
    # have gone away; returned arrays and their views must remain valid.
    churn = [np.full(reference.shape, 123.0) for reference in expected for _ in range(16)]
    for array, reference in zip(second, expected):
        view = array[::-1]
        np.testing.assert_array_equal(view, reference[::-1])
        view[0, 0] += 1
        assert array[-1, 0] == reference[-1, 0] + 1
    assert all(np.all(array == 123) for array in churn)


def test_triangulation_export_preserves_indices_and_hole_area():
    contours = [
        np.array([(0, 0), (5, 0), (5, 4), (0, 4)], dtype=np.float64),
        np.array([(1, 1), (1, 2), (3, 2), (3, 1)], dtype=np.float64),
    ]
    vertices = np.concatenate(contours)
    triangles = triangulate(contours)
    del contours
    gc.collect()
    assert triangles.dtype == np.int32
    assert triangles.shape == (8, 3)
    assert np.all((triangles >= 0) & (triangles < len(vertices)))
    points = vertices[triangles]
    u, v = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    areas = (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2
    assert np.all(areas > 0)
    assert areas.sum() == 18


def test_empty_polygon_exports():
    assert CrossSection().to_polygons() == []
    assert Manifold().slice(0).to_polygons() == []
    assert Manifold().project().to_polygons() == []
