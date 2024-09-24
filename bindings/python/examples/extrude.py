from manifold3d import CrossSection, Manifold


def run():
    # create a polygon
    polygon_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    polygons_points = [polygon_points]

    # create a cross-section
    cross_section = CrossSection(polygons_points)
    polygons = cross_section.to_polygons()
    polygon = polygons[0]
    if set([tuple(p) for p in polygon]) != set(polygon_points):
        raise Exception(
            f"polygon={polygon} differs from polygon_points={polygon_points}"
        )

    # extrude a polygon to create a manifold
    extruded_polygon = Manifold.extrude(cross_section, 10.0)
    eps = 0.001
    observed_volume = extruded_polygon.volume()
    expected_volume = 10.0
    if abs(observed_volume - expected_volume) > eps:
        raise Exception(
            f"observed_volume={observed_volume} differs from expected_volume={expected_volume}"
        )
    observed_surface_area = extruded_polygon.surface_area()
    expected_surface_area = 42.0
    if abs(observed_surface_area - expected_surface_area) > eps:
        raise Exception(
            f"observed_surface_area={observed_surface_area} differs from expected_surface_area={expected_surface_area}"
        )

    # get bounding box from manifold
    observed_bbox = extruded_polygon.bounding_box()
    expected_bbox = (0.0, 0.0, 0.0, 1.0, 1.0, 10.0)
    if observed_bbox != expected_bbox:
        raise Exception(
            f"observed_bbox={observed_bbox} differs from expected_bbox={expected_bbox}"
        )

    return extruded_polygon


if __name__ == "__main__":
    run()
