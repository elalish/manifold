import manifold3d as md


def test_circular_constraints_restore():
    angle = md.get_min_circular_angle()
    length = md.get_min_circular_edge_length()
    segments = md.get_circular_segments(2.0)
    try:
        md.set_min_circular_angle(22.5)
        md.set_min_circular_edge_length(0.125)
        assert md.get_min_circular_angle() == 22.5
        assert md.get_min_circular_edge_length() == 0.125
    finally:
        md.set_min_circular_angle(angle)
        md.set_min_circular_edge_length(length)
    assert md.get_min_circular_angle() == angle
    assert md.get_min_circular_edge_length() == length
    assert md.get_circular_segments(2.0) == segments
