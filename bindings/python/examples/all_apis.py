from manifold3d import *
import numpy as np


def all_root_level():
    set_min_circular_angle(10)
    set_min_circular_edge_length(1)
    set_circular_segments(22)
    n = get_circular_segments(1)
    assert n == 22
    poly = [[0, 0], [1, 0], [1, 1]]
    tris = triangulate([poly])
    tris = triangulate([np.array(poly)])


def all_cross_section():
    poly = [[0, 0], [1, 0], [1, 1]]
    c = CrossSection([np.array(poly)])
    c = CrossSection([poly])
    c = CrossSection() + c
    a = c.area()
    c = CrossSection.batch_boolean(
        [
            CrossSection.circle(1),
            CrossSection.square((3, 3)),
            CrossSection.circle(3).translate((1, 1)),
        ],
        OpType.Add,
    )
    c = CrossSection.batch_hull([c, c.translate((1, 0))])
    b = c.bounds()
    c = CrossSection.circle(1)
    c = CrossSection.compose([c, c.translate((1, 0))])
    cs = c.decompose()
    m = c.extrude(1)
    c = c.hull()
    c = CrossSection.hull_points(poly)
    c = CrossSection.hull_points(np.array(poly))
    e = c.is_empty()
    c = c.mirror((0, 1))
    n = c.num_contour()
    n = c.num_vert()
    c = c.offset(1, JoinType.Round)
    m = c.revolve()
    c = c.rotate(90)
    c = c.scale((2, 2))
    c = c.simplify()
    c = CrossSection.square((1, 1))
    p = c.to_polygons()
    c = c.transform([[1, 0, 0], [0, 1, 0]])
    c = c.translate((1, 1))
    c = c.warp(lambda p: (p[0] + 1, p[1] / 2))
    c = c.warp_batch(lambda ps: ps * [1, 0.5] + [1, 0])


def all_manifold():
    mesh = Manifold.sphere(1).to_mesh()
    m = Manifold(mesh)
    m = Manifold() + m
    m = m.as_original()
    m = Manifold.batch_boolean(
        [
            Manifold.cylinder(4, 1),
            Manifold.cube((3, 2, 1)),
            Manifold.cylinder(5, 3).translate((1, 1, 1)),
        ],
        OpType.Add,
    )
    m = Manifold.batch_hull([m, m.translate((0, 0, 1))])
    b = m.bounding_box()
    m = m.calculate_curvature(4, 5)
    m = m.calculate_normals(0)
    m = m.smooth_by_normals(0)
    m = Manifold.compose([m, m.translate((5, 0, 0))])
    m = Manifold.cube((1, 1, 1))
    m = Manifold.cylinder(1, 1)
    ms = m.decompose()
    m = Manifold.extrude(CrossSection.circle(1), 1)
    m = Manifold.revolve(CrossSection.circle(1))
    g = m.genus()
    a = m.surface_area()
    v = m.volume()
    m = m.hull()
    m = m.hull_points(mesh.vert_properties)
    e = m.is_empty()
    m = m.mirror((0, 0, 1))
    n = m.num_edge()
    n = m.num_prop()
    n = m.num_prop_vert()
    n = m.num_tri()
    n = m.num_vert()
    i = m.original_id()
    p = m.get_tolerance()
    pp = m.set_tolerance(0.0001)
    c = m.project()
    m = m.refine(2)
    m = m.refine_to_length(0.1)
    m = m.refine_to_tolerance(0.01)
    m = m.smooth_out()
    i = Manifold.reserve_ids(1)
    m = m.scale((1, 2, 3))
    m = m.set_properties(3, lambda pos, prop: pos)
    c = m.slice(0.5)
    m = Manifold.smooth(mesh, [0], [0.5])
    m = Manifold.sphere(1)
    m, n = m.split(m.translate((1, 0, 0)))
    m, n = m.split_by_plane((0, 0, 1), 0)
    e = m.status()
    m = Manifold.tetrahedron()
    mesh = m.to_mesh()
    ok = mesh.merge()
    m = m.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    m = m.translate((0, 0, 0))
    m = m.trim_by_plane((0, 0, 1), 0)
    m = m.warp(lambda p: (p[0] + 1, p[1] / 2, p[2] * 2))
    m = m.warp_batch(lambda ps: ps * 2 + [1, 0, 0])
    m = Manifold.cube()
    m2 = Manifold.cube().translate([2, 0, 0])
    d = m.min_gap(m2, 2)
    mesh2 = m.to_mesh64()
    ok = mesh.merge()


def run():
    all_root_level()
    all_cross_section()
    all_manifold()
    return Manifold()


if __name__ == "__main__":
    run()
