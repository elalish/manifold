#include <manifoldc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "types.h"

ManifoldVec3 warp(float x, float y, float z) {
  ManifoldVec3 res = {0.5 * x + 30, 0.5 * y + 30, z + 30};
  return res;
}

const float radius = 15;
const float xscale = 3;
const float yscale = 1;
const float zscale = 1;
const float bb = radius * 2;
float sdf_ellipsoid(float x, float y, float z) {
  float xs = x / xscale;
  float ys = y / yscale;
  float zs = z / zscale;
  return radius - sqrt(xs * xs + ys * ys + zs * zs);
}

int main(void) {
  size_t sz = manifold_manifold_size();
  ManifoldManifold *s1 = manifold_sphere(malloc(sz), 30., 360);
  ManifoldManifold *s1_trans = manifold_translate(malloc(sz), s1, 15., 0., 0.);
  ManifoldManifold *s2 = manifold_rotate(malloc(sz), s1_trans, 0., 0., -90.);
  ManifoldManifold *diff = manifold_difference(malloc(sz), s1, s2);
  ManifoldManifold *warped = manifold_warp(malloc(sz), diff, warp);
  ManifoldManifold *final = manifold_union(malloc(sz), diff, warped);

  manifold_delete_manifold(s1);
  manifold_delete_manifold(s1_trans);
  manifold_delete_manifold(s2);
  manifold_delete_manifold(diff);
  manifold_delete_manifold(warped);

  ManifoldMesh *mesh = manifold_get_mesh(malloc(manifold_mesh_size()), final);
  ManifoldExportOptions *options =
      manifold_export_options(malloc(manifold_export_options_size()));
  manifold_export_mesh("test.stl", mesh, options);

  // extrude
  ManifoldVec2 pts[] = {{0, 0}, {0, 30}, {30, 30}, {30, 0}};
  ManifoldSimplePolygon *sq[] = {manifold_simple_polygon(
      malloc(manifold_simple_polygon_size()), &pts[0], 4)};
  ManifoldPolygons *polys =
      manifold_polygons(malloc(manifold_polygons_size()), sq, 1);
  ManifoldManifold *pillar =
      manifold_extrude(malloc(sz), polys, 120, 360, 360, 2, 2);
  ManifoldMesh *pillar_mesh =
      manifold_get_mesh(malloc(manifold_mesh_size()), pillar);
  manifold_export_mesh("pillar.stl", pillar_mesh, options);

  // signed distance function
  ManifoldBox *bounds =
      manifold_box(malloc(manifold_box_size()), -bb * xscale, -bb * yscale,
                   -bb * zscale, bb * xscale, bb * yscale, bb * zscale);
  ManifoldMesh *sdf_mesh = manifold_level_set(malloc(manifold_mesh_size()),
                                              sdf_ellipsoid, bounds, 0.5, 0);
  manifold_export_mesh("sdf_test.stl", sdf_mesh, options);

  // getting dynamically sized vector data out
  ManifoldMeshRelation *rel =
      manifold_get_mesh_relation(malloc(manifold_mesh_relation_size()), final);
  size_t bary_len = manifold_mesh_relation_barycentric_length(rel);
  ManifoldVec3 *barycentric = manifold_mesh_relation_barycentric(
      malloc(sizeof(ManifoldVec3) * bary_len), rel);
  size_t print_until = (bary_len > 10 ? 10 : bary_len);
  for (int i = 0; i < print_until; ++i) {
    ManifoldVec3 v = barycentric[i];
    printf("(%f %f %f)\n", v.x, v.y, v.z);
  }
  ManifoldVec3 last = barycentric[bary_len - 1];
  printf("(%f %f %f)\n", last.x, last.y, last.z);

  manifold_delete_mesh(mesh);
  manifold_delete_box(bounds);
  manifold_delete_simple_polygon(sq[0]);
  manifold_delete_polygons(polys);
  manifold_delete_mesh(pillar_mesh);
  manifold_delete_mesh(sdf_mesh);
  manifold_delete_export_options(options);
  manifold_delete_manifold(final);
  manifold_delete_manifold(pillar);
  manifold_delete_mesh_relation(rel);
  free(barycentric);
  return 0;
}
