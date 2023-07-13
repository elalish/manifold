#include "manifoldc.h"

#include "gtest/gtest.h"
#include "manifold.h"
#include "polygon.h"
#include "sdf.h"
#include "types.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

TEST(CBIND, sphere) {
  int n = 25;
  size_t sz = manifold_manifold_size();
  ManifoldManifold *sphere = manifold_sphere(malloc(sz), 1.0f, 4 * n);

  EXPECT_EQ(manifold_status(sphere), MANIFOLD_NO_ERROR);
  EXPECT_EQ(manifold_num_tri(sphere), n * n * 8);

  manifold_delete_manifold(sphere);
}

TEST(CBIND, warp_translation) {
  size_t sz = manifold_manifold_size();
  ManifoldVec3 (*warp)(float, float, float) = [](float x, float y, float z) {
    ManifoldVec3 v = {x + 15.0f, y, z};
    return v;
  };
  ManifoldManifold *sphere = manifold_sphere(malloc(sz), 1.0f, 100);
  ManifoldManifold *trans = manifold_translate(malloc(sz), sphere, 15., 0., 0.);
  ManifoldManifold *warped = manifold_warp(malloc(sz), sphere, warp);
  ManifoldManifold *diff = manifold_difference(malloc(sz), trans, warped);

  ManifoldProperties props = manifold_get_properties(diff);

  EXPECT_NEAR(props.volume, 0, 0.0001);

  manifold_delete_manifold(sphere);
  manifold_delete_manifold(trans);
  manifold_delete_manifold(warped);
  manifold_delete_manifold(diff);
}

TEST(CBIND, level_set) {
  size_t sz = manifold_manifold_size();
  // can't convert lambda with captures to funptr
  float (*sdf)(float, float, float) = [](float x, float y, float z) {
    const float radius = 15;
    const float xscale = 3;
    const float yscale = 1;
    const float zscale = 1;
    float xs = x / xscale;
    float ys = y / yscale;
    float zs = z / zscale;
    return radius - sqrtf(xs * xs + ys * ys + zs * zs);
  };

  const float bb = 30;  // (radius * 2)
  // bounding box scaled according to factors used in *sdf
  ManifoldBox *bounds = manifold_box(malloc(manifold_box_size()), -bb * 3,
                                     -bb * 1, -bb * 1, bb * 3, bb * 1, bb * 1);
  ManifoldMeshGL *sdf_mesh =
      manifold_level_set(malloc(manifold_meshgl_size()), sdf, bounds, 0.5, 0);
  ManifoldManifold *sdf_man = manifold_of_meshgl(malloc(sz), sdf_mesh);

#ifdef MANIFOLD_EXPORT
  ManifoldExportOptions *options =
      manifold_export_options(malloc(manifold_export_options_size()));
  const char *name = "cbind_sdf_test.glb";
  manifold_export_meshgl(name, sdf_mesh, options);
  manifold_delete_export_options(options);
#endif

  EXPECT_EQ(manifold_status(sdf_man), MANIFOLD_NO_ERROR);

  manifold_delete_meshgl(sdf_mesh);
  manifold_delete_manifold(sdf_man);
  manifold_delete_box(bounds);
}

TEST(CBIND, extrude) {
  size_t sz = manifold_manifold_size();

  ManifoldVec2 pts[] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  ManifoldSimplePolygon *sq[] = {manifold_simple_polygon(
      malloc(manifold_simple_polygon_size()), &pts[0], 4)};
  ManifoldPolygons *polys =
      manifold_polygons(malloc(manifold_polygons_size()), sq, 1);
  ManifoldCrossSection *cross =
      manifold_cross_section_of_polygons(malloc(manifold_cross_section_size()),
                                         polys, MANIFOLD_FILL_RULE_POSITIVE);

  ManifoldManifold *cube = manifold_cube(malloc(sz), 1., 1., 1., 0);
  ManifoldManifold *extrusion =
      manifold_extrude(malloc(sz), cross, 1, 0, 0, 1, 1);

  ManifoldManifold *diff = manifold_difference(malloc(sz), cube, extrusion);
  ManifoldProperties props = manifold_get_properties(diff);

  EXPECT_TRUE(props.volume < 0.0001);

  manifold_delete_manifold(cube);
  manifold_delete_manifold(extrusion);
  manifold_delete_manifold(diff);
  manifold_delete_simple_polygon(sq[0]);
  manifold_delete_polygons(polys);
}

TEST(CBIND, compose_decompose) {
  size_t sz = manifold_manifold_size();

  ManifoldManifold *s1 = manifold_sphere(malloc(sz), 1.0f, 100);
  ManifoldManifold *s2 = manifold_translate(malloc(sz), s1, 2., 2., 2.);
  ManifoldManifoldVec *ss =
      manifold_manifold_vec(malloc(manifold_manifold_vec_size()), 2);
  manifold_manifold_vec_set(ss, 0, s1);
  manifold_manifold_vec_set(ss, 1, s2);
  ManifoldManifold *composed = manifold_compose(malloc(sz), ss);

  ManifoldManifoldVec *decomposed =
      manifold_decompose(malloc(manifold_manifold_vec_size()), composed);

  EXPECT_EQ(manifold_manifold_vec_length(decomposed), 2);

  manifold_delete_manifold(s1);
  manifold_delete_manifold(s2);
  manifold_delete_manifold_vec(ss);
  manifold_delete_manifold(composed);
  manifold_delete_manifold_vec(decomposed);
}

TEST(CBIND, polygons) {
  ManifoldVec2 vs[] = {{0, 0}, {1, 1}, {2, 2}};
  ManifoldSimplePolygon *sp =
      manifold_simple_polygon(malloc(manifold_simple_polygon_size()), vs, 3);
  ManifoldSimplePolygon *sps[] = {sp};
  ManifoldPolygons *ps =
      manifold_polygons(malloc(manifold_polygons_size()), sps, 1);

  EXPECT_EQ(vs[0].x, manifold_simple_polygon_get_point(sp, 0).x);
  EXPECT_EQ(vs[1].x, manifold_simple_polygon_get_point(sp, 1).x);
  EXPECT_EQ(vs[2].x, manifold_simple_polygon_get_point(sp, 2).x);
  EXPECT_EQ(vs[0].x, manifold_polygons_get_point(ps, 0, 0).x);
  EXPECT_EQ(vs[1].x, manifold_polygons_get_point(ps, 0, 1).x);
  EXPECT_EQ(vs[2].x, manifold_polygons_get_point(ps, 0, 2).x);

  manifold_delete_simple_polygon(sp);
  manifold_delete_polygons(ps);
}
