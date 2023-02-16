#include "manifoldc.h"

#include "gtest/gtest.h"
#include "manifold.h"
#include "polygon.h"
#include "sdf.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

float eps = 0.000001;

bool approx_float_array(float *a, float *b, size_t len) {
  for (int i = 0; i < len; ++i) {
    if (abs(a[i] - b[i]) > eps) {
      return false;
    };
  }
  return true;
}

TEST(CBIND, sphere) {
  int n = 25;
  size_t sz = manifold_manifold_size();
  ManifoldManifold *sphere = manifold_sphere(malloc(sz), 1.0f, 4 * n);

  EXPECT_EQ(manifold_status(sphere), NoError);
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

  ManifoldMeshGL *trans_mesh =
      manifold_get_meshgl(malloc(manifold_meshgl_size()), trans);
  ManifoldMeshGL *warped_mesh =
      manifold_get_meshgl(malloc(manifold_meshgl_size()), warped);

  size_t len_props = manifold_meshgl_vert_properties_length(trans_mesh);
  float *trans_props = manifold_meshgl_vert_properties(
      malloc(sizeof(float) * len_props), trans_mesh);
  float *warped_props = manifold_meshgl_vert_properties(
      malloc(sizeof(float) * len_props), warped_mesh);

  EXPECT_TRUE(approx_float_array(trans_props, warped_props, len_props));

  manifold_delete_manifold(sphere);
  manifold_delete_manifold(trans);
  manifold_delete_manifold(warped);
  manifold_delete_meshgl(trans_mesh);
  manifold_delete_meshgl(warped_mesh);
  delete trans_props;
  delete warped_props;
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
  const char *name = "cbind_sdf_test.stl";
  manifold_export_meshgl(name, sdf_mesh, options);
  manifold_delete_export_options(options);
#endif

  EXPECT_EQ(manifold_status(sdf_man), NoError);

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

  ManifoldManifold *cube = manifold_cube(malloc(sz), 1., 1., 1., 0);
  ManifoldManifold *extrusion =
      manifold_extrude(malloc(sz), polys, 1, 0, 0, 1, 1);

  ManifoldManifold *diff = manifold_difference(malloc(sz), cube, extrusion);
  ManifoldProperties props = manifold_get_properties(diff);

  EXPECT_TRUE(props.volume < eps);

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
  ManifoldManifold *ss[] = {s1, s2};
  ManifoldManifold *composed = manifold_compose(malloc(sz), ss, 2);

  ManifoldComponents *cs =
      manifold_get_components(malloc(manifold_components_size()), composed);
  size_t len = manifold_components_length(cs);
  void *bufs[len];
  for (int i = 0; i < len; ++i) {
    bufs[i] = malloc(sz);
  }
  ManifoldManifold **decomposed = manifold_decompose(bufs, composed, cs);

  EXPECT_EQ(len, 2);

  manifold_delete_manifold(s1);
  manifold_delete_manifold(s2);
  manifold_delete_manifold(composed);
  manifold_delete_components(cs);
  for (int i = 0; i < len; ++i) {
    manifold_delete_manifold(decomposed[i]);
  }
}
