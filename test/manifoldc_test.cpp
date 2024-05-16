#include "manifoldc.h"

#include "gtest/gtest.h"
#include "manifold.h"
#include "polygon.h"
#include "sdf.h"
#include "types.h"

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
  ManifoldVec3 (*warp)(float, float, float, void *) = [](float x, float y,
                                                         float z, void *) {
    ManifoldVec3 v = {x + 15.0f, y, z};
    return v;
  };
  float *context = (float *)malloc(1 * sizeof(float));
  context[0] = 15.0f;
  ManifoldVec3 (*warpcontext)(
      float, float, float, void *) = [](float x, float y, float z, void *ctx) {
    ManifoldVec3 v = {x + ((float *)ctx)[0], y, z};
    return v;
  };
  ManifoldManifold *sphere = manifold_sphere(malloc(sz), 1.0f, 100);
  ManifoldManifold *trans = manifold_translate(malloc(sz), sphere, 15., 0., 0.);
  ManifoldManifold *warped = manifold_warp(malloc(sz), sphere, warp, NULL);
  ManifoldManifold *diff = manifold_difference(malloc(sz), trans, warped);
  ManifoldManifold *warpedcontext =
      manifold_warp(malloc(sz), sphere, warpcontext, context);
  ManifoldManifold *diffcontext =
      manifold_difference(malloc(sz), trans, warped);

  ManifoldProperties props = manifold_get_properties(diff);
  ManifoldProperties propscontext = manifold_get_properties(diffcontext);

  EXPECT_NEAR(props.volume, 0, 0.0001);
  EXPECT_NEAR(propscontext.volume, 0, 0.0001);

  ManifoldBox *sphere_bounds =
      manifold_bounding_box(malloc(manifold_box_size()), sphere);
  ManifoldBox *trans_bounds =
      manifold_bounding_box(malloc(manifold_box_size()), trans);
  ManifoldBox *warped_bounds =
      manifold_bounding_box(malloc(manifold_box_size()), warped);
  ManifoldBox *warped_context_bounds =
      manifold_bounding_box(malloc(manifold_box_size()), warpedcontext);

  ManifoldVec3 sphere_dims = manifold_box_dimensions(sphere_bounds);
  ManifoldVec3 trans_dims = manifold_box_dimensions(sphere_bounds);
  ManifoldVec3 warped_dims = manifold_box_dimensions(sphere_bounds);
  ManifoldVec3 warped_context_dims = manifold_box_dimensions(sphere_bounds);

  EXPECT_FLOAT_EQ(trans_dims.x, sphere_dims.x);
  EXPECT_FLOAT_EQ(warped_dims.x, sphere_dims.x);
  EXPECT_FLOAT_EQ(warped_context_dims.x, sphere_dims.x);

  ManifoldVec3 trans_min = manifold_box_min(trans_bounds);
  ManifoldVec3 warped_min = manifold_box_min(warped_bounds);
  ManifoldVec3 warped_context_min = manifold_box_min(warped_context_bounds);

  EXPECT_FLOAT_EQ(warped_min.x, trans_min.x);
  EXPECT_FLOAT_EQ(warped_context_min.x, trans_min.x);

  manifold_delete_box(sphere_bounds);
  manifold_delete_box(trans_bounds);
  manifold_delete_box(warped_bounds);
  manifold_delete_box(warped_context_bounds);
  manifold_delete_manifold(sphere);
  manifold_delete_manifold(trans);
  manifold_delete_manifold(warped);
  manifold_delete_manifold(diff);
  manifold_delete_manifold(warpedcontext);
  manifold_delete_manifold(diffcontext);
  free(context);
}

TEST(CBIND, level_set) {
  size_t sz = manifold_manifold_size();
  // can't convert lambda with captures to funptr
  float (*sdf)(float, float, float, void *) = [](float x, float y, float z,
                                                 void *ctx) {
    const float radius = 15;
    const float xscale = 3;
    const float yscale = 1;
    const float zscale = 1;
    float xs = x / xscale;
    float ys = y / yscale;
    float zs = z / zscale;
    return radius - sqrtf(xs * xs + ys * ys + zs * zs);
  };
  float *context = (float *)malloc(4 * sizeof(float));
  context[0] = 15.0f;
  context[1] = 3.0f;
  context[2] = 1.0f;
  context[3] = 1.0f;
  float (*sdfcontext)(float, float, float, void *) = [](float x, float y,
                                                        float z, void *ctx) {
    float *context = (float *)ctx;
    const float radius = context[0];
    const float xscale = context[1];
    const float yscale = context[2];
    const float zscale = context[3];
    float xs = x / xscale;
    float ys = y / yscale;
    float zs = z / zscale;
    return radius - sqrtf(xs * xs + ys * ys + zs * zs);
  };

  const float bb = 30;  // (radius * 2)
  // bounding box scaled according to factors used in *sdf
  ManifoldBox *bounds = manifold_box(malloc(manifold_box_size()), -bb * 3,
                                     -bb * 1, -bb * 1, bb * 3, bb * 1, bb * 1);
  ManifoldMeshGL *sdf_mesh = manifold_level_set(malloc(manifold_meshgl_size()),
                                                sdf, bounds, 0.5, 0, NULL);
  ManifoldManifold *sdf_man = manifold_of_meshgl(malloc(sz), sdf_mesh);
  ManifoldMeshGL *sdf_mesh_context = manifold_level_set(
      malloc(manifold_meshgl_size()), sdfcontext, bounds, 0.5, 0, context);
  ManifoldManifold *sdf_man_context =
      manifold_of_meshgl(malloc(sz), sdf_mesh_context);

#ifdef MANIFOLD_EXPORT
  ManifoldExportOptions *options =
      manifold_export_options(malloc(manifold_export_options_size()));
  const char *name = "cbind_sdf_test.glb";
  manifold_export_meshgl(name, sdf_mesh, options);
  manifold_delete_export_options(options);
#endif

  EXPECT_EQ(manifold_status(sdf_man), MANIFOLD_NO_ERROR);
  EXPECT_EQ(manifold_status(sdf_man_context), MANIFOLD_NO_ERROR);

  // Analytic calculations for volume and surface area
  float a = context[0] * context[1];
  float b = context[0] * context[2];
  float c = context[0] * context[3];
  float s = 4.0f * glm::pi<float>() *
            std::pow(((std::pow(a * b, 1.6f) + std::pow(a * c, 1.6f) +
                       std::pow(b * c, 1.6f)) /
                      3.0f),
                     1.0f / 1.6f);
  float v = 4.0f * glm::pi<float>() / 3.0f * a * b * c;

  // Numerical calculations for volume and surface area
  ManifoldProperties sdf_props = manifold_get_properties(sdf_man);
  ManifoldProperties sdf_context_props =
      manifold_get_properties(sdf_man_context);

  // Assert that numerical properties are equal to each other and +/- 0.5% of
  // analytical
  EXPECT_FLOAT_EQ(sdf_props.volume, sdf_context_props.volume);
  EXPECT_FLOAT_EQ(sdf_props.surface_area, sdf_context_props.surface_area);
  EXPECT_NEAR(v, sdf_props.volume, 0.005 * v);
  EXPECT_NEAR(s, sdf_props.surface_area, 0.005 * s);

  manifold_delete_meshgl(sdf_mesh);
  manifold_delete_manifold(sdf_man);
  manifold_delete_meshgl(sdf_mesh_context);
  manifold_delete_manifold(sdf_man_context);
  manifold_delete_box(bounds);
  free(context);
}

TEST(CBIND, properties) {
  void (*props)(float *, ManifoldVec3, const float *,
                void *) = [](float *new_prop, ManifoldVec3 position,
                             const float *old_prop, void *ctx) {
    new_prop[0] =
        glm::sqrt(glm::sqrt(position.x * position.x + position.y * position.y) +
                  position.z * position.z) *
        5.0f;
  };
  float *context = (float *)malloc(1 * sizeof(float));
  context[0] = 5.0f;
  void (*propscontext)(float *, ManifoldVec3, const float *,
                       void *) = [](float *new_prop, ManifoldVec3 position,
                                    const float *old_prop, void *ctx) {
    new_prop[0] =
        glm::sqrt(glm::sqrt(position.x * position.x + position.y * position.y) +
                  position.z * position.z) *
        ((float *)ctx)[0];
  };

  ManifoldManifold *cube =
      manifold_cube(malloc(manifold_manifold_size()), 1.0f, 1.0f, 1.0f, 1);
  ManifoldManifold *cube_props = manifold_set_properties(
      malloc(manifold_manifold_size()), cube, 1, props, NULL);
  ManifoldManifold *cube_props_context = manifold_set_properties(
      malloc(manifold_manifold_size()), cube, 1, propscontext, context);

  manifold_delete_manifold(cube);
  manifold_delete_manifold(cube_props);
  manifold_delete_manifold(cube_props_context);
  free(context);
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
