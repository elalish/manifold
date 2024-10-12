#include "manifold/manifoldc.h"

#include "gtest/gtest.h"
#include "manifold/manifold.h"
#include "manifold/polygon.h"
#include "manifold/types.h"

TEST(CBIND, sphere) {
  int n = 25;
  ManifoldManifold *sphere =
      manifold_sphere(manifold_alloc_manifold(), 1.0, 4 * n);

  EXPECT_EQ(manifold_status(sphere), MANIFOLD_NO_ERROR);
  EXPECT_EQ(manifold_num_tri(sphere), n * n * 8);

  manifold_delete_manifold(sphere);
}

TEST(CBIND, warp_translation) {
  ManifoldVec3 (*warp)(double, double, double, void *) = [](double x, double y,
                                                            double z, void *) {
    ManifoldVec3 v = {x + 15.0, y, z};
    return v;
  };
  double *context = (double *)malloc(1 * sizeof(double));
  context[0] = 15.0;
  ManifoldVec3 (*warpcontext)(double, double, double, void *) =
      [](double x, double y, double z, void *ctx) {
        ManifoldVec3 v = {x + ((double *)ctx)[0], y, z};
        return v;
      };
  ManifoldManifold *sphere =
      manifold_sphere(manifold_alloc_manifold(), 1.0, 100);
  ManifoldManifold *trans =
      manifold_translate(manifold_alloc_manifold(), sphere, 15., 0., 0.);
  ManifoldManifold *warped =
      manifold_warp(manifold_alloc_manifold(), sphere, warp, NULL);
  ManifoldManifold *diff =
      manifold_difference(manifold_alloc_manifold(), trans, warped);
  ManifoldManifold *warpedcontext =
      manifold_warp(manifold_alloc_manifold(), sphere, warpcontext, context);
  ManifoldManifold *diffcontext =
      manifold_difference(manifold_alloc_manifold(), trans, warped);

  ManifoldProperties props = manifold_get_properties(diff);
  ManifoldProperties propscontext = manifold_get_properties(diffcontext);

  EXPECT_NEAR(props.volume, 0, 0.0001);
  EXPECT_NEAR(propscontext.volume, 0, 0.0001);

  ManifoldBox *sphere_bounds =
      manifold_bounding_box(manifold_alloc_box(), sphere);
  ManifoldBox *trans_bounds =
      manifold_bounding_box(manifold_alloc_box(), trans);
  ManifoldBox *warped_bounds =
      manifold_bounding_box(manifold_alloc_box(), warped);
  ManifoldBox *warped_context_bounds =
      manifold_bounding_box(manifold_alloc_box(), warpedcontext);

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
  // can't convert lambda with captures to funptr
  double (*sdf)(double, double, double, void *) = [](double x, double y,
                                                     double z, void *ctx) {
    const double radius = 15;
    const double xscale = 3;
    const double yscale = 1;
    const double zscale = 1;
    double xs = x / xscale;
    double ys = y / yscale;
    double zs = z / zscale;
    return radius - sqrtf(xs * xs + ys * ys + zs * zs);
  };
  double *context = (double *)malloc(4 * sizeof(double));
  context[0] = 15.0;
  context[1] = 3.0;
  context[2] = 1.0;
  context[3] = 1.0;
  double (*sdfcontext)(double, double, double,
                       void *) = [](double x, double y, double z, void *ctx) {
    double *context = (double *)ctx;
    const double radius = context[0];
    const double xscale = context[1];
    const double yscale = context[2];
    const double zscale = context[3];
    double xs = x / xscale;
    double ys = y / yscale;
    double zs = z / zscale;
    return radius - sqrtf(xs * xs + ys * ys + zs * zs);
  };

  const double bb = 30;  // (radius * 2)
  // bounding box scaled according to factors used in *sdf
  ManifoldBox *bounds = manifold_box(manifold_alloc_box(), -bb * 3, -bb * 1,
                                     -bb * 1, bb * 3, bb * 1, bb * 1);
  ManifoldManifold *sdf_man = manifold_level_set(manifold_alloc_manifold(), sdf,
                                                 bounds, 0.5, 0, -1, NULL);
  ManifoldManifold *sdf_man_context = manifold_level_set(
      manifold_alloc_manifold(), sdfcontext, bounds, 0.5, 0, -1, context);
  ManifoldMeshGL *sdf_mesh =
      manifold_get_meshgl(manifold_alloc_meshgl(), sdf_man);

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
  double a = context[0] * context[1];
  double b = context[0] * context[2];
  double c = context[0] * context[3];
  constexpr double kPi = 3.14159265358979323846264338327950288;
  double s = 4.0 * kPi *
             std::pow(((std::pow(a * b, 1.6) + std::pow(a * c, 1.6) +
                        std::pow(b * c, 1.6)) /
                       3.0),
                      1.0 / 1.6);
  double v = 4.0 * kPi / 3.0 * a * b * c;

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
  manifold_delete_manifold(sdf_man_context);
  manifold_delete_box(bounds);
  free(context);
}

TEST(CBIND, properties) {
  void (*props)(double *, ManifoldVec3, const double *,
                void *) = [](double *new_prop, ManifoldVec3 position,
                             const double *old_prop, void *ctx) {
    new_prop[0] =
        std::sqrt(std::sqrt(position.x * position.x + position.y * position.y) +
                  position.z * position.z) *
        5.0;
  };
  double *context = (double *)malloc(1 * sizeof(double));
  context[0] = 5.0;
  void (*propscontext)(double *, ManifoldVec3, const double *,
                       void *) = [](double *new_prop, ManifoldVec3 position,
                                    const double *old_prop, void *ctx) {
    new_prop[0] =
        std::sqrt(std::sqrt(position.x * position.x + position.y * position.y) +
                  position.z * position.z) *
        ((double *)ctx)[0];
  };

  ManifoldManifold *cube =
      manifold_cube(manifold_alloc_manifold(), 1.0, 1.0, 1.0, 1);
  ManifoldManifold *cube_props =
      manifold_set_properties(manifold_alloc_manifold(), cube, 1, props, NULL);
  ManifoldManifold *cube_props_context = manifold_set_properties(
      manifold_alloc_manifold(), cube, 1, propscontext, context);

  manifold_delete_manifold(cube);
  manifold_delete_manifold(cube_props);
  manifold_delete_manifold(cube_props_context);
  free(context);
}

TEST(CBIND, extrude) {
  ManifoldVec2 pts[] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  ManifoldSimplePolygon *sq[] = {
      manifold_simple_polygon(manifold_alloc_simple_polygon(), &pts[0], 4)};
  ManifoldPolygons *polys = manifold_polygons(manifold_alloc_polygons(), sq, 1);

  ManifoldManifold *cube =
      manifold_cube(manifold_alloc_manifold(), 1., 1., 1., 0);
  ManifoldManifold *extrusion =
      manifold_extrude(manifold_alloc_manifold(), polys, 1, 0, 0, 1, 1);

  ManifoldManifold *diff =
      manifold_difference(manifold_alloc_manifold(), cube, extrusion);
  ManifoldProperties props = manifold_get_properties(diff);

  EXPECT_TRUE(props.volume < 0.0001);

  manifold_delete_manifold(cube);
  manifold_delete_manifold(extrusion);
  manifold_delete_manifold(diff);
  manifold_delete_simple_polygon(sq[0]);
  manifold_delete_polygons(polys);
}

TEST(CBIND, compose_decompose) {
  ManifoldManifold *s1 = manifold_sphere(manifold_alloc_manifold(), 1.0, 100);
  ManifoldManifold *s2 =
      manifold_translate(manifold_alloc_manifold(), s1, 2., 2., 2.);
  ManifoldManifoldVec *ss =
      manifold_manifold_vec(manifold_alloc_manifold_vec(), 2);
  manifold_manifold_vec_set(ss, 0, s1);
  manifold_manifold_vec_set(ss, 1, s2);
  ManifoldManifold *composed = manifold_compose(manifold_alloc_manifold(), ss);

  ManifoldManifoldVec *decomposed =
      manifold_decompose(manifold_alloc_manifold_vec(), composed);

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
      manifold_simple_polygon(manifold_alloc_simple_polygon(), vs, 3);
  ManifoldSimplePolygon *sps[] = {sp};
  ManifoldPolygons *ps = manifold_polygons(manifold_alloc_polygons(), sps, 1);

  EXPECT_EQ(vs[0].x, manifold_simple_polygon_get_point(sp, 0).x);
  EXPECT_EQ(vs[1].x, manifold_simple_polygon_get_point(sp, 1).x);
  EXPECT_EQ(vs[2].x, manifold_simple_polygon_get_point(sp, 2).x);
  EXPECT_EQ(vs[0].x, manifold_polygons_get_point(ps, 0, 0).x);
  EXPECT_EQ(vs[1].x, manifold_polygons_get_point(ps, 0, 1).x);
  EXPECT_EQ(vs[2].x, manifold_polygons_get_point(ps, 0, 2).x);

  manifold_delete_simple_polygon(sp);
  manifold_delete_polygons(ps);
}
