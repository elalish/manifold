#include "manifold/manifoldc.h"

#include <cmath>

#include "gtest/gtest.h"
#include "manifold/types.h"

void *alloc_manifold_buffer() { return malloc(manifold_manifold_size()); }

void *alloc_box_buffer() { return malloc(manifold_box_size()); }

void *alloc_meshgl_buffer() { return malloc(manifold_meshgl_size()); }

void *alloc_meshgl64_buffer() { return malloc(manifold_meshgl64_size()); }

void *alloc_simple_polygon_buffer() {
  return malloc(manifold_simple_polygon_size());
}

void *alloc_polygons_buffer() { return malloc(manifold_polygons_size()); }

void *alloc_manifold_vec_buffer() {
  return malloc(manifold_manifold_vec_size());
}

TEST(CBIND, sphere) {
  int n = 25;
  ManifoldManifold *sphere =
      manifold_sphere(alloc_manifold_buffer(), 1.0, 4 * n);

  EXPECT_EQ(manifold_status(sphere), MANIFOLD_NO_ERROR);
  EXPECT_EQ(manifold_num_tri(sphere), n * n * 8);

  manifold_destruct_manifold(sphere);
  free(sphere);
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
  ManifoldManifold *sphere = manifold_sphere(alloc_manifold_buffer(), 1.0, 100);
  ManifoldManifold *trans =
      manifold_translate(alloc_manifold_buffer(), sphere, 15., 0., 0.);
  ManifoldManifold *warped =
      manifold_warp(alloc_manifold_buffer(), sphere, warp, NULL);
  ManifoldManifold *diff =
      manifold_difference(alloc_manifold_buffer(), trans, warped);
  ManifoldManifold *warpedcontext =
      manifold_warp(alloc_manifold_buffer(), sphere, warpcontext, context);
  ManifoldManifold *diffcontext =
      manifold_difference(alloc_manifold_buffer(), trans, warped);

  EXPECT_NEAR(manifold_volume(diff), 0, 0.0001);
  EXPECT_NEAR(manifold_volume(diffcontext), 0, 0.0001);

  ManifoldBox *sphere_bounds =
      manifold_bounding_box(alloc_box_buffer(), sphere);
  ManifoldBox *trans_bounds = manifold_bounding_box(alloc_box_buffer(), trans);
  ManifoldBox *warped_bounds =
      manifold_bounding_box(alloc_box_buffer(), warped);
  ManifoldBox *warped_context_bounds =
      manifold_bounding_box(alloc_box_buffer(), warpedcontext);

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

  manifold_destruct_box(sphere_bounds);
  manifold_destruct_box(trans_bounds);
  manifold_destruct_box(warped_bounds);
  manifold_destruct_box(warped_context_bounds);
  manifold_destruct_manifold(sphere);
  manifold_destruct_manifold(trans);
  manifold_destruct_manifold(warped);
  manifold_destruct_manifold(diff);
  manifold_destruct_manifold(warpedcontext);
  manifold_destruct_manifold(diffcontext);

  free(sphere_bounds);
  free(trans_bounds);
  free(warped_bounds);
  free(warped_context_bounds);
  free(sphere);
  free(trans);
  free(warped);
  free(diff);
  free(warpedcontext);
  free(diffcontext);
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
  ManifoldBox *bounds = manifold_box(alloc_box_buffer(), -bb * 3, -bb * 1,
                                     -bb * 1, bb * 3, bb * 1, bb * 1);
  ManifoldManifold *sdf_man = manifold_level_set(alloc_manifold_buffer(), sdf,
                                                 bounds, 0.5, 0, -1, NULL);
  ManifoldManifold *sdf_man_context = manifold_level_set(
      alloc_manifold_buffer(), sdfcontext, bounds, 0.5, 0, -1, context);
  ManifoldMeshGL *sdf_mesh =
      manifold_get_meshgl(alloc_meshgl_buffer(), sdf_man);

#ifdef MANIFOLD_EXPORT
  ManifoldExportOptions *options =
      manifold_export_options(malloc(manifold_export_options_size()));
  const char *name = "cbind_sdf_test.glb";
  manifold_export_meshgl(name, sdf_mesh, options);
  manifold_destruct_export_options(options);
  free(options);
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

  // Assert that numerical properties are equal to each other and +/- 0.5% of
  // analytical
  EXPECT_FLOAT_EQ(manifold_volume(sdf_man), manifold_volume(sdf_man_context));
  EXPECT_FLOAT_EQ(manifold_surface_area(sdf_man),
                  manifold_surface_area(sdf_man_context));
  EXPECT_NEAR(v, manifold_volume(sdf_man), 0.005 * v);
  EXPECT_NEAR(s, manifold_surface_area(sdf_man), 0.005 * s);

  manifold_destruct_meshgl(sdf_mesh);
  manifold_destruct_manifold(sdf_man);
  manifold_destruct_manifold(sdf_man_context);
  manifold_destruct_box(bounds);
  free(sdf_mesh);
  free(sdf_man);
  free(sdf_man_context);
  free(bounds);
  free(context);
}

TEST(CBIND, level_set_64) {
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
  ManifoldBox *bounds = manifold_box(alloc_box_buffer(), -bb * 3, -bb * 1,
                                     -bb * 1, bb * 3, bb * 1, bb * 1);
  ManifoldManifold *sdf_man = manifold_level_set(alloc_manifold_buffer(), sdf,
                                                 bounds, 0.5, 0, -1, NULL);
  ManifoldManifold *sdf_man_context = manifold_level_set(
      alloc_manifold_buffer(), sdfcontext, bounds, 0.5, 0, -1, context);
  ManifoldMeshGL64 *sdf_mesh =
      manifold_get_meshgl64(alloc_meshgl64_buffer(), sdf_man);

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

  // Assert that numerical properties are equal to each other and +/- 0.5% of
  // analytical
  EXPECT_FLOAT_EQ(manifold_volume(sdf_man), manifold_volume(sdf_man_context));
  EXPECT_FLOAT_EQ(manifold_surface_area(sdf_man),
                  manifold_surface_area(sdf_man_context));
  EXPECT_NEAR(v, manifold_volume(sdf_man), 0.005 * v);
  EXPECT_NEAR(s, manifold_surface_area(sdf_man), 0.005 * s);

  manifold_destruct_meshgl64(sdf_mesh);
  manifold_destruct_manifold(sdf_man);
  manifold_destruct_manifold(sdf_man_context);
  manifold_destruct_box(bounds);
  free(sdf_mesh);
  free(sdf_man);
  free(sdf_man_context);
  free(bounds);
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
      manifold_cube(alloc_manifold_buffer(), 1.0, 1.0, 1.0, 1);
  EXPECT_EQ(manifold_num_prop(cube), 0);

  ManifoldManifold *cube_props =
      manifold_set_properties(alloc_manifold_buffer(), cube, 1, props, NULL);
  EXPECT_EQ(manifold_num_prop(cube_props), 1);

  ManifoldManifold *cube_props_context = manifold_set_properties(
      alloc_manifold_buffer(), cube, 1, propscontext, context);
  EXPECT_EQ(manifold_num_prop(cube_props_context), 1);

  manifold_destruct_manifold(cube);
  manifold_destruct_manifold(cube_props);
  manifold_destruct_manifold(cube_props_context);
  free(cube);
  free(cube_props);
  free(cube_props_context);
  free(context);
}

TEST(CBIND, extrude) {
  ManifoldVec2 pts[] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  ManifoldSimplePolygon *sq[] = {
      manifold_simple_polygon(alloc_simple_polygon_buffer(), &pts[0], 4)};
  ManifoldPolygons *polys = manifold_polygons(alloc_polygons_buffer(), sq, 1);

  ManifoldManifold *cube =
      manifold_cube(alloc_manifold_buffer(), 1., 1., 1., 0);
  ManifoldManifold *extrusion =
      manifold_extrude(alloc_manifold_buffer(), polys, 1, 0, 0, 1, 1);

  ManifoldManifold *diff =
      manifold_difference(alloc_manifold_buffer(), cube, extrusion);

  EXPECT_TRUE(manifold_volume(diff) < 0.0001);

  manifold_destruct_manifold(cube);
  manifold_destruct_manifold(extrusion);
  manifold_destruct_manifold(diff);
  manifold_destruct_simple_polygon(sq[0]);
  manifold_destruct_polygons(polys);

  free(cube);
  free(extrusion);
  free(diff);
  free(sq[0]);
  free(polys);
}

TEST(CBIND, compose_decompose) {
  ManifoldManifold *s1 = manifold_sphere(alloc_manifold_buffer(), 1.0, 100);
  ManifoldManifold *s2 =
      manifold_translate(alloc_manifold_buffer(), s1, 2., 2., 2.);
  ManifoldManifoldVec *ss =
      manifold_manifold_vec(alloc_manifold_vec_buffer(), 2);
  manifold_manifold_vec_set(ss, 0, s1);
  manifold_manifold_vec_set(ss, 1, s2);
  ManifoldManifold *composed = manifold_compose(alloc_manifold_buffer(), ss);

  ManifoldManifoldVec *decomposed =
      manifold_decompose(alloc_manifold_vec_buffer(), composed);

  EXPECT_EQ(manifold_manifold_vec_length(decomposed), 2);

  manifold_destruct_manifold(s1);
  manifold_destruct_manifold(s2);
  manifold_destruct_manifold_vec(ss);
  manifold_destruct_manifold(composed);
  manifold_destruct_manifold_vec(decomposed);
  free(s1);
  free(s2);
  free(ss);
  free(composed);
  free(decomposed);
}

TEST(CBIND, polygons) {
  ManifoldVec2 vs[] = {{0, 0}, {1, 1}, {2, 2}};
  ManifoldSimplePolygon *sp =
      manifold_simple_polygon(alloc_simple_polygon_buffer(), vs, 3);
  ManifoldSimplePolygon *sps[] = {sp};
  ManifoldPolygons *ps = manifold_polygons(alloc_polygons_buffer(), sps, 1);

  EXPECT_EQ(vs[0].x, manifold_simple_polygon_get_point(sp, 0).x);
  EXPECT_EQ(vs[1].x, manifold_simple_polygon_get_point(sp, 1).x);
  EXPECT_EQ(vs[2].x, manifold_simple_polygon_get_point(sp, 2).x);
  EXPECT_EQ(vs[0].x, manifold_polygons_get_point(ps, 0, 0).x);
  EXPECT_EQ(vs[1].x, manifold_polygons_get_point(ps, 0, 1).x);
  EXPECT_EQ(vs[2].x, manifold_polygons_get_point(ps, 0, 2).x);

  manifold_destruct_simple_polygon(sp);
  manifold_destruct_polygons(ps);
  free(sp);
  free(ps);
}

TEST(CBIND, triangulation) {
  ManifoldVec2 vs[] = {{0, 0}, {1, 1}, {1, 2}};
  ManifoldSimplePolygon *sp =
      manifold_simple_polygon(manifold_alloc_simple_polygon(), vs, 3);
  ManifoldSimplePolygon *sps[] = {sp};
  ManifoldPolygons *ps = manifold_polygons(manifold_alloc_polygons(), sps, 1);
  ManifoldTriangulation *triangulation =
      manifold_triangulate(manifold_alloc_triangulation(), ps, 1e-6);

  manifold_delete_simple_polygon(sp);
  manifold_delete_polygons(ps);

  size_t num_tri = manifold_triangulation_num_tri(triangulation);
  int *tri_verts = (int *)manifold_triangulation_tri_verts(
      malloc(num_tri * 3 * sizeof(int)), triangulation);

  EXPECT_EQ(num_tri, 1);
  EXPECT_EQ(tri_verts[0], 0);
  EXPECT_EQ(tri_verts[1], 1);
  EXPECT_EQ(tri_verts[2], 2);
  manifold_delete_triangulation(triangulation);
  free(tri_verts);
}
