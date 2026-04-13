#include "manifold/manifoldc.h"

#include <cmath>
#include <fstream>

#include "gtest/gtest.h"
#include "manifold/types.h"
#include "test.h"

void* alloc_manifold_buffer() { return malloc(manifold_manifold_size()); }

void* alloc_box_buffer() { return malloc(manifold_box_size()); }

void* alloc_rect_buffer() { return malloc(manifold_rect_size()); }

void* alloc_meshgl_buffer() { return malloc(manifold_meshgl_size()); }

void* alloc_meshgl64_buffer() { return malloc(manifold_meshgl64_size()); }

void* alloc_simple_polygon_buffer() {
  return malloc(manifold_simple_polygon_size());
}

void* alloc_polygons_buffer() { return malloc(manifold_polygons_size()); }

void* alloc_manifold_vec_buffer() {
  return malloc(manifold_manifold_vec_size());
}

TEST(CBIND, sphere) {
  int n = 25;
  ManifoldManifold* sphere =
      manifold_sphere(alloc_manifold_buffer(), 1.0, 4 * n);

  EXPECT_EQ(manifold_status(sphere), MANIFOLD_NO_ERROR);
  EXPECT_EQ(manifold_num_tri(sphere), n * n * 8);

  manifold_destruct_manifold(sphere);
  free(sphere);
}

TEST(CBIND, warp_translation) {
  ManifoldVec3 (*warp)(double, double, double, void*) = [](double x, double y,
                                                           double z, void*) {
    ManifoldVec3 v = {x + 15.0, y, z};
    return v;
  };
  double* context = (double*)malloc(1 * sizeof(double));
  context[0] = 15.0;
  ManifoldVec3 (*warpcontext)(double, double, double, void*) =
      [](double x, double y, double z, void* ctx) {
        ManifoldVec3 v = {x + ((double*)ctx)[0], y, z};
        return v;
      };
  ManifoldManifold* sphere = manifold_sphere(alloc_manifold_buffer(), 1.0, 100);
  ManifoldManifold* trans =
      manifold_translate(alloc_manifold_buffer(), sphere, 15., 0., 0.);
  ManifoldManifold* warped =
      manifold_warp(alloc_manifold_buffer(), sphere, warp, NULL);
  ManifoldManifold* diff =
      manifold_difference(alloc_manifold_buffer(), trans, warped);
  ManifoldManifold* warpedcontext =
      manifold_warp(alloc_manifold_buffer(), sphere, warpcontext, context);
  ManifoldManifold* diffcontext =
      manifold_difference(alloc_manifold_buffer(), trans, warped);

  EXPECT_NEAR(manifold_volume(diff), 0, 0.0001);
  EXPECT_NEAR(manifold_volume(diffcontext), 0, 0.0001);

  ManifoldBox* sphere_bounds =
      manifold_bounding_box(alloc_box_buffer(), sphere);
  ManifoldBox* trans_bounds = manifold_bounding_box(alloc_box_buffer(), trans);
  ManifoldBox* warped_bounds =
      manifold_bounding_box(alloc_box_buffer(), warped);
  ManifoldBox* warped_context_bounds =
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

TEST(CBIND, include_pt_mutates_bounds) {
  ManifoldRect* rect = manifold_rect(alloc_rect_buffer(), 0.0, 0.0, 1.0, 1.0);
  manifold_rect_include_pt(rect, 2.0, 3.0);

  EXPECT_TRUE(manifold_rect_contains_pt(rect, 2.0, 3.0));
  ManifoldVec2 rect_max = manifold_rect_max(rect);
  EXPECT_FLOAT_EQ(rect_max.x, 2.0);
  EXPECT_FLOAT_EQ(rect_max.y, 3.0);

  ManifoldBox* box =
      manifold_box(alloc_box_buffer(), 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
  manifold_box_include_pt(box, 2.0, 3.0, 4.0);

  EXPECT_TRUE(manifold_box_contains_pt(box, 2.0, 3.0, 4.0));
  ManifoldVec3 box_max = manifold_box_max(box);
  EXPECT_FLOAT_EQ(box_max.x, 2.0);
  EXPECT_FLOAT_EQ(box_max.y, 3.0);
  EXPECT_FLOAT_EQ(box_max.z, 4.0);

  manifold_destruct_rect(rect);
  manifold_destruct_box(box);
  free(rect);
  free(box);
}

TEST(CBIND, obj_round_trip) {
  ManifoldManifold* cube =
      manifold_cube(alloc_manifold_buffer(), 1.0, 1.0, 1.0, 1);
  char* buffer = NULL;
  manifold_write_obj(
      cube,
      [](char* tmp, void* arg) {
        size_t len = strlen(tmp);
        char** bufferPtr = (char**)arg;
        *bufferPtr = (char*)malloc(len + 1);
        strncpy(*bufferPtr, tmp, len + 1);
      },
      &buffer);
  EXPECT_NE(buffer, (char*)NULL);
  ManifoldManifold* result = manifold_read_obj(alloc_manifold_buffer(), buffer);
  EXPECT_EQ(manifold_volume(result), 1.0);
  manifold_destruct_manifold(cube);
  manifold_destruct_manifold(result);
  free(cube);
  free(result);
  free(buffer);
}

TEST(CBIND, level_set) {
  // can't convert lambda with captures to funptr
  double (*sdf)(double, double, double, void*) = [](double x, double y,
                                                    double z, void* ctx) {
    const double radius = 15;
    const double xscale = 3;
    const double yscale = 1;
    const double zscale = 1;
    double xs = x / xscale;
    double ys = y / yscale;
    double zs = z / zscale;
    return radius - sqrtf(xs * xs + ys * ys + zs * zs);
  };
  double* context = (double*)malloc(4 * sizeof(double));
  context[0] = 15.0;
  context[1] = 3.0;
  context[2] = 1.0;
  context[3] = 1.0;
  double (*sdfcontext)(double, double, double,
                       void*) = [](double x, double y, double z, void* ctx) {
    double* context = (double*)ctx;
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
  ManifoldBox* bounds = manifold_box(alloc_box_buffer(), -bb * 3, -bb * 1,
                                     -bb * 1, bb * 3, bb * 1, bb * 1);
  ManifoldManifold* sdf_man = manifold_level_set(alloc_manifold_buffer(), sdf,
                                                 bounds, 0.5, 0, -1, NULL);
  ManifoldManifold* sdf_man_context = manifold_level_set(
      alloc_manifold_buffer(), sdfcontext, bounds, 0.5, 0, -1, context);
  ManifoldMeshGL* sdf_mesh =
      manifold_get_meshgl(alloc_meshgl_buffer(), sdf_man);

  if (options.exportModels) {
    manifold_write_obj(
        sdf_man,
        [](char* buffer, void*) {
          std::ofstream of("cbind_sdf_test.obj");
          of << buffer;
          of.close();
        },
        NULL);
  }

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
  double (*sdf)(double, double, double, void*) = [](double x, double y,
                                                    double z, void* ctx) {
    const double radius = 15;
    const double xscale = 3;
    const double yscale = 1;
    const double zscale = 1;
    double xs = x / xscale;
    double ys = y / yscale;
    double zs = z / zscale;
    return radius - sqrtf(xs * xs + ys * ys + zs * zs);
  };
  double* context = (double*)malloc(4 * sizeof(double));
  context[0] = 15.0;
  context[1] = 3.0;
  context[2] = 1.0;
  context[3] = 1.0;
  double (*sdfcontext)(double, double, double,
                       void*) = [](double x, double y, double z, void* ctx) {
    double* context = (double*)ctx;
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
  ManifoldBox* bounds = manifold_box(alloc_box_buffer(), -bb * 3, -bb * 1,
                                     -bb * 1, bb * 3, bb * 1, bb * 1);
  ManifoldManifold* sdf_man = manifold_level_set(alloc_manifold_buffer(), sdf,
                                                 bounds, 0.5, 0, -1, NULL);
  ManifoldManifold* sdf_man_context = manifold_level_set(
      alloc_manifold_buffer(), sdfcontext, bounds, 0.5, 0, -1, context);
  ManifoldMeshGL64* sdf_mesh =
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
  void (*props)(double*, ManifoldVec3, const double*,
                void*) = [](double* new_prop, ManifoldVec3 position,
                            const double* old_prop, void* ctx) {
    new_prop[0] =
        std::sqrt(std::sqrt(position.x * position.x + position.y * position.y) +
                  position.z * position.z) *
        5.0;
  };
  double* context = (double*)malloc(1 * sizeof(double));
  context[0] = 5.0;
  void (*propscontext)(double*, ManifoldVec3, const double*,
                       void*) = [](double* new_prop, ManifoldVec3 position,
                                   const double* old_prop, void* ctx) {
    new_prop[0] =
        std::sqrt(std::sqrt(position.x * position.x + position.y * position.y) +
                  position.z * position.z) *
        ((double*)ctx)[0];
  };

  ManifoldManifold* cube =
      manifold_cube(alloc_manifold_buffer(), 1.0, 1.0, 1.0, 1);
  EXPECT_EQ(manifold_num_prop(cube), 0);

  ManifoldManifold* cube_props =
      manifold_set_properties(alloc_manifold_buffer(), cube, 1, props, NULL);
  EXPECT_EQ(manifold_num_prop(cube_props), 1);

  ManifoldManifold* cube_props_context = manifold_set_properties(
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
  ManifoldSimplePolygon* sq[] = {
      manifold_simple_polygon(alloc_simple_polygon_buffer(), &pts[0], 4)};
  ManifoldPolygons* polys = manifold_polygons(alloc_polygons_buffer(), sq, 1);

  ManifoldManifold* cube =
      manifold_cube(alloc_manifold_buffer(), 1., 1., 1., 0);
  ManifoldManifold* extrusion =
      manifold_extrude(alloc_manifold_buffer(), polys, 1, 0, 0, 1, 1);

  ManifoldManifold* diff =
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
  ManifoldManifold* s1 = manifold_sphere(alloc_manifold_buffer(), 1.0, 100);
  ManifoldManifold* s2 =
      manifold_translate(alloc_manifold_buffer(), s1, 2., 2., 2.);
  ManifoldManifoldVec* ss =
      manifold_manifold_vec(alloc_manifold_vec_buffer(), 2);
  manifold_manifold_vec_set(ss, 0, s1);
  manifold_manifold_vec_set(ss, 1, s2);
  ManifoldManifold* composed = manifold_compose(alloc_manifold_buffer(), ss);

  ManifoldManifoldVec* decomposed =
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
  ManifoldSimplePolygon* sp =
      manifold_simple_polygon(alloc_simple_polygon_buffer(), vs, 3);
  ManifoldSimplePolygon* sps[] = {sp};
  ManifoldPolygons* ps = manifold_polygons(alloc_polygons_buffer(), sps, 1);

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
  ManifoldSimplePolygon* sp =
      manifold_simple_polygon(manifold_alloc_simple_polygon(), vs, 3);
  ManifoldSimplePolygon* sps[] = {sp};
  ManifoldPolygons* ps = manifold_polygons(manifold_alloc_polygons(), sps, 1);
  ManifoldTriangulation* triangulation =
      manifold_triangulate(manifold_alloc_triangulation(), ps, 1e-6);

  manifold_delete_simple_polygon(sp);
  manifold_delete_polygons(ps);

  size_t num_tri = manifold_triangulation_num_tri(triangulation);
  int* tri_verts = (int*)manifold_triangulation_tri_verts(
      malloc(num_tri * 3 * sizeof(int)), triangulation);

  EXPECT_EQ(num_tri, 1);
  EXPECT_EQ(tri_verts[0], 0);
  EXPECT_EQ(tri_verts[1], 1);
  EXPECT_EQ(tri_verts[2], 2);
  manifold_delete_triangulation(triangulation);
  free(tri_verts);
}

TEST(CBIND, meshgl_merge_returns_mem) {
  // Create a cube and get its meshgl (which already has merge vectors).
  ManifoldManifold* cube =
      manifold_cube(alloc_manifold_buffer(), 1.0, 1.0, 1.0, 0);
  ManifoldMeshGL* original = manifold_get_meshgl(alloc_meshgl_buffer(), cube);

  // This mesh is already manifold, so Merge() internally returns false.
  // The bug: old code returned the input pointer instead of the output buffer.
  void* mem = alloc_meshgl_buffer();
  ManifoldMeshGL* merged = manifold_meshgl_merge(mem, original);

  // The returned pointer must always come from mem, never from original.
  EXPECT_EQ(reinterpret_cast<void*>(merged), mem);
  EXPECT_NE(merged, original);

  // The merged mesh should still be valid — construct a Manifold from it.
  ManifoldManifold* result =
      manifold_of_meshgl(alloc_manifold_buffer(), merged);
  EXPECT_EQ(manifold_status(result), MANIFOLD_NO_ERROR);
  EXPECT_NEAR(manifold_volume(result), 1.0, 0.0001);

  manifold_destruct_manifold(result);
  manifold_destruct_meshgl(merged);
  manifold_destruct_meshgl(original);
  manifold_destruct_manifold(cube);
  free(result);
  free(mem);
  free(original);
  free(cube);
}

TEST(CBIND, meshgl64_merge_returns_mem) {
  ManifoldManifold* cube =
      manifold_cube(alloc_manifold_buffer(), 1.0, 1.0, 1.0, 0);
  ManifoldMeshGL64* original =
      manifold_get_meshgl64(alloc_meshgl64_buffer(), cube);

  void* mem = alloc_meshgl64_buffer();
  ManifoldMeshGL64* merged = manifold_meshgl64_merge(mem, original);

  EXPECT_EQ(reinterpret_cast<void*>(merged), mem);
  EXPECT_NE(merged, original);

  ManifoldManifold* result =
      manifold_of_meshgl64(alloc_manifold_buffer(), merged);
  EXPECT_EQ(manifold_status(result), MANIFOLD_NO_ERROR);
  EXPECT_NEAR(manifold_volume(result), 1.0, 0.0001);

  manifold_destruct_manifold(result);
  manifold_destruct_meshgl64(merged);
  manifold_destruct_meshgl64(original);
  manifold_destruct_manifold(cube);
  free(result);
  free(mem);
  free(original);
  free(cube);
}

TEST(CBIND, ray_cast) {
  ManifoldManifold* cube =
      manifold_cube(alloc_manifold_buffer(), 1.0, 1.0, 1.0, 1);

  void* mem = malloc(manifold_ray_hit_vec_size());
  ManifoldRayHitVec* hits =
      manifold_ray_cast(mem, cube, 0.0, 0.0, -5.0, 0.0, 0.0, 5.0);
  ASSERT_EQ(manifold_ray_hit_vec_length(hits), 2);
  ManifoldRayHit h0 = manifold_ray_hit_vec_get(hits, 0);
  ManifoldRayHit h1 = manifold_ray_hit_vec_get(hits, 1);
  EXPECT_FLOAT_EQ(h0.position.z, -0.5);
  EXPECT_FLOAT_EQ(h0.normal.z, -1.0);
  EXPECT_FLOAT_EQ(h1.position.z, 0.5);
  EXPECT_FLOAT_EQ(h1.normal.z, 1.0);
  manifold_destruct_ray_hit_vec(hits);
  free(mem);

  void* mem2 = malloc(manifold_ray_hit_vec_size());
  ManifoldRayHitVec* miss =
      manifold_ray_cast(mem2, cube, 10.0, 10.0, -5.0, 10.0, 10.0, 5.0);
  EXPECT_EQ(manifold_ray_hit_vec_length(miss), 0);
  manifold_destruct_ray_hit_vec(miss);
  free(mem2);

  manifold_destruct_manifold(cube);
  free(cube);
}

TEST(CBIND, tolerance) {
  ManifoldManifold* sphere = manifold_sphere(alloc_manifold_buffer(), 1.0, 100);

  // GetTolerance should return a non-negative value.
  double tol = manifold_get_tolerance(sphere);
  EXPECT_GE(tol, 0.0);

  // SetTolerance should be reflected by GetTolerance.
  ManifoldManifold* with_tol =
      manifold_set_tolerance(alloc_manifold_buffer(), sphere, 0.5);
  EXPECT_EQ(manifold_get_tolerance(with_tol), 0.5);

  // Simplify should produce a valid manifold with fewer or equal triangles.
  ManifoldManifold* simplified =
      manifold_simplify(alloc_manifold_buffer(), sphere, 0.1);
  EXPECT_EQ(manifold_status(simplified), MANIFOLD_NO_ERROR);
  EXPECT_LE(manifold_num_tri(simplified), manifold_num_tri(sphere));

  manifold_destruct_manifold(sphere);
  manifold_destruct_manifold(with_tol);
  manifold_destruct_manifold(simplified);
  free(sphere);
  free(with_tol);
  free(simplified);
}

TEST(CBIND, num_prop_vert) {
  ManifoldManifold* cube =
      manifold_cube(alloc_manifold_buffer(), 1.0, 1.0, 1.0, 0);

  // A cube has 8 geometric vertices but more property vertices (due to
  // duplicated normals at sharp edges).
  EXPECT_EQ(manifold_num_vert(cube), 8);
  EXPECT_GE(manifold_num_prop_vert(cube), manifold_num_vert(cube));

  manifold_destruct_manifold(cube);
  free(cube);
}

TEST(CBIND, meshgl_run_accessors) {
  // Create two shapes with original IDs so the boolean result has 2 runs.
  ManifoldManifold* cube_tmp =
      manifold_cube(alloc_manifold_buffer(), 1, 1, 1, 0);
  ManifoldManifold* cube =
      manifold_as_original(alloc_manifold_buffer(), cube_tmp);
  ManifoldManifold* sphere_tmp =
      manifold_sphere(alloc_manifold_buffer(), 0.6, 32);
  ManifoldManifold* sphere_trans =
      manifold_translate(alloc_manifold_buffer(), sphere_tmp, 0.5, 0.5, 0.5);
  ManifoldManifold* sphere =
      manifold_as_original(alloc_manifold_buffer(), sphere_trans);
  ManifoldManifold* result =
      manifold_union(alloc_manifold_buffer(), cube, sphere);
  EXPECT_EQ(manifold_status(result), MANIFOLD_NO_ERROR);

  // MeshGL
  ManifoldMeshGL* mesh = manifold_get_meshgl(alloc_meshgl_buffer(), result);
  EXPECT_GE(manifold_meshgl_tolerance(mesh), 0.0f);
  EXPECT_EQ(manifold_meshgl_num_run(mesh), 2);
  EXPECT_EQ(manifold_meshgl_run_flags_length(mesh),
            manifold_meshgl_num_run(mesh));

  size_t flags_len = manifold_meshgl_run_flags_length(mesh);
  uint8_t* flags = (uint8_t*)manifold_meshgl_run_flags(malloc(flags_len), mesh);
  // Just verify we got data without crashing.
  EXPECT_NE(flags, (uint8_t*)NULL);
  free(flags);

  // MeshGL64
  ManifoldMeshGL64* mesh64 =
      manifold_get_meshgl64(alloc_meshgl64_buffer(), result);
  EXPECT_GE(manifold_meshgl64_tolerance(mesh64), 0.0);
  EXPECT_EQ(manifold_meshgl64_num_run(mesh64), 2);
  EXPECT_EQ(manifold_meshgl64_run_flags_length(mesh64),
            manifold_meshgl64_num_run(mesh64));

  size_t flags64_len = manifold_meshgl64_run_flags_length(mesh64);
  uint8_t* flags64 =
      (uint8_t*)manifold_meshgl64_run_flags(malloc(flags64_len), mesh64);
  EXPECT_NE(flags64, (uint8_t*)NULL);
  free(flags64);

  manifold_destruct_meshgl(mesh);
  manifold_destruct_meshgl64(mesh64);
  manifold_destruct_manifold(result);
  manifold_destruct_manifold(sphere);
  manifold_destruct_manifold(sphere_trans);
  manifold_destruct_manifold(sphere_tmp);
  manifold_destruct_manifold(cube);
  manifold_destruct_manifold(cube_tmp);
  free(mesh);
  free(mesh64);
  free(result);
  free(sphere);
  free(sphere_trans);
  free(sphere_tmp);
  free(cube);
  free(cube_tmp);
}

TEST(CBIND, meshgl_update_normals) {
  // Get a mesh with normals, then verify UpdateNormals doesn't corrupt it.
  ManifoldManifold* cube =
      manifold_cube(alloc_manifold_buffer(), 1.0, 1.0, 1.0, 0);
  ManifoldManifold* with_normals =
      manifold_calculate_normals(alloc_manifold_buffer(), cube, 3, 60.0);
  EXPECT_EQ(manifold_status(with_normals), MANIFOLD_NO_ERROR);

  // MeshGL with normals at channel 3
  ManifoldMeshGL* mesh =
      manifold_get_meshgl_w_normals(alloc_meshgl_buffer(), with_normals, 3);
  size_t tri_before = manifold_meshgl_num_tri(mesh);
  manifold_meshgl_update_normals(mesh, 3);
  EXPECT_EQ(manifold_meshgl_num_tri(mesh), tri_before);

  // Verify the mesh is still valid by constructing a Manifold from it.
  ManifoldManifold* rebuilt = manifold_of_meshgl(alloc_manifold_buffer(), mesh);
  EXPECT_EQ(manifold_status(rebuilt), MANIFOLD_NO_ERROR);

  // MeshGL64 same test
  ManifoldMeshGL64* mesh64 =
      manifold_get_meshgl64_w_normals(alloc_meshgl64_buffer(), with_normals, 3);
  manifold_meshgl64_update_normals(mesh64, 3);
  ManifoldManifold* rebuilt64 =
      manifold_of_meshgl64(alloc_manifold_buffer(), mesh64);
  EXPECT_EQ(manifold_status(rebuilt64), MANIFOLD_NO_ERROR);

  manifold_destruct_manifold(rebuilt64);
  manifold_destruct_meshgl64(mesh64);
  manifold_destruct_manifold(rebuilt);
  manifold_destruct_meshgl(mesh);
  manifold_destruct_manifold(with_normals);
  manifold_destruct_manifold(cube);
  free(rebuilt64);
  free(mesh64);
  free(rebuilt);
  free(mesh);
  free(with_normals);
  free(cube);
}
