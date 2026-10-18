#include "manifold/manifoldc.h"

#include <cmath>
#ifndef MANIFOLD_NO_IOSTREAM
#include <fstream>
#endif

#include "gtest/gtest.h"
#include "manifold/types.h"
#include "manifold/version.h"
#include "test.h"

TEST(CBIND, version) {
  int version[3];
  manifold_version(version);

  EXPECT_EQ(version[0], MANIFOLD_VERSION_MAJOR);
  EXPECT_EQ(version[1], MANIFOLD_VERSION_MINOR);
  EXPECT_EQ(version[2], MANIFOLD_VERSION_PATCH);
}

TEST(CBIND, sphere) {
  int n = 25;
  ManifoldManifold* sphere = manifold_sphere(1.0, 4 * n);

  EXPECT_EQ(manifold_status(sphere), MANIFOLD_NO_ERROR);
  EXPECT_EQ(manifold_num_tri(sphere), n * n * 8);

  manifold_delete_manifold(sphere);
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
  ManifoldManifold* sphere = manifold_sphere(1.0, 100);
  ManifoldManifold* trans = manifold_translate(sphere, 15., 0., 0.);
  ManifoldManifold* warped = manifold_warp(sphere, warp, NULL);
  ManifoldManifold* diff = manifold_difference(trans, warped);
  ManifoldManifold* warpedcontext = manifold_warp(sphere, warpcontext, context);
  ManifoldManifold* diffcontext = manifold_difference(trans, warped);

  EXPECT_NEAR(manifold_volume(diff), 0, 0.0001);
  EXPECT_NEAR(manifold_volume(diffcontext), 0, 0.0001);

  ManifoldBox* sphere_bounds = manifold_bounding_box(sphere);
  ManifoldBox* trans_bounds = manifold_bounding_box(trans);
  ManifoldBox* warped_bounds = manifold_bounding_box(warped);
  ManifoldBox* warped_context_bounds = manifold_bounding_box(warpedcontext);

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

TEST(CBIND, include_pt_mutates_bounds) {
  ManifoldRect* rect = manifold_rect(0.0, 0.0, 1.0, 1.0);
  manifold_rect_include_pt(rect, 2.0, 3.0);

  EXPECT_TRUE(manifold_rect_contains_pt(rect, 2.0, 3.0));
  ManifoldVec2 rect_max = manifold_rect_max(rect);
  EXPECT_FLOAT_EQ(rect_max.x, 2.0);
  EXPECT_FLOAT_EQ(rect_max.y, 3.0);

  ManifoldBox* box = manifold_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
  manifold_box_include_pt(box, 2.0, 3.0, 4.0);

  EXPECT_TRUE(manifold_box_contains_pt(box, 2.0, 3.0, 4.0));
  ManifoldVec3 box_max = manifold_box_max(box);
  EXPECT_FLOAT_EQ(box_max.x, 2.0);
  EXPECT_FLOAT_EQ(box_max.y, 3.0);
  EXPECT_FLOAT_EQ(box_max.z, 4.0);

  manifold_delete_rect(rect);
  manifold_delete_box(box);
}

#ifndef MANIFOLD_NO_IOSTREAM
TEST(CBIND, obj_round_trip) {
  ManifoldManifold* cube = manifold_cube(1.0, 1.0, 1.0, 1);
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
  ManifoldManifold* result = manifold_read_obj(buffer);
  EXPECT_EQ(manifold_volume(result), 1.0);
  manifold_delete_manifold(cube);
  manifold_delete_manifold(result);
  free(buffer);
}
#endif

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
  ManifoldBox* bounds =
      manifold_box(-bb * 3, -bb * 1, -bb * 1, bb * 3, bb * 1, bb * 1);
  ManifoldManifold* sdf_man = manifold_level_set(sdf, bounds, 0.5, 0, -1, NULL);
  ManifoldManifold* sdf_man_context =
      manifold_level_set(sdfcontext, bounds, 0.5, 0, -1, context);
  ManifoldMeshGL* sdf_mesh = manifold_get_meshgl(sdf_man);

#ifndef MANIFOLD_NO_IOSTREAM
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

  manifold_delete_meshgl(sdf_mesh);
  manifold_delete_manifold(sdf_man);
  manifold_delete_manifold(sdf_man_context);
  manifold_delete_box(bounds);
  free(context);
}

TEST(CBIND, execution_context_factories) {
  double (*sdf)(double, double, double, void*) = [](double x, double y,
                                                    double z, void*) {
    return 15.0 - sqrtf(x * x + y * y + z * z);
  };
  const double bb = 30;
  ManifoldBox* bounds = manifold_box(-bb, -bb, -bb, bb, bb, bb);

  // LevelSet under an ExecutionContext: runs to completion, progress hits 1.
  ManifoldExecutionContext* ec = manifold_execution_context();
  ManifoldManifold* sdf_man =
      manifold_execution_context_level_set(ec, sdf, bounds, 0.5, 0, -1, NULL);
  EXPECT_EQ(manifold_status(sdf_man), MANIFOLD_NO_ERROR);
  EXPECT_FALSE(manifold_execution_context_cancelled(ec));
  EXPECT_DOUBLE_EQ(manifold_execution_context_progress(ec), 1.0);

  // FromMeshGL under an ExecutionContext: round-trip a cube's mesh.
  ManifoldManifold* cube = manifold_cube(2.0, 2.0, 2.0, 1);
  ManifoldMeshGL* mesh = manifold_get_meshgl(cube);
  ManifoldExecutionContext* ec2 = manifold_execution_context();
  ManifoldManifold* ingested = manifold_execution_context_of_meshgl(ec2, mesh);
  EXPECT_EQ(manifold_status(ingested), MANIFOLD_NO_ERROR);
  EXPECT_DOUBLE_EQ(manifold_execution_context_progress(ec2), 1.0);

  // A pre-cancelled ctx must propagate through the factory. This is what
  // distinguishes the ctx factory from the plain one: NoError + progress==1
  // alone would also pass if the binding bypassed the ctx (a fresh ctx reads
  // progress 1.0), so assert the cancel actually lands.
  ManifoldExecutionContext* ec3 = manifold_execution_context();
  manifold_execution_context_cancel(ec3);
  ManifoldManifold* cancelled =
      manifold_execution_context_level_set(ec3, sdf, bounds, 0.5, 0, -1, NULL);
  EXPECT_EQ(manifold_status(cancelled), MANIFOLD_CANCELLED);

  manifold_delete_execution_context(ec);
  manifold_delete_execution_context(ec2);
  manifold_delete_execution_context(ec3);
  manifold_delete_manifold(sdf_man);
  manifold_delete_manifold(cube);
  manifold_delete_manifold(ingested);
  manifold_delete_manifold(cancelled);
  manifold_delete_meshgl(mesh);
  manifold_delete_box(bounds);
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
  ManifoldBox* bounds =
      manifold_box(-bb * 3, -bb * 1, -bb * 1, bb * 3, bb * 1, bb * 1);
  ManifoldManifold* sdf_man = manifold_level_set(sdf, bounds, 0.5, 0, -1, NULL);
  ManifoldManifold* sdf_man_context =
      manifold_level_set(sdfcontext, bounds, 0.5, 0, -1, context);
  ManifoldMeshGL64* sdf_mesh = manifold_get_meshgl64(sdf_man);

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

  manifold_delete_meshgl64(sdf_mesh);
  manifold_delete_manifold(sdf_man);
  manifold_delete_manifold(sdf_man_context);
  manifold_delete_box(bounds);
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

  ManifoldManifold* cube = manifold_cube(1.0, 1.0, 1.0, 1);
  EXPECT_EQ(manifold_num_prop(cube), 0);

  ManifoldManifold* cube_props = manifold_set_properties(cube, 1, props, NULL);
  EXPECT_EQ(manifold_num_prop(cube_props), 1);

  ManifoldManifold* cube_props_context =
      manifold_set_properties(cube, 1, propscontext, context);
  EXPECT_EQ(manifold_num_prop(cube_props_context), 1);

  manifold_delete_manifold(cube);
  manifold_delete_manifold(cube_props);
  manifold_delete_manifold(cube_props_context);
  free(context);
}

TEST(CBIND, extrude) {
  ManifoldVec2 pts[] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  ManifoldSimplePolygon* sq[] = {manifold_simple_polygon(&pts[0], 4)};
  ManifoldPolygons* polys = manifold_polygons(sq, 1);

  ManifoldManifold* cube = manifold_cube(1., 1., 1., 0);
  ManifoldManifold* extrusion = manifold_extrude(polys, 1, 0, 0, 1, 1);

  ManifoldManifold* diff = manifold_difference(cube, extrusion);

  EXPECT_TRUE(manifold_volume(diff) < 0.0001);

  manifold_delete_manifold(cube);
  manifold_delete_manifold(extrusion);
  manifold_delete_manifold(diff);
  manifold_delete_simple_polygon(sq[0]);
  manifold_delete_polygons(polys);
}

TEST(CBIND, compose_decompose) {
  ManifoldManifold* s1 = manifold_sphere(1.0, 100);
  ManifoldManifold* s2 = manifold_translate(s1, 2., 2., 2.);
  ManifoldManifoldVec* ss = manifold_manifold_vec(2);
  manifold_manifold_vec_set(ss, 0, s1);
  manifold_manifold_vec_set(ss, 1, s2);
  ManifoldManifold* composed = manifold_compose(ss);

  ManifoldManifoldVec* decomposed = manifold_decompose(composed);

  EXPECT_EQ(manifold_manifold_vec_length(decomposed), 2);

  manifold_delete_manifold(s1);
  manifold_delete_manifold(s2);
  manifold_delete_manifold_vec(ss);
  manifold_delete_manifold(composed);
  manifold_delete_manifold_vec(decomposed);
}

TEST(CBIND, polygons) {
  ManifoldVec2 vs[] = {{0, 0}, {1, 1}, {2, 2}};
  ManifoldSimplePolygon* sp = manifold_simple_polygon(vs, 3);
  ManifoldSimplePolygon* sps[] = {sp};
  ManifoldPolygons* ps = manifold_polygons(sps, 1);

  EXPECT_EQ(vs[0].x, manifold_simple_polygon_get_point(sp, 0).x);
  EXPECT_EQ(vs[1].x, manifold_simple_polygon_get_point(sp, 1).x);
  EXPECT_EQ(vs[2].x, manifold_simple_polygon_get_point(sp, 2).x);
  EXPECT_EQ(vs[0].x, manifold_polygons_get_point(ps, 0, 0).x);
  EXPECT_EQ(vs[1].x, manifold_polygons_get_point(ps, 0, 1).x);
  EXPECT_EQ(vs[2].x, manifold_polygons_get_point(ps, 0, 2).x);

  manifold_delete_simple_polygon(sp);
  manifold_delete_polygons(ps);
}

TEST(CBIND, triangulation) {
  ManifoldVec2 vs[] = {{0, 0}, {1, 1}, {1, 2}};
  ManifoldSimplePolygon* sp = manifold_simple_polygon(vs, 3);
  ManifoldSimplePolygon* sps[] = {sp};
  ManifoldPolygons* ps = manifold_polygons(sps, 1);
  ManifoldTriangulation* triangulation = manifold_triangulate(ps, 1e-6);

  manifold_delete_simple_polygon(sp);
  manifold_delete_polygons(ps);

  size_t num_tri = manifold_triangulation_num_tri(triangulation);
  int* tri_verts = (int*)manifold_triangulation_tri_verts(
      reinterpret_cast<int*>(malloc(num_tri * 3 * sizeof(int))), triangulation);

  EXPECT_EQ(num_tri, 1);
  EXPECT_EQ(tri_verts[0], 0);
  EXPECT_EQ(tri_verts[1], 1);
  EXPECT_EQ(tri_verts[2], 2);
  manifold_delete_triangulation(triangulation);
  free(tri_verts);
}

TEST(CBIND, meshgl_merge_returns_mem) {
  // Create a cube and get its meshgl (which already has merge vectors).
  ManifoldManifold* cube = manifold_cube(1.0, 1.0, 1.0, 0);
  ManifoldMeshGL* original = manifold_get_meshgl(cube);

  // This mesh is already manifold, so Merge() internally returns false.
  // The bug: old code returned the input pointer instead of the output buffer.
  ManifoldMeshGL* merged = manifold_meshgl_merge(original);

  EXPECT_NE(merged, original);

  // The merged mesh should still be valid — construct a Manifold from it.
  ManifoldManifold* result = manifold_of_meshgl(merged);
  EXPECT_EQ(manifold_status(result), MANIFOLD_NO_ERROR);
  EXPECT_NEAR(manifold_volume(result), 1.0, 0.0001);

  manifold_delete_manifold(result);
  manifold_delete_meshgl(merged);
  manifold_delete_meshgl(original);
  manifold_delete_manifold(cube);
}

TEST(CBIND, meshgl64_merge_returns_mem) {
  ManifoldManifold* cube = manifold_cube(1.0, 1.0, 1.0, 0);
  ManifoldMeshGL64* original = manifold_get_meshgl64(cube);

  ManifoldMeshGL64* merged = manifold_meshgl64_merge(original);

  EXPECT_NE(merged, original);

  ManifoldManifold* result = manifold_of_meshgl64(merged);
  EXPECT_EQ(manifold_status(result), MANIFOLD_NO_ERROR);
  EXPECT_NEAR(manifold_volume(result), 1.0, 0.0001);

  manifold_delete_manifold(result);
  manifold_delete_meshgl64(merged);
  manifold_delete_meshgl64(original);
  manifold_delete_manifold(cube);
}

TEST(CBIND, ray_cast) {
  ManifoldManifold* cube = manifold_cube(1.0, 1.0, 1.0, 1);

  ManifoldRayHitVec* hits =
      manifold_ray_cast(cube, 0.0, 0.0, -5.0, 0.0, 0.0, 5.0);
  ASSERT_EQ(manifold_ray_hit_vec_length(hits), 2);
  ManifoldRayHit h0 = manifold_ray_hit_vec_get(hits, 0);
  ManifoldRayHit h1 = manifold_ray_hit_vec_get(hits, 1);
  EXPECT_FLOAT_EQ(h0.position.z, -0.5);
  EXPECT_FLOAT_EQ(h0.normal.z, -1.0);
  EXPECT_FLOAT_EQ(h1.position.z, 0.5);
  EXPECT_FLOAT_EQ(h1.normal.z, 1.0);
  EXPECT_EQ(h0.crossing, 1);
  EXPECT_EQ(h1.crossing, -1);
  manifold_delete_ray_hit_vec(hits);

  ManifoldRayHitVec* miss =
      manifold_ray_cast(cube, 10.0, 10.0, -5.0, 10.0, 10.0, 5.0);
  EXPECT_EQ(manifold_ray_hit_vec_length(miss), 0);
  manifold_delete_ray_hit_vec(miss);

  manifold_delete_manifold(cube);
}

TEST(CBIND, simplify_smoke) {
  ManifoldManifold* sphere = manifold_sphere(1.0, 100);

  // Simplify should produce a valid manifold with fewer or equal triangles.
  ManifoldManifold* simplified = manifold_simplify(sphere, 0.1);
  EXPECT_EQ(manifold_status(simplified), MANIFOLD_NO_ERROR);
  EXPECT_LE(manifold_num_tri(simplified), manifold_num_tri(sphere));

  manifold_delete_manifold(sphere);
  manifold_delete_manifold(simplified);
}

TEST(CBIND, num_prop_vert) {
  ManifoldManifold* cube = manifold_cube(1.0, 1.0, 1.0, 0);

  // A cube has 8 geometric vertices but more property vertices (due to
  // duplicated normals at sharp edges).
  EXPECT_EQ(manifold_num_vert(cube), 8);
  EXPECT_GE(manifold_num_prop_vert(cube), manifold_num_vert(cube));

  manifold_delete_manifold(cube);
}

TEST(CBIND, meshgl_run_accessors) {
  // Create two shapes with original IDs so the boolean result has 2 runs.
  ManifoldManifold* cube_tmp = manifold_cube(1, 1, 1, 0);
  ManifoldManifold* cube = manifold_as_original(cube_tmp, -1);
  ManifoldManifold* sphere_tmp = manifold_sphere(0.6, 32);
  ManifoldManifold* sphere_trans =
      manifold_translate(sphere_tmp, 0.5, 0.5, 0.5);
  ManifoldManifold* sphere = manifold_as_original(sphere_trans, -1);
  ManifoldManifold* result = manifold_union(cube, sphere);
  EXPECT_EQ(manifold_status(result), MANIFOLD_NO_ERROR);

  // MeshGL
  ManifoldMeshGL* mesh = manifold_get_meshgl(result);
  EXPECT_GE(manifold_meshgl_tolerance(mesh), 0.0f);
  EXPECT_EQ(manifold_meshgl_num_run(mesh), 2);
  EXPECT_EQ(manifold_meshgl_run_flags_length(mesh),
            manifold_meshgl_num_run(mesh));

  size_t flags_len = manifold_meshgl_run_flags_length(mesh);
  uint8_t* flags = manifold_meshgl_run_flags(mesh);
  // Just verify we got data without crashing.
  EXPECT_NE(flags, (uint8_t*)NULL);

  // MeshGL64
  ManifoldMeshGL64* mesh64 = manifold_get_meshgl64(result);
  EXPECT_GE(manifold_meshgl64_tolerance(mesh64), 0.0);
  EXPECT_EQ(manifold_meshgl64_num_run(mesh64), 2);
  EXPECT_EQ(manifold_meshgl64_run_flags_length(mesh64),
            manifold_meshgl64_num_run(mesh64));

  size_t flags64_len = manifold_meshgl64_run_flags_length(mesh64);
  uint8_t* flags64 = manifold_meshgl64_run_flags(mesh64);
  EXPECT_NE(flags64, (uint8_t*)NULL);

  manifold_delete_meshgl(mesh);
  manifold_delete_meshgl64(mesh64);
  manifold_delete_manifold(result);
  manifold_delete_manifold(sphere);
  manifold_delete_manifold(sphere_trans);
  manifold_delete_manifold(sphere_tmp);
  manifold_delete_manifold(cube);
  manifold_delete_manifold(cube_tmp);
}

TEST(CBIND, run_flag_accessors) {
  // CalculateNormals sets the per-run hasNormals bit; a subtractee cavity
  // gets backside set. Exercise the C accessors for both bits.
  ManifoldManifold* oldCube = manifold_cube(2.0, 2.0, 2.0, 1);
  ManifoldManifold* cube = manifold_as_original(oldCube, -1);
  ManifoldManifold* oldSphere = manifold_sphere(1.0, 32);
  ManifoldManifold* sphere = manifold_as_original(oldSphere, -1);
  ManifoldManifold* cut = manifold_difference(cube, sphere);
  ManifoldManifold* with_normals = manifold_calculate_normals(cut, 0, 60.0);
  ManifoldMeshGL* mesh = manifold_get_meshgl(with_normals);

  const size_t runs = manifold_meshgl_num_run(mesh);
  ASSERT_GE(runs, 2u);

  int backside_count = 0;
  int has_normals_count = 0;
  for (size_t r = 0; r < runs; ++r) {
    if (manifold_meshgl_backside(mesh, r)) ++backside_count;
    if (manifold_meshgl_has_normals(mesh, r)) ++has_normals_count;
  }
  EXPECT_EQ(backside_count, 1);
  EXPECT_EQ(has_normals_count, static_cast<int>(runs));

  // Out-of-range returns false.
  EXPECT_EQ(manifold_meshgl_backside(mesh, runs + 10), 0);
  EXPECT_EQ(manifold_meshgl_has_normals(mesh, runs + 10), 0);

  manifold_delete_meshgl(mesh);
  manifold_delete_manifold(with_normals);
  manifold_delete_manifold(cut);
  manifold_delete_manifold(sphere);
  manifold_delete_manifold(oldSphere);
  manifold_delete_manifold(cube);
  manifold_delete_manifold(oldCube);
}

TEST(CBIND, cross_section_non_zero) {
  // Doubled clockwise input distinguishes NonZero from Positive and EvenOdd.
  ManifoldVec2 pts[] = {{0, 0}, {0, 2}, {2, 2}, {2, 0}};
  ManifoldSimplePolygon* ring =
      manifold_simple_polygon(pts, 4);
  ManifoldSimplePolygon* rings[] = {ring, ring};
  ManifoldPolygons* polygons = manifold_polygons(rings, 2);
  ManifoldCrossSection* single = manifold_cross_section_non_zero_simple_polygon(ring);
  ManifoldCrossSection* multiple = manifold_cross_section_non_zero_polygons(polygons);
  EXPECT_NEAR(manifold_cross_section_area(single), 4., 1e-9);
  EXPECT_NEAR(manifold_cross_section_area(multiple), 4., 1e-9);
  manifold_delete_cross_section(single);
  manifold_delete_cross_section(multiple);
  manifold_delete_polygons(polygons);
  manifold_delete_simple_polygon(ring);
}

TEST(CBIND, cross_section_even_odd) {
  // Two same-direction squares, one nested in the other: positive fill leaves
  // the pair solid, even-odd cuts the inner square out.
  ManifoldVec2 outerPts[] = {{-2, -2}, {2, -2}, {2, 2}, {-2, 2}};
  ManifoldVec2 innerPts[] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  ManifoldSimplePolygon* outer = manifold_simple_polygon(outerPts, 4);
  ManifoldSimplePolygon* inner = manifold_simple_polygon(innerPts, 4);
  ManifoldSimplePolygon* sps[] = {outer, inner};
  ManifoldPolygons* ps = manifold_polygons(sps, 2);

  ManifoldCrossSection* positive = manifold_cross_section_of_polygons(ps);
  ManifoldCrossSection* evenOdd = manifold_cross_section_even_odd_polygons(ps);
  EXPECT_NEAR(manifold_cross_section_area(positive), 16.0, 1e-9);
  EXPECT_NEAR(manifold_cross_section_area(evenOdd), 12.0, 1e-9);

  // The single-contour entry point, on a clockwise square: even-odd goes by
  // parity so it fills, while positive fill discards the negative winding.
  ManifoldVec2 cwPts[] = {{0, 0}, {0, 2}, {2, 2}, {2, 0}};
  ManifoldSimplePolygon* cw = manifold_simple_polygon(cwPts, 4);
  ManifoldCrossSection* cwPositive =
      manifold_cross_section_of_simple_polygon(cw);
  ManifoldCrossSection* cwEvenOdd =
      manifold_cross_section_even_odd_simple_polygon(cw);
  EXPECT_NEAR(manifold_cross_section_area(cwPositive), 0.0, 1e-9);
  EXPECT_NEAR(manifold_cross_section_area(cwEvenOdd), 4.0, 1e-9);

  manifold_delete_simple_polygon(outer);
  manifold_delete_simple_polygon(inner);
  manifold_delete_simple_polygon(cw);
  manifold_delete_polygons(ps);
  manifold_delete_cross_section(positive);
  manifold_delete_cross_section(evenOdd);
  manifold_delete_cross_section(cwPositive);
  manifold_delete_cross_section(cwEvenOdd);
}

TEST(CBIND, execution_context_happy_path) {
  ManifoldExecutionContext* ctx = manifold_execution_context();
  EXPECT_EQ(manifold_execution_context_cancelled(ctx), 0);
  // Fresh ctx with no scheduled work reads as trivially complete (1.0).
  EXPECT_DOUBLE_EQ(manifold_execution_context_progress(ctx), 1.0);

  // Uncancelled ctx attached to a manifold means Status routes through ctx.
  ManifoldManifold* cube1 = manifold_cube(1, 1, 1, 0);
  ManifoldManifold* cube2 = manifold_cube(1, 1, 1, 1);
  ManifoldManifold* u = manifold_union(cube1, cube2);
  ManifoldManifold* uctx = manifold_with_context(u, ctx);
  EXPECT_EQ(manifold_status(uctx), MANIFOLD_NO_ERROR);
  EXPECT_EQ(manifold_execution_context_cancelled(ctx), 0);

  manifold_delete_manifold(uctx);
  manifold_delete_manifold(u);
  manifold_delete_manifold(cube2);
  manifold_delete_manifold(cube1);
  manifold_delete_execution_context(ctx);
}

TEST(CBIND, execution_context_cancel) {
  // alloc + delete pattern (matches the Rust binding's usage of the C API).
  ManifoldExecutionContext* ctx = manifold_execution_context();
  manifold_execution_context_cancel(ctx);
  EXPECT_EQ(manifold_execution_context_cancelled(ctx), 1);

  // Status on a Manifold that needs evaluation, with the cancelled ctx
  // attached, short-circuits to MANIFOLD_CANCELLED.
  ManifoldManifold* cube1 = manifold_cube(1, 1, 1, 0);
  ManifoldManifold* cube2 = manifold_cube(1, 1, 1, 1);
  ManifoldManifold* u = manifold_union(cube1, cube2);
  ManifoldManifold* uctx = manifold_with_context(u, ctx);
  EXPECT_EQ(manifold_status(uctx), MANIFOLD_CANCELLED);

  manifold_delete_manifold(uctx);
  manifold_delete_manifold(u);
  manifold_delete_manifold(cube2);
  manifold_delete_manifold(cube1);
  manifold_delete_execution_context(ctx);
}
