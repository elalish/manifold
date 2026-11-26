#include "manifold/manifoldc.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
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

const ManifoldCircularQuality kQuality32 = {32, 10.0, 1.0};
const ManifoldCircularQuality kQuality100 = {100, 10.0, 1.0};

TEST(CBIND, error_string) {
  for (int i = 0; i <= MANIFOLD_CANCELLED; ++i) {
    EXPECT_STREQ(manifold_error_string(static_cast<ManifoldError>(i)),
                 manifold::ToString(static_cast<manifold::Manifold::Error>(i)));
  }
  EXPECT_STREQ(
      manifold_error_string(static_cast<ManifoldError>(MANIFOLD_CANCELLED + 1)),
      "Unknown error");
}

TEST(CBIND, circular_quality) {
  const ManifoldCircularQuality q = {0, 22.5, 0.125};
  EXPECT_EQ(manifold_circular_quality_segments_for(2.0, &q), 16);
  EXPECT_EQ(manifold_circular_quality_segments_for(1.0, NULL), 8);
  const ManifoldCircularQuality exact =
      manifold_circular_quality_for_divisions(21, 0.7);
  EXPECT_EQ(exact.segments, 10543);

  ManifoldCrossSection* circle = manifold_cross_section_circle(2.0, &q);
  EXPECT_EQ(manifold_cross_section_num_vert(circle), 16);
  ManifoldCrossSection* defaults = manifold_cross_section_circle(1.0, NULL);
  EXPECT_EQ(manifold_cross_section_num_vert(defaults), 8);
  ManifoldManifold* cylinder = manifold_cylinder(1.0, 2.0, -1.0, &q, 0);
  EXPECT_EQ(manifold_num_vert(cylinder), 32);
  manifold_delete_cross_section(circle);
  manifold_delete_cross_section(defaults);
  manifold_delete_manifold(cylinder);
}

TEST(CBIND, meshgl_no_props) {
  const float vertProps[] = {0, 0, 0};
  ManifoldMeshGL* mesh = manifold_meshgl(vertProps, 1, 0, NULL, 0);
  EXPECT_EQ(manifold_meshgl_num_vert(mesh), 0);
  manifold_delete_meshgl(mesh);
}

TEST(CBIND, meshgl_options) {
  const float vertProps[] = {
      0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
  };
  const uint32_t triVerts[] = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  const uint32_t runIndices[] = {0, 3, 6, 9, 12};
  uint32_t runOriginalIDs[] = {manifold_reserve_ids(1),
                               manifold_reserve_ids(1)};
  const uint32_t mergeFrom[] = {0, 1, 2};
  const uint32_t mergeTo[] = {1, 2, 3};
  const float runTransform[] = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3,
  };
  const uint8_t runFlags[] = {2, 3, 4, 5};
  const uint32_t faceIDs[] = {10, 11, 12, 13};
  const float tangents[48] = {};

  ManifoldMeshGLOptions options = manifold_meshgl_options_init();
  options.run_indices = runIndices;
  options.run_indices_length = 5;
  options.run_original_ids = runOriginalIDs;
  options.run_original_ids_length = 2;
  options.merge_from_vert = mergeFrom;
  options.merge_to_vert = mergeTo;
  options.merge_verts_length = 3;
  options.halfedge_tangents = tangents;
  options.run_transform = runTransform;
  options.run_transform_length = 24;
  options.run_flags = runFlags;
  options.run_flags_length = 4;
  options.face_id = faceIDs;

  ManifoldMeshGL* mesh =
      manifold_meshgl_w_options(vertProps, 4, 6, triVerts, 4, &options);
  ASSERT_EQ(manifold_meshgl_merge_length(mesh), 3);
  EXPECT_EQ(manifold_meshgl_merge_from_vert(mesh)[0], 0);
  EXPECT_EQ(manifold_meshgl_merge_to_vert(mesh)[0], 1);
  EXPECT_EQ(manifold_meshgl_tangent_length(mesh), 48);
  EXPECT_EQ(manifold_meshgl_run_index_length(mesh), 5);
  EXPECT_EQ(manifold_meshgl_run_original_id_length(mesh), 2);
  ASSERT_EQ(manifold_meshgl_run_transform_length(mesh), 24);
  EXPECT_FLOAT_EQ(manifold_meshgl_run_transform(mesh)[23], 3);
  ASSERT_EQ(manifold_meshgl_run_flags_length(mesh), 4);
  EXPECT_EQ(manifold_meshgl_run_flags(mesh)[3], 5);
  ASSERT_EQ(manifold_meshgl_face_id_length(mesh), 4);
  EXPECT_EQ(manifold_meshgl_face_id(mesh)[3], 13);

  ManifoldMeshGL* plain =
      manifold_meshgl_w_options(vertProps, 4, 6, triVerts, 4, nullptr);
  ASSERT_NE(plain, nullptr);
  EXPECT_EQ(manifold_meshgl_face_id_length(plain), 0);

  ManifoldMeshGLOptions v40Options = options;
  v40Options.struct_size =
      offsetof(ManifoldMeshGLOptions, face_id) + sizeof(v40Options.face_id);
  ManifoldMeshGL* v40 =
      manifold_meshgl_w_options(vertProps, 4, 6, triVerts, 4, &v40Options);
  ASSERT_NE(v40, nullptr);
  EXPECT_EQ(manifold_meshgl_face_id_length(v40), 4);

  ManifoldMeshGLOptions largerOptions = options;
  largerOptions.struct_size = sizeof(largerOptions) + 8;
  ManifoldMeshGL* larger =
      manifold_meshgl_w_options(vertProps, 4, 6, triVerts, 4, &largerOptions);
  ASSERT_NE(larger, nullptr);
  EXPECT_EQ(manifold_meshgl_face_id_length(larger), 4);

  manifold_delete_meshgl(larger);
  manifold_delete_meshgl(v40);
  manifold_delete_meshgl(plain);
  manifold_delete_meshgl(mesh);
}

TEST(CBIND, meshgl_options_run_flags_rotate_normals) {
  ManifoldManifold* cube = manifold_cube(1, 1, 1, 1);
  ManifoldManifold* normals = manifold_calculate_normals(cube, 60);
  ManifoldMeshGL* source = manifold_get_meshgl(normals);
  const size_t numVert = manifold_meshgl_num_vert(source);
  const size_t numProp = manifold_meshgl_num_prop(source);
  const size_t numTri = manifold_meshgl_num_tri(source);

  ManifoldMeshGLOptions options = manifold_meshgl_options_init();
  options.run_indices = manifold_meshgl_run_index(source);
  options.run_indices_length = manifold_meshgl_run_index_length(source);
  options.run_original_ids = manifold_meshgl_run_original_id(source);
  options.run_original_ids_length =
      manifold_meshgl_run_original_id_length(source);
  options.merge_from_vert = manifold_meshgl_merge_from_vert(source);
  options.merge_to_vert = manifold_meshgl_merge_to_vert(source);
  options.merge_verts_length = manifold_meshgl_merge_length(source);
  options.halfedge_tangents = manifold_meshgl_halfedge_tangent(source);
  options.run_transform = manifold_meshgl_run_transform(source);
  options.run_transform_length = manifold_meshgl_run_transform_length(source);
  options.run_flags = manifold_meshgl_run_flags(source);
  options.run_flags_length = manifold_meshgl_run_flags_length(source);
  ManifoldMeshGL* flagged = manifold_meshgl_w_options(
      manifold_meshgl_vert_properties(source), numVert, numProp,
      manifold_meshgl_tri_verts(source), numTri, &options);

  ManifoldManifold* flaggedManifold = manifold_of_meshgl(flagged);
  ASSERT_EQ(manifold_status(flaggedManifold), MANIFOLD_NO_ERROR);
  ManifoldManifold* flaggedRotated = manifold_rotate(flaggedManifold, 90, 0, 0);
  ManifoldMeshGL* flaggedOut = manifold_get_meshgl(flaggedRotated);
  ManifoldManifold* expectedRotated = manifold_rotate(normals, 90, 0, 0);
  ManifoldMeshGL* expected = manifold_get_meshgl(expectedRotated);
  ASSERT_EQ(manifold_meshgl_vert_properties_length(flaggedOut),
            manifold_meshgl_vert_properties_length(expected));
  const float* flaggedProps = manifold_meshgl_vert_properties(flaggedOut);
  const float* expectedProps = manifold_meshgl_vert_properties(expected);
  for (size_t vert = 0; vert < manifold_meshgl_num_vert(flaggedOut); ++vert) {
    for (size_t prop = 3; prop < 6; ++prop) {
      EXPECT_NEAR(flaggedProps[vert * numProp + prop],
                  expectedProps[vert * numProp + prop], 1e-5);
    }
  }

  uint8_t plainFlags[1] = {0};
  options.run_flags = plainFlags;
  options.run_flags_length = 1;
  ManifoldMeshGL* plain = manifold_meshgl_w_options(
      manifold_meshgl_vert_properties(source), numVert, numProp,
      manifold_meshgl_tri_verts(source), numTri, &options);
  ManifoldManifold* plainManifold = manifold_of_meshgl(plain);
  ManifoldManifold* plainRotated = manifold_rotate(plainManifold, 90, 0, 0);
  ManifoldMeshGL* plainOut = manifold_get_meshgl(plainRotated);
  const float* plainProps = manifold_meshgl_vert_properties(plainOut);
  bool differsFromRotatedNormals = false;
  for (size_t vert = 0; vert < manifold_meshgl_num_vert(plainOut); ++vert) {
    for (size_t prop = 3; prop < 6; ++prop) {
      differsFromRotatedNormals |=
          std::abs(plainProps[vert * numProp + prop] -
                   expectedProps[vert * numProp + prop]) > 1e-5;
    }
  }
  EXPECT_TRUE(differsFromRotatedNormals);

  manifold_delete_meshgl(plainOut);
  manifold_delete_manifold(plainRotated);
  manifold_delete_manifold(plainManifold);
  manifold_delete_meshgl(plain);
  manifold_delete_meshgl(expected);
  manifold_delete_manifold(expectedRotated);
  manifold_delete_meshgl(flaggedOut);
  manifold_delete_manifold(flaggedRotated);
  manifold_delete_manifold(flaggedManifold);
  manifold_delete_meshgl(flagged);
  manifold_delete_meshgl(source);
  manifold_delete_manifold(normals);
  manifold_delete_manifold(cube);
}

TEST(CBIND, meshgl64_options) {
  const double vertProps[] = {
      0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
  };
  const uint64_t triVerts[] = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  const uint64_t runIndices[] = {0, 3, 6, 9, 12};
  const uint32_t runOriginalIDs[] = {7, 8};
  const uint64_t mergeFrom[] = {0, 1, 2};
  const uint64_t mergeTo[] = {1, 2, 3};
  const double runTransform[] = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3,
  };
  const uint8_t runFlags[] = {2, 3, 4, 5};
  const uint64_t faceIDs[] = {10, 11, 12, 13};
  const double tangents[48] = {};

  ManifoldMeshGL64Options options = manifold_meshgl64_options_init();
  options.run_indices = runIndices;
  options.run_indices_length = 5;
  options.run_original_ids = runOriginalIDs;
  options.run_original_ids_length = 2;
  options.merge_from_vert = mergeFrom;
  options.merge_to_vert = mergeTo;
  options.merge_verts_length = 3;
  options.halfedge_tangents = tangents;
  options.run_transform = runTransform;
  options.run_transform_length = 24;
  options.run_flags = runFlags;
  options.run_flags_length = 4;
  options.face_id = faceIDs;

  ManifoldMeshGL64* mesh =
      manifold_meshgl64_w_options(vertProps, 4, 6, triVerts, 4, &options);
  ASSERT_EQ(manifold_meshgl64_merge_length(mesh), 3);
  EXPECT_EQ(manifold_meshgl64_merge_from_vert(mesh)[0], 0);
  EXPECT_EQ(manifold_meshgl64_merge_to_vert(mesh)[0], 1);
  EXPECT_EQ(manifold_meshgl64_tangent_length(mesh), 48);
  EXPECT_EQ(manifold_meshgl64_run_index_length(mesh), 5);
  EXPECT_EQ(manifold_meshgl64_run_original_id_length(mesh), 2);
  ASSERT_EQ(manifold_meshgl64_run_transform_length(mesh), 24);
  EXPECT_DOUBLE_EQ(manifold_meshgl64_run_transform(mesh)[23], 3);
  ASSERT_EQ(manifold_meshgl64_run_flags_length(mesh), 4);
  EXPECT_EQ(manifold_meshgl64_run_flags(mesh)[3], 5);
  ASSERT_EQ(manifold_meshgl64_face_id_length(mesh), 4);
  EXPECT_EQ(manifold_meshgl64_face_id(mesh)[3], 13);

  ManifoldMeshGL64* plain =
      manifold_meshgl64_w_options(vertProps, 4, 6, triVerts, 4, nullptr);
  ASSERT_NE(plain, nullptr);
  EXPECT_EQ(manifold_meshgl64_face_id_length(plain), 0);

  ManifoldMeshGL64Options v40Options = options;
  v40Options.struct_size =
      offsetof(ManifoldMeshGL64Options, face_id) + sizeof(v40Options.face_id);
  ManifoldMeshGL64* v40 =
      manifold_meshgl64_w_options(vertProps, 4, 6, triVerts, 4, &v40Options);
  ASSERT_NE(v40, nullptr);
  EXPECT_EQ(manifold_meshgl64_face_id_length(v40), 4);

  ManifoldMeshGL64Options largerOptions = options;
  largerOptions.struct_size = sizeof(largerOptions) + 8;
  ManifoldMeshGL64* larger =
      manifold_meshgl64_w_options(vertProps, 4, 6, triVerts, 4, &largerOptions);
  ASSERT_NE(larger, nullptr);
  EXPECT_EQ(manifold_meshgl64_face_id_length(larger), 4);

  manifold_delete_meshgl64(larger);
  manifold_delete_meshgl64(v40);
  manifold_delete_meshgl64(plain);
  manifold_delete_meshgl64(mesh);
}

#ifndef __EMSCRIPTEN__
TEST(CBIND, meshgl_options_short_size_aborts) {
#ifdef GTEST_FLAG_SET
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#else
  testing::GTEST_FLAG(death_test_style) = "threadsafe";
#endif
  const float vertProps[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const uint32_t triVerts[] = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  const double vertProps64[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const uint64_t triVerts64[] = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  ManifoldMeshGLOptions shortOptions = manifold_meshgl_options_init();
  shortOptions.struct_size = offsetof(ManifoldMeshGLOptions, face_id) +
                             sizeof(shortOptions.face_id) - 1;
  EXPECT_DEATH(
      manifold_meshgl_w_options(vertProps, 4, 3, triVerts, 4, &shortOptions),
      "struct_size");
  shortOptions.struct_size = 0;
  EXPECT_DEATH(
      manifold_meshgl_w_options(vertProps, 4, 3, triVerts, 4, &shortOptions),
      "struct_size");
  ManifoldMeshGL64Options shortOptions64 = manifold_meshgl64_options_init();
  shortOptions64.struct_size = offsetof(ManifoldMeshGL64Options, face_id) +
                               sizeof(shortOptions64.face_id) - 1;
  EXPECT_DEATH(manifold_meshgl64_w_options(vertProps64, 4, 3, triVerts64, 4,
                                           &shortOptions64),
               "struct_size");
  shortOptions64.struct_size = 0;
  EXPECT_DEATH(manifold_meshgl64_w_options(vertProps64, 4, 3, triVerts64, 4,
                                           &shortOptions64),
               "struct_size");
}
#endif

TEST(CBIND, sphere) {
  int n = 25;
  const ManifoldCircularQuality q = {4 * n, 10.0, 1.0};
  ManifoldManifold* sphere = manifold_sphere(1.0, &q);

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
  ManifoldManifold* sphere = manifold_sphere(1.0, &kQuality100);
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

struct BatchCalls {
  int calls = 0;
  size_t count = 0;
};

ManifoldVec3 WarpPoint3(double x, double y, double z, void*) {
  return {x + 0.3 * std::sin(2 * y), y * (1 + 0.2 * z), z + 0.1 * x * y};
}

void WarpBatch3(double* xyz, size_t count, void* ctx) {
  BatchCalls* calls = static_cast<BatchCalls*>(ctx);
  ++calls->calls;
  calls->count += count;
  for (size_t i = 0; i < count; ++i) {
    ManifoldVec3 v =
        WarpPoint3(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], nullptr);
    xyz[3 * i] = v.x;
    xyz[3 * i + 1] = v.y;
    xyz[3 * i + 2] = v.z;
  }
}

TEST(CBIND, warp_batch_matches_warp) {
  ManifoldManifold* cube = manifold_cube(1, 2, 3, 1);
  ManifoldManifold* sphere = manifold_sphere(1.0, &kQuality32);
  ManifoldManifold* shapes[] = {
      manifold_copy(cube),
      manifold_copy(sphere),
      manifold_cylinder(2.0, 1.0, 0.5, &kQuality32, 0),
      manifold_union(cube, sphere),
      manifold_empty(),
  };
  manifold_delete_manifold(cube);
  manifold_delete_manifold(sphere);
  for (ManifoldManifold* shape : shapes) {
    const double volume = manifold_volume(shape);
    BatchCalls calls;
    ManifoldManifold* batch = manifold_warp_batch(shape, WarpBatch3, &calls);
    ManifoldManifold* single = manifold_warp(shape, WarpPoint3, nullptr);
    EXPECT_EQ(calls.calls, 1);
    EXPECT_EQ(calls.count, manifold_num_vert(shape));
    EXPECT_EQ(manifold_status(batch), MANIFOLD_NO_ERROR);

    ManifoldMeshGL64* a = manifold_get_meshgl64(batch);
    ManifoldMeshGL64* b = manifold_get_meshgl64(single);
    ASSERT_EQ(manifold_meshgl64_vert_properties_length(a),
              manifold_meshgl64_vert_properties_length(b));
    ASSERT_EQ(manifold_meshgl64_tri_length(a), manifold_meshgl64_tri_length(b));
    const double* propA = manifold_meshgl64_vert_properties(a);
    const double* propB = manifold_meshgl64_vert_properties(b);
    for (size_t i = 0; i < manifold_meshgl64_vert_properties_length(a); ++i) {
      EXPECT_EQ(propA[i], propB[i]);
    }
    const uint64_t* triA = manifold_meshgl64_tri_verts(a);
    const uint64_t* triB = manifold_meshgl64_tri_verts(b);
    for (size_t i = 0; i < manifold_meshgl64_tri_length(a); ++i) {
      EXPECT_EQ(triA[i], triB[i]);
    }
    if (volume > 0) {
      EXPECT_NE(manifold_volume(batch), volume);
    }
    EXPECT_EQ(manifold_volume(shape), volume);

    manifold_delete_meshgl64(a);
    manifold_delete_meshgl64(b);
    manifold_delete_manifold(batch);
    manifold_delete_manifold(single);
    manifold_delete_manifold(shape);
  }
}

TEST(CBIND, warp_batch_skips_errored_input) {
  const double verts[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const uint64_t tris[] = {0, 1, 2};
  ManifoldMeshGL64* mesh = manifold_meshgl64(verts, 3, 3, tris, 1);
  ManifoldManifold* bad = manifold_of_meshgl64(mesh);
  ASSERT_EQ(manifold_status(bad), MANIFOLD_NOT_MANIFOLD);

  BatchCalls calls;
  ManifoldManifold* warped = manifold_warp_batch(bad, WarpBatch3, &calls);
  EXPECT_EQ(calls.calls, 0);
  EXPECT_EQ(manifold_status(warped), MANIFOLD_NOT_MANIFOLD);

  manifold_delete_manifold(warped);
  manifold_delete_manifold(bad);
  manifold_delete_meshgl64(mesh);
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

TEST(CBIND, set_properties_null_zero_fills) {
  ManifoldManifold* cube = manifold_cube(1.0, 1.0, 1.0, 1);
  ManifoldManifold* zeroed = manifold_set_properties(cube, 2, NULL, NULL);
  EXPECT_EQ(manifold_status(zeroed), MANIFOLD_NO_ERROR);
  EXPECT_EQ(manifold_num_prop(zeroed), 2);

  ManifoldMeshGL* mesh = manifold_get_meshgl(zeroed);
  const size_t num_prop = manifold_meshgl_num_prop(mesh);
  ASSERT_EQ(num_prop, 5);
  const float* props = manifold_meshgl_vert_properties(mesh);
  const size_t len = manifold_meshgl_vert_properties_length(mesh);
  ASSERT_GT(len, 0);
  for (size_t i = 0; i < len; i += num_prop) {
    EXPECT_EQ(props[i + 3], 0);
    EXPECT_EQ(props[i + 4], 0);
  }

  manifold_delete_meshgl(mesh);
  manifold_delete_manifold(zeroed);
  manifold_delete_manifold(cube);
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

TEST(CBIND, batch_boolean_decompose) {
  ManifoldManifold* s1 = manifold_sphere(1.0, &kQuality100);
  ManifoldManifold* s2 = manifold_translate(s1, 2., 2., 2.);
  ManifoldManifoldVec* ss = manifold_manifold_vec(2);
  manifold_manifold_vec_set(ss, 0, s1);
  manifold_manifold_vec_set(ss, 1, s2);
  ManifoldManifold* composed = manifold_batch_boolean(ss, MANIFOLD_ADD);

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

TEST(CBIND, polygon_copies_own_their_storage) {
  const ManifoldVec2 expected[] = {{0, 0}, {3, 0}, {0, 2}};
  ManifoldVec2 points[] = {expected[0], expected[1], expected[2]};
  ManifoldSimplePolygon* triangle = manifold_simple_polygon(points, 3);
  ManifoldSimplePolygon* empty = manifold_simple_polygon(nullptr, 0);
  const ManifoldSimplePolygon* inputs[] = {empty, triangle};
  ManifoldPolygons* polygons = manifold_polygons(inputs, 2);

  points[1] = {100, 200};
  manifold_delete_simple_polygon(triangle);
  manifold_delete_simple_polygon(empty);

  ASSERT_EQ(manifold_polygons_length(polygons), 2);
  ManifoldSimplePolygon* empty_copy = manifold_polygons_get_simple(polygons, 0);
  ManifoldSimplePolygon* triangle_copy =
      manifold_polygons_get_simple(polygons, 1);
  manifold_delete_polygons(polygons);

  EXPECT_EQ(manifold_simple_polygon_length(empty_copy), 0);
  ASSERT_EQ(manifold_simple_polygon_length(triangle_copy), 3);
  for (size_t i = 0; i < 3; ++i) {
    ManifoldVec2 actual = manifold_simple_polygon_get_point(triangle_copy, i);
    EXPECT_EQ(actual.x, expected[i].x);
    EXPECT_EQ(actual.y, expected[i].y);
  }
  manifold_delete_simple_polygon(empty_copy);
  manifold_delete_simple_polygon(triangle_copy);
}

TEST(CBIND, exported_polygons_outlive_cross_section) {
  ManifoldCrossSection* square = manifold_cross_section_square(3, 2, false);
  ManifoldPolygons* polygons = manifold_cross_section_to_polygons(square);
  manifold_delete_cross_section(square);

  ASSERT_EQ(manifold_polygons_length(polygons), 1);
  EXPECT_EQ(manifold_polygons_simple_length(polygons, 0), 4);
  ManifoldCrossSection* restored = manifold_cross_section_of_polygons(polygons);
  manifold_delete_polygons(polygons);
  EXPECT_DOUBLE_EQ(manifold_cross_section_area(restored), 6.0);
  manifold_delete_cross_section(restored);
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
  const int* tri_verts = manifold_triangulation_tri_verts(triangulation);

  EXPECT_EQ(num_tri, 1);
  EXPECT_EQ(tri_verts[0], 0);
  EXPECT_EQ(tri_verts[1], 1);
  EXPECT_EQ(tri_verts[2], 2);
  manifold_delete_triangulation(triangulation);
}

TEST(CBIND, meshgl_merge_returns_mem) {
  // Create a cube and get its meshgl (which already has merge vectors).
  ManifoldManifold* cube = manifold_cube(1.0, 1.0, 1.0, 0);
  ManifoldMeshGL* original = manifold_get_meshgl(cube);

  // This mesh is already manifold, so Merge() returns an unchanged copy.
  // The bug: old code returned the input pointer instead of the output buffer.
  ManifoldMeshGL* merged = manifold_meshgl_merge(original, 0.0f);

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

  ManifoldMeshGL64* merged = manifold_meshgl64_merge(original, 0.0);

  EXPECT_NE(merged, original);

  ManifoldManifold* result = manifold_of_meshgl64(merged);
  EXPECT_EQ(manifold_status(result), MANIFOLD_NO_ERROR);
  EXPECT_NEAR(manifold_volume(result), 1.0, 0.0001);

  manifold_delete_manifold(result);
  manifold_delete_meshgl64(merged);
  manifold_delete_meshgl64(original);
  manifold_delete_manifold(cube);
}

TEST(CBIND, precision) {
  ManifoldManifold* cube = manifold_cube(1.0, 1.0, 1.0, 0);
  ManifoldMeshGL* mesh = manifold_get_meshgl(cube);
  ManifoldMeshGL64* mesh64 = manifold_get_meshgl64(cube);
  ManifoldManifold* ingested = manifold_of_meshgl(mesh);
  ManifoldManifold* ingested64 = manifold_of_meshgl64(mesh64);
  EXPECT_EQ(manifold_get_precision(cube), FLT_EPSILON);
  EXPECT_EQ(manifold_get_precision(ingested), FLT_EPSILON);
  EXPECT_EQ(manifold_get_precision(ingested64),
            MANIFOLD_MIN_RELATIVE_PRECISION);

  ManifoldManifold* stamped = manifold_with_precision(cube, 1e-2);
  EXPECT_EQ(manifold_get_precision(stamped), 1e-2);
  EXPECT_EQ(manifold_num_tri(stamped), manifold_num_tri(cube));
  ManifoldManifold* fine = manifold_with_precision(ingested, 1e-20);
  EXPECT_EQ(manifold_get_precision(fine), MANIFOLD_MIN_RELATIVE_PRECISION);
  // 0, negative values, NaN and infinity give FLT_EPSILON, even for a
  // manifold made from a MeshGL64.
  for (double unset :
       {0.0, -1.0, static_cast<double>(NAN), static_cast<double>(INFINITY)}) {
    ManifoldManifold* reset = manifold_with_precision(ingested64, unset);
    EXPECT_EQ(manifold_get_precision(reset), FLT_EPSILON);
    manifold_delete_manifold(reset);
  }

  ManifoldManifold* empty = manifold_empty();
  ManifoldManifold* coarseEmpty = manifold_with_precision(empty, 1e-3);
  EXPECT_EQ(manifold_epsilon(empty), -1);
  EXPECT_EQ(manifold_epsilon(coarseEmpty), -1);

  manifold_delete_manifold(coarseEmpty);
  manifold_delete_manifold(empty);
  manifold_delete_manifold(fine);
  manifold_delete_manifold(stamped);
  manifold_delete_manifold(ingested64);
  manifold_delete_manifold(ingested);
  manifold_delete_meshgl64(mesh64);
  manifold_delete_meshgl(mesh);
  manifold_delete_manifold(cube);
}

TEST(CBIND, execution_context_of_meshgl_precision) {
  ManifoldManifold* cube = manifold_cube(1.0, 1.0, 1.0, 0);
  ManifoldMeshGL* mesh = manifold_get_meshgl(cube);
  ManifoldMeshGL64* mesh64 = manifold_get_meshgl64(cube);
  ManifoldExecutionContext* ec = manifold_execution_context();
  ManifoldManifold* ingested = manifold_execution_context_of_meshgl(ec, mesh);
  ManifoldManifold* ingested64 =
      manifold_execution_context_of_meshgl64(ec, mesh64);
  EXPECT_EQ(manifold_get_precision(ingested), FLT_EPSILON);
  EXPECT_EQ(manifold_get_precision(ingested64),
            MANIFOLD_MIN_RELATIVE_PRECISION);
  EXPECT_DOUBLE_EQ(manifold_epsilon(ingested), FLT_EPSILON);
  EXPECT_DOUBLE_EQ(manifold_epsilon(ingested64),
                   MANIFOLD_MIN_RELATIVE_PRECISION);

  manifold_delete_manifold(ingested);
  manifold_delete_manifold(ingested64);
  manifold_delete_execution_context(ec);
  manifold_delete_meshgl(mesh);
  manifold_delete_meshgl64(mesh64);
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

  ManifoldManifold* orig = manifold_as_original(cube, -1);
  ManifoldRayHitVec* origHits =
      manifold_ray_cast(orig, 0.0, 0.0, -5.0, 0.0, 0.0, 5.0);
  ASSERT_EQ(manifold_ray_hit_vec_length(origHits), 2);
  ManifoldRayHit h2 = manifold_ray_hit_vec_get(origHits, 0);
  EXPECT_EQ(h2.original_id, (uint32_t)manifold_original_id(orig));
  EXPECT_GT(h2.original_id, 0u);
  EXPECT_EQ(h2.face_id, 0u);
  manifold_delete_ray_hit_vec(origHits);
  manifold_delete_manifold(orig);

  manifold_delete_manifold(cube);
}

TEST(CBIND, simplify_smoke) {
  ManifoldManifold* sphere = manifold_sphere(1.0, &kQuality100);

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
  ManifoldManifold* sphere_tmp = manifold_sphere(0.6, &kQuality32);
  ManifoldManifold* sphere_trans =
      manifold_translate(sphere_tmp, 0.5, 0.5, 0.5);
  ManifoldManifold* sphere = manifold_as_original(sphere_trans, -1);
  ManifoldManifold* result = manifold_union(cube, sphere);
  EXPECT_EQ(manifold_status(result), MANIFOLD_NO_ERROR);

  // MeshGL
  ManifoldMeshGL* mesh = manifold_get_meshgl(result);
  EXPECT_EQ(manifold_meshgl_num_run(mesh), 2);
  EXPECT_EQ(manifold_meshgl_run_flags_length(mesh),
            manifold_meshgl_num_run(mesh));

  size_t flags_len = manifold_meshgl_run_flags_length(mesh);
  const uint8_t* flags = manifold_meshgl_run_flags(mesh);
  // Just verify we got data without crashing.
  EXPECT_NE(flags, (uint8_t*)NULL);

  // MeshGL64
  ManifoldMeshGL64* mesh64 = manifold_get_meshgl64(result);
  EXPECT_EQ(manifold_meshgl64_num_run(mesh64), 2);
  EXPECT_EQ(manifold_meshgl64_run_flags_length(mesh64),
            manifold_meshgl64_num_run(mesh64));

  size_t flags64_len = manifold_meshgl64_run_flags_length(mesh64);
  const uint8_t* flags64 = manifold_meshgl64_run_flags(mesh64);
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
  ManifoldManifold* oldSphere = manifold_sphere(1.0, &kQuality32);
  ManifoldManifold* sphere = manifold_as_original(oldSphere, -1);
  ManifoldManifold* cut = manifold_difference(cube, sphere);
  ManifoldManifold* with_normals = manifold_calculate_normals(cut, 60.0);
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
  ManifoldSimplePolygon* ring = manifold_simple_polygon(pts, 4);
  ManifoldSimplePolygon* rings[] = {ring, ring};
  ManifoldPolygons* polygons = manifold_polygons(rings, 2);
  ManifoldCrossSection* single =
      manifold_cross_section_non_zero_simple_polygon(ring);
  ManifoldCrossSection* multiple =
      manifold_cross_section_non_zero_polygons(polygons);
  EXPECT_NEAR(manifold_cross_section_area(single), 4., 1e-9);
  EXPECT_NEAR(manifold_cross_section_area(multiple), 4., 1e-9);
  manifold_delete_cross_section(single);
  manifold_delete_cross_section(multiple);
  manifold_delete_polygons(polygons);
  manifold_delete_simple_polygon(ring);
}

TEST(CBIND, cross_section_warp) {
  ManifoldVec2 pts[] = {{0, 0}, {0, 2}, {2, 2}, {2, 0}};
  ManifoldSimplePolygon* polygon = manifold_simple_polygon(pts, 4);
  ManifoldCrossSection* cross_section =
      manifold_cross_section_non_zero_simple_polygon(polygon);
  double dx = 1.0;
  ManifoldCrossSection* warped = manifold_cross_section_warp(
      cross_section,
      [](double x, double y, void* ctx) {
        return ManifoldVec2{x + *static_cast<double*>(ctx), y};
      },
      &dx);
  ManifoldRect* bounds = manifold_cross_section_bounds(warped);
  EXPECT_DOUBLE_EQ(manifold_rect_min(bounds).x, 1.0);
  EXPECT_DOUBLE_EQ(manifold_rect_max(bounds).x, 3.0);
  EXPECT_DOUBLE_EQ(manifold_rect_min(bounds).y, 0.0);
  EXPECT_DOUBLE_EQ(manifold_rect_max(bounds).y, 2.0);
  manifold_delete_rect(bounds);
  manifold_delete_cross_section(warped);
  manifold_delete_cross_section(cross_section);
  manifold_delete_simple_polygon(polygon);
}

ManifoldVec2 WarpPoint2(double x, double y, void*) {
  return {x + 0.2 * std::sin(y), 1.1 * y + 0.05 * x};
}

void WarpBatch2(double* xy, size_t count, void* ctx) {
  BatchCalls* calls = static_cast<BatchCalls*>(ctx);
  ++calls->calls;
  calls->count += count;
  for (size_t i = 0; i < count; ++i) {
    ManifoldVec2 v = WarpPoint2(xy[2 * i], xy[2 * i + 1], nullptr);
    xy[2 * i] = v.x;
    xy[2 * i + 1] = v.y;
  }
}

TEST(CBIND, cross_section_warp_batch_matches_warp) {
  ManifoldCrossSection* square = manifold_cross_section_square(3, 2, 1);
  ManifoldCrossSection* circle =
      manifold_cross_section_circle(0.5, &kQuality32);
  ManifoldCrossSection* shapes[] = {
      manifold_cross_section_copy(square),
      manifold_cross_section_copy(circle),
      manifold_cross_section_difference(square, circle),
      manifold_cross_section_empty(),
  };
  manifold_delete_cross_section(square);
  manifold_delete_cross_section(circle);
  for (ManifoldCrossSection* shape : shapes) {
    const double area = manifold_cross_section_area(shape);
    BatchCalls calls;
    ManifoldCrossSection* batch =
        manifold_cross_section_warp_batch(shape, WarpBatch2, &calls);
    ManifoldCrossSection* single =
        manifold_cross_section_warp(shape, WarpPoint2, nullptr);
    EXPECT_EQ(calls.calls, 1);
    EXPECT_EQ(calls.count, manifold_cross_section_num_vert(shape));
    if (area > 0) {
      EXPECT_NE(manifold_cross_section_area(batch), area);
    }
    EXPECT_EQ(manifold_cross_section_area(shape), area);

    ManifoldPolygons* a = manifold_cross_section_to_polygons(batch);
    ManifoldPolygons* b = manifold_cross_section_to_polygons(single);
    ASSERT_EQ(manifold_polygons_length(a), manifold_polygons_length(b));
    for (size_t i = 0; i < manifold_polygons_length(a); ++i) {
      ASSERT_EQ(manifold_polygons_simple_length(a, i),
                manifold_polygons_simple_length(b, i));
      for (size_t j = 0; j < manifold_polygons_simple_length(a, i); ++j) {
        ManifoldVec2 pa = manifold_polygons_get_point(a, i, j);
        ManifoldVec2 pb = manifold_polygons_get_point(b, i, j);
        EXPECT_EQ(pa.x, pb.x);
        EXPECT_EQ(pa.y, pb.y);
      }
    }

    manifold_delete_polygons(a);
    manifold_delete_polygons(b);
    manifold_delete_cross_section(batch);
    manifold_delete_cross_section(single);
    manifold_delete_cross_section(shape);
  }
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
