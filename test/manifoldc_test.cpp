#include "manifoldc.h"

#include "gtest/gtest.h"
#include "manifold.h"
#include "polygon.h"
#include "sdf.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

float eps = 0.000001;

bool approx_vec3(ManifoldVec3 &a, ManifoldVec3 &b) {
  return abs(a.x - b.x) < eps && abs(a.y - b.y) < eps && abs(a.z - b.z) < eps;
}

bool approx_vec3_array(ManifoldVec3 *a, ManifoldVec3 *b, size_t len) {
  for (int i = 0; i < len; ++i) {
    if (!approx_vec3(a[i], b[i])) {
      return false;
    };
  }
  return true;
}

TEST(CBIND, sphere) {
  int n = 25;
  size_t sz = manifold_manifold_size();
  ManifoldManifold *sphere = manifold_sphere(malloc(sz), 1.0f, 4 * n);

  EXPECT_EQ(manifold_status(sphere), NO_ERROR);
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

  ManifoldMesh *trans_mesh =
      manifold_get_mesh(malloc(manifold_mesh_size()), trans);
  ManifoldMesh *warped_mesh =
      manifold_get_mesh(malloc(manifold_mesh_size()), warped);

  size_t n_verts = manifold_mesh_vert_length(trans_mesh);
  ManifoldVec3 *trans_verts = manifold_mesh_vert_pos(
      malloc(sizeof(ManifoldVec3) * n_verts), trans_mesh);
  ManifoldVec3 *warped_verts = manifold_mesh_vert_pos(
      malloc(sizeof(ManifoldVec3) * n_verts), warped_mesh);

  EXPECT_TRUE(approx_vec3_array(trans_verts, warped_verts, n_verts));

  manifold_delete_manifold(sphere);
  manifold_delete_manifold(trans);
  manifold_delete_manifold(warped);
  manifold_delete_mesh(trans_mesh);
  manifold_delete_mesh(warped_mesh);
  delete trans_verts;
  delete warped_verts;
}
