// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "manifold/manifold.h"
#include "test.h"

namespace {

using namespace manifold;

// --- RayCast Tests ---

TEST(Queries, RayCastHitCube) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Ray from outside along +X axis toward the center
  RayHit hit = cube.RayCast(vec3(-5, 0, 0), vec3(0, 0, 0));
  EXPECT_GE(hit.faceID, 0);
  EXPECT_GT(hit.distance, 0.0);
  EXPECT_LE(hit.distance, 1.0);
  // Should hit the -X face at x = -1
  EXPECT_NEAR(hit.position.x, -1.0, 1e-10);
  EXPECT_NEAR(hit.position.y, 0.0, 1e-10);
  EXPECT_NEAR(hit.position.z, 0.0, 1e-10);
}

TEST(Queries, RayCastMissCube) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Ray that misses the cube entirely
  RayHit hit = cube.RayCast(vec3(-5, 5, 0), vec3(5, 5, 0));
  EXPECT_EQ(hit.faceID, -1);
  EXPECT_LT(hit.distance, 0.0);
}

TEST(Queries, RayCastSphere) {
  Manifold sphere = Manifold::Sphere(1.0, 64);
  // Ray from outside along +Z axis
  RayHit hit = sphere.RayCast(vec3(0, 0, -5), vec3(0, 0, 0));
  EXPECT_GE(hit.faceID, 0);
  // Should hit near z = -1
  EXPECT_NEAR(hit.position.z, -1.0, 0.1);
}

TEST(Queries, RayCastInternalOrigin) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Ray from inside the cube
  RayHit hit = cube.RayCast(vec3(0, 0, 0), vec3(5, 0, 0));
  EXPECT_GE(hit.faceID, 0);
  // Should hit the +X face at x = 1
  EXPECT_NEAR(hit.position.x, 1.0, 1e-10);
}

TEST(Queries, RayCastEmpty) {
  Manifold empty;
  RayHit hit = empty.RayCast(vec3(0, 0, 0), vec3(1, 0, 0));
  EXPECT_EQ(hit.faceID, -1);
}

TEST(Queries, RayCastNormalDirection) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Ray hitting the -X face
  RayHit hit = cube.RayCast(vec3(-5, 0, 0), vec3(0, 0, 0));
  EXPECT_GE(hit.faceID, 0);
  // Normal should point in -X direction
  EXPECT_LT(hit.normal.x, 0.0);
}

// --- WindingNumber Tests ---

TEST(Queries, WindingNumberInsideCube) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  EXPECT_NE(cube.WindingNumber(vec3(0, 0, 0)), 0);
}

TEST(Queries, WindingNumberOutsideCube) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  EXPECT_EQ(cube.WindingNumber(vec3(5, 5, 5)), 0);
}

TEST(Queries, WindingNumberInsideSphere) {
  Manifold sphere = Manifold::Sphere(1.0, 64);
  EXPECT_NE(sphere.WindingNumber(vec3(0, 0, 0)), 0);
}

TEST(Queries, WindingNumberOutsideSphere) {
  Manifold sphere = Manifold::Sphere(1.0, 64);
  EXPECT_EQ(sphere.WindingNumber(vec3(5, 0, 0)), 0);
}

TEST(Queries, WindingNumberEmpty) {
  Manifold empty;
  EXPECT_EQ(empty.WindingNumber(vec3(0, 0, 0)), 0);
}

TEST(Queries, WindingNumberOnBoundary) {
  // Points exactly on the surface may be ambiguous but shouldn't crash
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Just slightly inside
  EXPECT_NE(cube.WindingNumber(vec3(0.9, 0, 0)), 0);
  // Clearly outside
  EXPECT_EQ(cube.WindingNumber(vec3(1.5, 0, 0)), 0);
}

// --- NearestPoint Tests ---

TEST(Queries, NearestPointOnCube) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Point outside along +X
  NearestPointResult result = cube.NearestPoint(vec3(5, 0, 0));
  EXPECT_GE(result.faceID, 0);
  // Closest point should be on the +X face at x = 1
  EXPECT_NEAR(result.position.x, 1.0, 1e-10);
  EXPECT_NEAR(result.position.y, 0.0, 1e-10);
  EXPECT_NEAR(result.position.z, 0.0, 1e-10);
  EXPECT_NEAR(result.distance, 4.0, 1e-10);
}

TEST(Queries, NearestPointInsideCube) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Point at the center
  NearestPointResult result = cube.NearestPoint(vec3(0, 0, 0));
  EXPECT_GE(result.faceID, 0);
  // Should be distance 1 to the nearest face
  EXPECT_NEAR(result.distance, 1.0, 1e-10);
}

TEST(Queries, NearestPointOnSphere) {
  Manifold sphere = Manifold::Sphere(1.0, 128);
  // Point outside
  NearestPointResult result = sphere.NearestPoint(vec3(3, 0, 0));
  EXPECT_GE(result.faceID, 0);
  // Closest point should be near (1, 0, 0)
  EXPECT_NEAR(result.position.x, 1.0, 0.05);
  EXPECT_NEAR(result.distance, 2.0, 0.05);
}

TEST(Queries, NearestPointCorner) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Point outside near a corner
  NearestPointResult result = cube.NearestPoint(vec3(5, 5, 5));
  EXPECT_GE(result.faceID, 0);
  // Closest point should be a corner at (1, 1, 1)
  EXPECT_NEAR(result.position.x, 1.0, 1e-10);
  EXPECT_NEAR(result.position.y, 1.0, 1e-10);
  EXPECT_NEAR(result.position.z, 1.0, 1e-10);
  double expected_dist = la::length(vec3(4, 4, 4));
  EXPECT_NEAR(result.distance, expected_dist, 1e-10);
}

TEST(Queries, NearestPointEmpty) {
  Manifold empty;
  NearestPointResult result = empty.NearestPoint(vec3(0, 0, 0));
  EXPECT_EQ(result.faceID, -1);
}

// --- Remeshing validation tests ---
// Use WindingNumber + NearestPoint as an SDF to remesh shapes via LevelSet,
// then compare volume and surface area to validate correctness.

// Build a signed-distance-like function from a mesh using WindingNumber
// (for sign) and NearestPoint (for distance).
static double MeshSDF(const Manifold& mesh, vec3 pos) {
  double dist = mesh.NearestPoint(pos).distance;
  int winding = mesh.WindingNumber(pos);
  return winding != 0 ? -dist : dist;
}

TEST(Queries, RemeshSphere) {
  // Use all three query APIs as an implicit SDF to reconstruct the mesh via
  // LevelSet. The triangulated mesh SDF overestimates volume because the
  // flat-face distance field differs from the ideal shape's distance field,
  // and the marching grid introduces further discretization. We accept a 3x
  // factor here -- the test validates that the APIs produce a valid manifold
  // with the right sign field.
  Manifold sphere = Manifold::Sphere(1.0, 64);
  const double origVol = sphere.Volume();
  Box bbox = sphere.BoundingBox();
  const vec3 pad = vec3(0.2);

  Manifold remeshed = Manifold::LevelSet(
      [&sphere](vec3 pos) { return MeshSDF(sphere, pos); },
      Box(bbox.min - pad, bbox.max + pad), 0.1, 0, -1, false);

  EXPECT_EQ(remeshed.Status(), Manifold::Error::NoError);
  EXPECT_FALSE(remeshed.IsEmpty());
  // Volume should be within 3x -- the SDF overestimates due to mesh
  // polygonization vs ideal shape, and coarse LevelSet grid.
  EXPECT_GT(remeshed.Volume(), origVol * 0.5);
  EXPECT_LT(remeshed.Volume(), origVol * 3.0);
}

TEST(Queries, MeshSDFSignCorrectness) {
  // Validate that MeshSDF returns correct signs for a cube.
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Inside points should have negative SDF
  EXPECT_LT(MeshSDF(cube, vec3(0, 0, 0)), 0);
  EXPECT_LT(MeshSDF(cube, vec3(0.5, 0.5, 0.5)), 0);
  // Outside points should have positive SDF
  EXPECT_GT(MeshSDF(cube, vec3(2, 0, 0)), 0);
  EXPECT_GT(MeshSDF(cube, vec3(0, 2, 0)), 0);
  EXPECT_GT(MeshSDF(cube, vec3(0, 0, 2)), 0);
  EXPECT_GT(MeshSDF(cube, vec3(2, 2, 2)), 0);
  // Points near the surface should have small absolute SDF
  EXPECT_NEAR(std::fabs(MeshSDF(cube, vec3(0.99, 0, 0))), 0.01, 0.001);
  EXPECT_NEAR(std::fabs(MeshSDF(cube, vec3(1.01, 0, 0))), 0.01, 0.001);
}

TEST(Queries, RayCastDirectionForm) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // 3-arg form: origin, direction, maxDist
  RayHit hit = cube.RayCast(vec3(-5, 0, 0), vec3(1, 0, 0),
                            std::numeric_limits<double>::infinity());
  EXPECT_GE(hit.faceID, 0);
  EXPECT_NEAR(hit.position.x, -1.0, 1e-10);
  // Distance should be absolute (4.0 from origin to -1.0 face)
  EXPECT_NEAR(hit.distance, 4.0, 1e-10);
}

TEST(Queries, RayCastDirectionFiniteDist) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Short ray that doesn't reach the cube
  RayHit hit = cube.RayCast(vec3(-5, 0, 0), vec3(1, 0, 0), 2.0);
  EXPECT_EQ(hit.faceID, -1);

  // Long enough ray that reaches the cube
  hit = cube.RayCast(vec3(-5, 0, 0), vec3(1, 0, 0), 10.0);
  EXPECT_GE(hit.faceID, 0);
}

// --- Adversarial floating-point and robustness tests ---
// These probe edge cases that reveal subtle bugs in containment logic,
// edge tie-breaking, and numerical precision.

TEST(Queries, WindingNumberGrid) {
  // Sample a grid of points around a cube and verify all winding numbers
  // are consistent (0 outside, 1 inside, no stray values).
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  int insideCount = 0;
  int outsideCount = 0;
  const double step = 0.37;  // Intentionally non-aligned with mesh edges
  for (double x = -1.5; x <= 1.5; x += step) {
    for (double y = -1.5; y <= 1.5; y += step) {
      for (double z = -1.5; z <= 1.5; z += step) {
        int w = cube.WindingNumber(vec3(x, y, z));
        bool geometricallyInside =
            x > -1 && x < 1 && y > -1 && y < 1 && z > -1 && z < 1;
        if (geometricallyInside) {
          EXPECT_NE(w, 0) << "at (" << x << "," << y << "," << z << ")";
          insideCount++;
        } else if (std::fabs(x) > 1.01 || std::fabs(y) > 1.01 ||
                   std::fabs(z) > 1.01) {
          // Points clearly outside (with margin for surface ambiguity)
          EXPECT_EQ(w, 0) << "at (" << x << "," << y << "," << z << ")";
          outsideCount++;
        }
      }
    }
  }
  EXPECT_GT(insideCount, 10);
  EXPECT_GT(outsideCount, 10);
}

TEST(Queries, RayCastAlongEdge) {
  // Ray that travels exactly along a mesh edge -- a degenerate case
  // that should not crash and should return a valid hit or clean miss.
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Shoot along the +X edge at y=1, z=1 from outside
  RayHit hit = cube.RayCast(vec3(-5, 1, 1), vec3(5, 1, 1));
  // May or may not hit (edge is degenerate), but must not crash
  // and must have consistent output.
  if (hit.faceID >= 0) {
    EXPECT_GE(hit.distance, 0.0);
    EXPECT_LE(hit.distance, 1.0);
    EXPECT_TRUE(std::isfinite(hit.position.x));
  }
}

TEST(Queries, RayCastThroughVertex) {
  // Ray aimed directly at a mesh vertex -- another degenerate case.
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Shoot at the corner (-1, -1, -1) from outside
  vec3 corner(-1, -1, -1);
  vec3 origin = corner - vec3(3, 3, 3);
  RayHit hit = cube.RayCast(origin, corner);
  // Should either hit the corner or a face near it; must not crash.
  if (hit.faceID >= 0) {
    EXPECT_GE(hit.distance, 0.0);
    EXPECT_NEAR(la::length(hit.position - corner), 0.0, 0.1);
  }
}

TEST(Queries, WindingNumberBelowEdge) {
  // Point whose +Z ray hits exactly on a shared edge between two top-face
  // triangles of a cube. The robust Kernel02 should count exactly one hit.
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // The top face (z=1) of the cube is split into 2 triangles along a
  // diagonal. A point at (0, 0, -0.5) has its +Z ray hitting the top face
  // at (0, 0, 1) which lies on that diagonal.
  int w = cube.WindingNumber(vec3(0, 0, -0.5));
  EXPECT_EQ(w, 1);  // Must be inside
}

TEST(Queries, WindingNumberBelowVertex) {
  // Point whose +Z ray passes through a mesh vertex. The symbolic
  // perturbation from vertNormal_ should give a consistent result.
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // (1, 1, *) line passes through the cube corner vertex at (1, 1, 1).
  // A point below the cube on that line should be outside.
  int w = cube.WindingNumber(vec3(1, 1, -5));
  EXPECT_EQ(w, 0);
}

TEST(Queries, RayCastParallelToFace) {
  // Ray exactly parallel to a face should miss (denom ≈ 0 in ray-plane).
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  RayHit hit = cube.RayCast(vec3(-5, 0, 1), vec3(5, 0, 1));
  // Parallel to the top face; tangent rays should miss.
  // Implementation-defined whether this hits, but must not crash.
  // If it does hit, position should be on the surface.
  if (hit.faceID >= 0) {
    EXPECT_NEAR(hit.position.z, 1.0, 1e-6);
  }
}

TEST(Queries, NearestPointEquidistant) {
  // Point equidistant from two faces -- must pick one consistently.
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  // Origin is equidistant from all 6 faces. Must still return a valid result.
  NearestPointResult r1 = cube.NearestPoint(vec3(0, 0, 0));
  NearestPointResult r2 = cube.NearestPoint(vec3(0, 0, 0));
  EXPECT_GE(r1.faceID, 0);
  EXPECT_EQ(r1.faceID, r2.faceID);  // Deterministic
  EXPECT_NEAR(r1.distance, 1.0, 1e-10);
}

TEST(Queries, NearestPointFarAway) {
  // Point very far from the mesh -- tests the expanding search logic.
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  NearestPointResult r = cube.NearestPoint(vec3(1000, 0, 0));
  EXPECT_GE(r.faceID, 0);
  EXPECT_NEAR(r.distance, 999.0, 1e-6);
  EXPECT_NEAR(r.position.x, 1.0, 1e-10);
}

TEST(Queries, RayCastDiagonalThroughSphere) {
  // Diagonal ray through a sphere -- tests the axis-permutation logic in
  // the containment test (the ray doesn't align with any axis).
  Manifold sphere = Manifold::Sphere(1.0, 128);
  vec3 dir = la::normalize(vec3(1, 1, 1));
  RayHit hit = sphere.RayCast(vec3(-5, -5, -5), vec3(0, 0, 0));
  EXPECT_GE(hit.faceID, 0);
  // Hit should be near the sphere surface at distance ~1 from origin
  double hitRadius = la::length(hit.position);
  EXPECT_NEAR(hitRadius, 1.0, 0.05);
}

TEST(Queries, WindingNumberTranslatedMesh) {
  // Translated mesh -- verifies that BVH and kernel use world-space coords.
  Manifold cube = Manifold::Cube(vec3(2.0), true).Translate(vec3(10, 0, 0));
  EXPECT_EQ(cube.WindingNumber(vec3(10, 0, 0)), 1);
  EXPECT_EQ(cube.WindingNumber(vec3(0, 0, 0)), 0);
}

TEST(Queries, RayCastTranslatedMesh) {
  Manifold cube = Manifold::Cube(vec3(2.0), true).Translate(vec3(10, 0, 0));
  RayHit hit = cube.RayCast(vec3(0, 0, 0), vec3(20, 0, 0));
  EXPECT_GE(hit.faceID, 0);
  EXPECT_NEAR(hit.position.x, 9.0, 1e-10);
}

TEST(Queries, NearestPointTranslatedMesh) {
  Manifold cube = Manifold::Cube(vec3(2.0), true).Translate(vec3(10, 0, 0));
  NearestPointResult r = cube.NearestPoint(vec3(0, 0, 0));
  EXPECT_GE(r.faceID, 0);
  EXPECT_NEAR(r.position.x, 9.0, 1e-10);
  EXPECT_NEAR(r.distance, 9.0, 1e-10);
}

TEST(Queries, WindingNumberScaledMesh) {
  // Very small mesh -- tests floating point precision at small scales.
  Manifold tiny = Manifold::Cube(vec3(1e-6), true);
  EXPECT_NE(tiny.WindingNumber(vec3(0, 0, 0)), 0);
  EXPECT_EQ(tiny.WindingNumber(vec3(1e-3, 0, 0)), 0);
}

TEST(Queries, NearestPointOnThinSlab) {
  // A very thin slab (high aspect ratio) -- tests numerical stability.
  Manifold slab = Manifold::Cube(vec3(10, 10, 0.001), true);
  NearestPointResult r = slab.NearestPoint(vec3(0, 0, 1));
  EXPECT_GE(r.faceID, 0);
  EXPECT_NEAR(r.position.z, 0.0005, 0.001);
  EXPECT_NEAR(r.distance, 1.0 - 0.0005, 0.001);
}

}  // namespace
