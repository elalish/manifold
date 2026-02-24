// Copyright 2025 The Manifold Authors.
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

using namespace manifold;

namespace {

// Check if a manifold is approximately convex by comparing volume to hull
// volume.
bool IsApproxConvex(const Manifold& m, double tol = 0.001) {
  double vol = m.Volume();
  double hullVol = m.Hull().Volume();
  return vol > 0 && std::abs(hullVol - vol) < vol * tol;
}

// Union volume of all pieces (checks exact tiling, not sum which
// double-counts overlaps at shared tet faces).
double UnionVolume(const std::vector<Manifold>& pieces) {
  return Manifold::BatchBoolean(pieces, OpType::Add).Volume();
}

// Make an L-shaped solid (cube minus corner cube).
Manifold MakeLShape() {
  return Manifold::Cube({2, 2, 2}) -
         Manifold::Cube({1, 1, 2}).Translate({1, 1, 0});
}

}  // namespace

TEST(ConvexDecomposition, AlreadyConvex) {
  Manifold cube = Manifold::Cube({2, 2, 2});
  auto pieces = cube.ConvexDecomposition();
  EXPECT_EQ(pieces.size(), 1);
  if (!pieces.empty()) {
    EXPECT_NEAR(pieces[0].Volume(), cube.Volume(), cube.Volume() * 0.01);
  }
}

TEST(ConvexDecomposition, AlreadyConvexSphere) {
  Manifold sphere = Manifold::Sphere(1.0, 32);
  auto pieces = sphere.ConvexDecomposition();
  EXPECT_EQ(pieces.size(), 1);
  if (!pieces.empty()) {
    EXPECT_NEAR(pieces[0].Volume(), sphere.Volume(), sphere.Volume() * 0.01);
  }
}

TEST(ConvexDecomposition, LShape) {
  Manifold shape = MakeLShape();
  double origVol = shape.Volume();

  auto pieces = shape.ConvexDecomposition();
  EXPECT_GE(pieces.size(), 2u);
  EXPECT_LE(pieces.size(), 5u);

  for (const auto& p : pieces) {
    EXPECT_TRUE(IsApproxConvex(p))
        << "Non-convex piece with volume " << p.Volume();
  }

  double totalVol = UnionVolume(pieces);
  EXPECT_NEAR(totalVol, origVol, origVol * 1e-4);
}

TEST(ConvexDecomposition, CubeSphere) {
  Manifold shape = Manifold::Cube({2, 2, 2}) - Manifold::Sphere(0.8, 32);
  double origVol = shape.Volume();

  auto pieces = shape.ConvexDecomposition();
  EXPECT_GE(pieces.size(), 2u);

  double totalVol = UnionVolume(pieces);
  EXPECT_NEAR(totalVol, origVol, origVol * 1e-4);

  int convexCount = 0;
  for (const auto& p : pieces)
    if (IsApproxConvex(p)) convexCount++;
  EXPECT_GE(convexCount, (int)pieces.size() / 2);
}

TEST(ConvexDecomposition, TwoSpheres) {
  Manifold shape = Manifold::Sphere(1.0, 24) +
                   Manifold::Sphere(1.0, 24).Translate({1.5, 0, 0});
  double origVol = shape.Volume();

  auto pieces = shape.ConvexDecomposition();
  // Pre-pass detects the coplanar reflex ring and splits perfectly into 2.
  EXPECT_EQ(pieces.size(), 2u);

  for (const auto& p : pieces) {
    EXPECT_TRUE(IsApproxConvex(p))
        << "Non-convex piece with volume " << p.Volume();
  }

  double totalVol = UnionVolume(pieces);
  EXPECT_NEAR(totalVol, origVol, origVol * 1e-4);
}

TEST(ConvexDecomposition, ThreeSpheres) {
  Manifold shape = Manifold::Sphere(1.0, 24) +
                   Manifold::Sphere(1.0, 24).Translate({1.5, 0, 0}) +
                   Manifold::Sphere(1.0, 24).Translate({0.75, 1.3, 0});
  double origVol = shape.Volume();

  auto pieces = shape.ConvexDecomposition();
  EXPECT_GE(pieces.size(), 2u);

  // Curved shapes lose small volume (~0.002%) from convex hulling of DT
  // pieces on curved surfaces (chord vs arc).
  double totalVol = UnionVolume(pieces);
  EXPECT_NEAR(totalVol, origVol, origVol * 0.005);

  int convexCount = 0;
  for (const auto& p : pieces)
    if (IsApproxConvex(p)) convexCount++;
  EXPECT_GE(convexCount, (int)pieces.size() / 2);
}

TEST(ConvexDecomposition, CubeCube) {
  Manifold shape = Manifold::Cube({2, 2, 2}) -
                   Manifold::Cube({1, 1, 1}).Translate({0.5, 0.5, 0});
  double origVol = shape.Volume();

  auto pieces = shape.ConvexDecomposition();
  EXPECT_GE(pieces.size(), 2u);
  EXPECT_LE(pieces.size(), 15u);

  for (const auto& p : pieces) {
    EXPECT_TRUE(IsApproxConvex(p))
        << "Non-convex piece with volume " << p.Volume();
  }

  double totalVol = UnionVolume(pieces);
  EXPECT_NEAR(totalVol, origVol, origVol * 1e-4);
}

TEST(ConvexDecomposition, ClusterSizeEffect) {
  Manifold shape = MakeLShape();

  auto pieces2 = shape.ConvexDecomposition(2);
  auto pieces4 = shape.ConvexDecomposition(4);

  // Both should produce valid decompositions
  EXPECT_GE(pieces2.size(), 2u);
  EXPECT_GE(pieces4.size(), 2u);
  // Higher cluster size generally produces fewer or equal pieces,
  // but parallel floating-point variability can cause ±1 difference
  EXPECT_LE((int)pieces4.size(), (int)pieces2.size() + 1);
}

TEST(ConvexDecomposition, EmptyManifold) {
  Manifold empty;
  auto pieces = empty.ConvexDecomposition();
  EXPECT_EQ(pieces.size(), 0u);
}

// Long skinny shapes stress the DT with near-degenerate tets.
TEST(ConvexDecomposition, ThinBeam) {
  // 10:1 aspect ratio beam with an L-bend
  Manifold beam = Manifold::Cube({10, 1, 1});
  Manifold shape = beam + beam.Rotate(0, 0, 90).Translate({0, 10, 0});
  double origVol = shape.Volume();

  auto pieces = shape.ConvexDecomposition();
  EXPECT_GE(pieces.size(), 2u);

  double totalVol = UnionVolume(pieces);
  EXPECT_NEAR(totalVol, origVol, origVol * 1e-4);
}

TEST(ConvexDecomposition, ThinWedge) {
  // Very thin wedge: cube sliced by a near-grazing plane
  Manifold cube = Manifold::Cube({4, 4, 4}, true);
  Manifold wedge = cube.TrimByPlane({0.1, 1, 0}, 0.5);
  double origVol = wedge.Volume();

  auto pieces = wedge.ConvexDecomposition();
  EXPECT_GE(pieces.size(), 1u);

  double totalVol = UnionVolume(pieces);
  EXPECT_NEAR(totalVol, origVol, origVol * 1e-4);

  for (const auto& p : pieces) {
    EXPECT_TRUE(IsApproxConvex(p))
        << "Non-convex piece with volume " << p.Volume();
  }
}

TEST(ConvexDecomposition, FlatSlab) {
  // 20:20:1 slab with 5 notches on each side, rotated 90 degrees,
  // creating long skinny triangular regions where they cross.
  Manifold slab = Manifold::Cube({20, 20, 1});
  // Top notches: long slots running in X direction
  for (int i = 0; i < 5; i++) {
    slab -= Manifold::Cube({14, 1, 0.6}).Translate({3, 2.0 + i * 3.5, 0.4});
  }
  // Bottom notches: long slots running in Y direction (rotated 90)
  for (int i = 0; i < 5; i++) {
    slab -= Manifold::Cube({1, 14, 0.6}).Translate({2.0 + i * 3.5, 3, 0});
  }
  double origVol = slab.Volume();

  auto pieces = slab.ConvexDecomposition();
  EXPECT_GE(pieces.size(), 2u);

  double totalVol = UnionVolume(pieces);
  EXPECT_NEAR(totalVol, origVol, origVol * 1e-4);
}

TEST(ConvexDecomposition, HighAspectCylinder) {
  // Tall thin cylinder (20:1 aspect ratio) with a chunk removed
  Manifold cyl = Manifold::Cylinder(20, 1, 1, 24);
  Manifold shape = cyl - Manifold::Cube({2, 2, 5}).Translate({-1, -1, 7.5});
  double origVol = shape.Volume();

  auto pieces = shape.ConvexDecomposition();
  EXPECT_GE(pieces.size(), 2u);

  double totalVol = UnionVolume(pieces);
  EXPECT_NEAR(totalVol, origVol, origVol * 1e-4);

  int convexCount = 0;
  for (const auto& p : pieces)
    if (IsApproxConvex(p)) convexCount++;
  EXPECT_GE(convexCount, (int)pieces.size() / 2);
}

TEST(ConvexDecomposition, ThinFin) {
  // Very thin fin: 0.1 thick, 10 wide, 5 tall, with a V-notch
  Manifold fin = Manifold::Cube({10, 0.1, 5});
  Manifold notch = Manifold::Cube({3, 0.1, 3}).Translate({3.5, 0, 0});
  Manifold shape = fin - notch;
  double origVol = shape.Volume();

  auto pieces = shape.ConvexDecomposition();
  EXPECT_GE(pieces.size(), 2u);

  double totalVol = UnionVolume(pieces);
  EXPECT_NEAR(totalVol, origVol, origVol * 1e-4);
}

TEST(ConvexDecomposition, MaxDepth) {
  // Higher maxDepth should produce same or more convex pieces
  Manifold shape = MakeLShape();
  auto d0 = shape.ConvexDecomposition(2, 0);
  auto d3 = shape.ConvexDecomposition(2, 3);
  EXPECT_GE(d0.size(), 1u);
  EXPECT_GE(d3.size(), 1u);
  double v0 = UnionVolume(d0);
  double v3 = UnionVolume(d3);
  EXPECT_NEAR(v0, shape.Volume(), shape.Volume() * 1e-4);
  EXPECT_NEAR(v3, shape.Volume(), shape.Volume() * 1e-4);
}

TEST(ConvexDecomposition, SingleTetrahedron) {
  // Tetrahedron is already convex — tests the early exit path
  Manifold tet = Manifold::Tetrahedron();
  auto pieces = tet.ConvexDecomposition();
  EXPECT_EQ(pieces.size(), 1u);
  EXPECT_NEAR(pieces[0].Volume(), tet.Volume(), tet.Volume() * 1e-4);
}

TEST(ConvexDecomposition, MultipleComponents) {
  // Two disjoint cubes — tests Decompose() path
  Manifold shape = Manifold::Cube({1, 1, 1}) +
                   Manifold::Cube({1, 1, 1}).Translate({5, 0, 0});
  auto pieces = shape.ConvexDecomposition();
  EXPECT_EQ(pieces.size(), 2u);
}

TEST(ConvexDecomposition, Deterministic) {
  // Cube is fully deterministic (fast-path convexity check)
  Manifold cube = Manifold::Cube({2, 2, 2});
  auto p1 = cube.ConvexDecomposition();
  auto p2 = cube.ConvexDecomposition();
  EXPECT_EQ(p1.size(), p2.size());

  // L-shape: piece counts should be consistent (within ±1 due to
  // parallel floating-point variability in boolean operations)
  Manifold shape = MakeLShape();
  auto pieces1 = shape.ConvexDecomposition();
  auto pieces2 = shape.ConvexDecomposition();
  EXPECT_LE(std::abs((int)pieces1.size() - (int)pieces2.size()), 1);
}
