// Copyright 2021 The Manifold Authors.
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

#include "samples.h"

#ifdef MANIFOLD_CROSS_SECTION
#include "manifold/cross_section.h"
#endif
#include "test.h"

using namespace manifold;

std::vector<int> EdgePairs(const MeshGL in) {
  const int numHalfedge = 3 * in.NumTri();
  std::vector<int> edgePair(numHalfedge);

  std::map<std::pair<int, int>, int> halfedgeLink;
  for (int i = 0; i < numHalfedge; ++i) {
    std::pair<int, int> key = std::make_pair(
        in.triVerts[i], in.triVerts[(i + 1) % 3 == 0 ? i - 2 : i + 1]);
    if (key.first > key.second) std::swap(key.first, key.second);
    const auto result = halfedgeLink.emplace(std::make_pair(key, i));
    if (!result.second) {
      const int pair = result.first->second;
      edgePair[pair] = i;
      edgePair[i] = pair;
    }
  }
  return edgePair;
}

#ifdef MANIFOLD_CROSS_SECTION
// If you print this knot (with support), you can snap a half-inch marble into
// it and it'll roll around (dimensions in mm).
TEST(Samples, Knot13) {
  Manifold knot13 = TorusKnot(1, 3, 25, 10, 3.75);
  if (options.exportModels) WriteTestOBJ("knot13.obj", knot13);
  EXPECT_EQ(knot13.Genus(), 1);
  EXPECT_NEAR(knot13.Volume(), 20786, 1);
  EXPECT_NEAR(knot13.SurfaceArea(), 11177, 1);
  CheckGL(knot13);
}

// This creates two interlinked knots.
TEST(Samples, Knot42) {
  Manifold knot42 = TorusKnot(4, 2, 15, 6, 5);
  if (options.exportModels) WriteTestOBJ("knot42.obj", knot42);
  std::vector<Manifold> knots = knot42.Decompose();
  ASSERT_EQ(knots.size(), 2);
  EXPECT_EQ(knots[0].Genus(), 1);
  EXPECT_EQ(knots[1].Genus(), 1);
  EXPECT_NEAR(knots[0].Volume(), knots[1].Volume(), 1);
  EXPECT_NEAR(knots[0].SurfaceArea(), knots[1].SurfaceArea(), 1);
  CheckGL(knot42);
}
#endif

TEST(Samples, Scallop) {
  Manifold scallop = Scallop();

  if (options.exportModels) WriteTestOBJ("scallopFacets.obj", scallop);

  auto colorCurvature = [](double* newProp, vec3 pos, const double* oldProp) {
    const double curvature = oldProp[0];
    const vec3 red(1, 0, 0);
    const vec3 blue(0, 0, 1);
    const double limit = 15;
    vec3 color = la::lerp(blue, red, smoothstep(-limit, limit, curvature));
    for (const int i : {0, 1, 2}) {
      newProp[i] = color[i];
    }
  };

  scallop = scallop.Refine(50).CalculateCurvature(-1, 0).SetProperties(
      3, colorCurvature);
  EXPECT_NEAR(scallop.Volume(), 39.9, 0.1);
  EXPECT_NEAR(scallop.SurfaceArea(), 79.3, 0.1);
  EXPECT_EQ(scallop.NumVert(), scallop.NumPropVert());
  CheckGL(scallop);

  if (options.exportModels) WriteTestOBJ("scallop.obj", scallop);
}

TEST(Samples, TetPuzzle) {
  Manifold puzzle = TetPuzzle(50, 0.2, 50);
  EXPECT_LE(puzzle.NumDegenerateTris(), 2);
  CheckGL(puzzle);

  Manifold puzzle2 = puzzle.Rotate(0, 0, 180);
  EXPECT_TRUE((puzzle ^ puzzle2).IsEmpty());
  quat q = rotation_quat(normalize(vec3(1, -1, -1)), vec3(0, 0, 1));
  puzzle = puzzle.Transform({la::qmat(q), vec3()});
  if (options.exportModels) WriteTestOBJ("tetPuzzle.obj", puzzle);
}

TEST(Samples, FrameReduced) {
  Manifold frame = RoundedFrame(100, 10, 4);
  EXPECT_EQ(frame.NumDegenerateTris(), 0);
  EXPECT_EQ(frame.Genus(), 5);
  EXPECT_NEAR(frame.Volume(), 227333, 10);
  EXPECT_NEAR(frame.SurfaceArea(), 62635, 1);
  CheckGL(frame);
  if (options.exportModels) WriteTestOBJ("roundedFrameReduced.obj", frame);
}

TEST(Samples, Frame) {
  Manifold frame = RoundedFrame(100, 10);
  EXPECT_EQ(frame.NumDegenerateTris(), 0);
  EXPECT_EQ(frame.Genus(), 5);
  CheckGL(frame);
  if (options.exportModels) WriteTestOBJ("roundedFrame.obj", frame);
}

// This creates a bracelet sample which involves many operations between shapes
// that are not in general position, e.g. coplanar faces.
#ifdef MANIFOLD_CROSS_SECTION
TEST(Samples, Bracelet) {
  Manifold bracelet = StretchyBracelet();
  EXPECT_EQ(bracelet.NumDegenerateTris(), 0);
  EXPECT_EQ(bracelet.Genus(), 1);
  CheckGL(bracelet);

  CrossSection projection(bracelet.Project());
  projection = projection.Simplify(bracelet.BoundingBox().Scale() * 1e-8);
  Rect rect = projection.Bounds();
  Box box = bracelet.BoundingBox();
  EXPECT_FLOAT_EQ(rect.min.x, box.min.x);
  EXPECT_FLOAT_EQ(rect.min.y, box.min.y);
  EXPECT_FLOAT_EQ(rect.max.x, box.max.x);
  EXPECT_FLOAT_EQ(rect.max.y, box.max.y);
  EXPECT_NEAR(projection.Area(), 649, 1);
  EXPECT_EQ(projection.NumContour(), 2);
  Manifold extrusion = Manifold::Extrude(projection.ToPolygons(), 1);
  EXPECT_EQ(extrusion.NumDegenerateTris(), 0);
  EXPECT_EQ(extrusion.Genus(), 1);

  CrossSection slice(bracelet.Slice());
  EXPECT_EQ(slice.NumContour(), 2);
  EXPECT_NEAR(slice.Area(), 230.6, 0.1);
  extrusion = Manifold::Extrude(slice.ToPolygons(), 1);
  EXPECT_EQ(extrusion.Genus(), 1);

  if (options.exportModels) WriteTestOBJ("bracelet.obj", bracelet);
}

TEST(Samples, GyroidModule) {
  const double size = 20;
  Manifold gyroid = GyroidModule(size);
  EXPECT_LE(gyroid.NumDegenerateTris(), 4);
  EXPECT_EQ(gyroid.Genus(), 15);
  CheckGL(gyroid);

  const Box bounds = gyroid.BoundingBox();
  const double epsilon = gyroid.GetEpsilon();
  EXPECT_NEAR(bounds.min.z, 0, epsilon);
  EXPECT_NEAR(bounds.max.z, size * std::sqrt(2.0), epsilon);

  CrossSection slice(gyroid.Slice(5));
  EXPECT_EQ(slice.NumContour(), 4);
  EXPECT_NEAR(slice.Area(), 121.9, 0.1);
  Manifold extrusion = Manifold::Extrude(slice.ToPolygons(), 1);
  EXPECT_EQ(extrusion.Genus(), -3);

  if (options.exportModels) WriteTestOBJ("gyroidModule.obj", gyroid);
}
#endif

TEST(Samples, Sponge1) {
  Manifold sponge = MengerSponge(1);
  EXPECT_EQ(sponge.NumDegenerateTris(), 0);
  EXPECT_EQ(sponge.NumVert(), 40);
  EXPECT_EQ(sponge.Genus(), 5);
  CheckGL(sponge);
  if (options.exportModels) WriteTestOBJ("mengerSponge1.obj", sponge);
}

// This sample needs a lot of memory to run and is therefore disabled for
// emscripten.
#ifndef __EMSCRIPTEN__
// A fractal with many degenerate intersections, which also tests exact 90
// degree rotations.
#ifdef MANIFOLD_CROSS_SECTION
TEST(Samples, Sponge4) {
  Manifold sponge = MengerSponge(4);
  EXPECT_LE(sponge.NumDegenerateTris(), 8);
  EXPECT_EQ(sponge.Genus(), 26433);  // should be 1:5, 2:81, 3:1409, 4:26433
  CheckGL(sponge);

  std::pair<Manifold, Manifold> cutSponge = sponge.SplitByPlane({1, 1, 1}, 0);
  EXPECT_EQ(cutSponge.first.Genus(), 13394);
  EXPECT_EQ(cutSponge.second.Genus(), 13394);

  CrossSection projection(cutSponge.first.Project());
  projection = projection.Simplify(cutSponge.first.GetEpsilon());
  Rect rect = projection.Bounds();
  Box box = cutSponge.first.BoundingBox();
  EXPECT_EQ(rect.min.x, box.min.x);
  EXPECT_EQ(rect.min.y, box.min.y);
  EXPECT_EQ(rect.max.x, box.max.x);
  EXPECT_EQ(rect.max.y, box.max.y);
  EXPECT_NEAR(projection.Area(), 0.535, 0.001);
  Manifold extrusion = Manifold::Extrude(projection.ToPolygons(), 1);
  EXPECT_LE(extrusion.NumDegenerateTris(), 32);
  EXPECT_EQ(extrusion.Genus(), 502);

  if (options.exportModels) {
    WriteTestOBJ("mengerHalf.obj", cutSponge.first);

    const Manifold out = sponge.SetProperties(
        3, [](double* newProp, vec3 pos, const double* oldProp) {
          for (const int i : {0, 1, 2}) newProp[i] = 0.5 * (pos[i] + 0.5);
        });
    WriteTestOBJ("mengerSponge.obj", out);
  }
  Manifold sponge2 = MengerSponge(4);
  std::pair<Manifold, Manifold> cutSponge2 = sponge2.SplitByPlane({1, 1, 1}, 0);
  CheckGLEquiv(cutSponge.first.GetMeshGL(), cutSponge2.first.GetMeshGL());
}
#endif
#endif

TEST(Samples, CondensedMatter16) {
  Manifold cm = CondensedMatter(16);
  CheckGL(cm);
  Manifold cm2 = CondensedMatter(16);
  CheckGLEquiv(cm.GetMeshGL(), cm2.GetMeshGL());
  if (options.exportModels) WriteTestOBJ("condensedMatter16.obj", cm);
}

#ifndef __EMSCRIPTEN__
TEST(Samples, CondensedMatter64) {
  // FIXME: Triangulation can be invalid
  ManifoldParamGuard guard;
  ManifoldParams().processOverlaps = true;
  Manifold cm = CondensedMatter(64);
  CheckGL(cm);

  Manifold cm2 = CondensedMatter(64);
  CheckGLEquiv(cm.GetMeshGL(), cm2.GetMeshGL());
  if (options.exportModels) WriteTestOBJ("condensedMatter64.obj", cm);
}
#endif
