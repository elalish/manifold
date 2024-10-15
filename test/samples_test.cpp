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
#include "../src/utils.h"
#include "manifold/polygon.h"
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
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("knot13.glb", knot13.GetMeshGL(), {});
#endif
  EXPECT_EQ(knot13.Genus(), 1);
  auto prop = knot13.GetProperties();
  EXPECT_NEAR(prop.volume, 20786, 1);
  EXPECT_NEAR(prop.surfaceArea, 11177, 1);
  CheckGL(knot13);
}

// This creates two interlinked knots.
TEST(Samples, Knot42) {
  Manifold knot42 = TorusKnot(4, 2, 15, 6, 5);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("knot42.glb", knot42.GetMeshGL(), {});
#endif
  std::vector<Manifold> knots = knot42.Decompose();
  ASSERT_EQ(knots.size(), 2);
  EXPECT_EQ(knots[0].Genus(), 1);
  EXPECT_EQ(knots[1].Genus(), 1);
  auto prop0 = knots[0].GetProperties();
  auto prop1 = knots[1].GetProperties();
  EXPECT_NEAR(prop0.volume, prop1.volume, 1);
  EXPECT_NEAR(prop0.surfaceArea, prop1.surfaceArea, 1);
  CheckGL(knot42);
}
#endif

TEST(Samples, Scallop) {
  Manifold scallop = Scallop();

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    MeshGL in = scallop.GetMeshGL();
    std::vector<int> edgePair = EdgePairs(in);

    ExportOptions options;
    const int numVert = scallop.NumVert();
    const int numHalfedge = 3 * scallop.NumTri();
    const int numProp = in.numProp;
    for (size_t i = 0; i < scallop.NumVert(); ++i) {
      options.mat.vertColor.push_back({0, 0, 1, 1});
    }
    for (int i = 0; i < numHalfedge; ++i) {
      const int vert = in.triVerts[i];
      for (int j : {0, 1, 2}) {
        in.vertProperties.push_back(in.vertProperties[numProp * vert + j] +
                                    in.halfedgeTangent[4 * i + j] *
                                        in.halfedgeTangent[4 * i + 3]);
      }
      options.mat.vertColor.push_back({0.5, 0.5, 0, 1});
      const int j = edgePair[i % 3 == 0 ? i + 2 : i - 1];
      in.triVerts.push_back(vert);
      in.triVerts.push_back(numVert + i);
      in.triVerts.push_back(numVert + j);
    }
    options.faceted = true;
    options.mat.roughness = 0.5;
    ExportMesh("scallopFacets.glb", scallop.GetMeshGL(), options);
  }
#endif

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
  auto prop = scallop.GetProperties();
  EXPECT_NEAR(prop.volume, 39.9, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 79.3, 0.1);
  CheckGL(scallop);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    MeshGL out = scallop.GetMeshGL();
    ExportOptions options2;
    // options2.faceted = false;
    options2.mat.roughness = 0.1;
    options2.mat.metalness = 0;
    options2.mat.colorChannels = {3, 4, 5, -1};
    ExportMesh("scallop.glb", out, options2);
  }
#endif
}

TEST(Samples, TetPuzzle) {
  Manifold puzzle = TetPuzzle(50, 0.2, 50);
  EXPECT_LE(puzzle.NumDegenerateTris(), 2);
  CheckGL(puzzle);

  Manifold puzzle2 = puzzle.Rotate(0, 0, 180);
  EXPECT_TRUE((puzzle ^ puzzle2).IsEmpty());
  puzzle = puzzle.Transform(RotateUp({1, -1, -1}));
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("tetPuzzle.glb", puzzle.GetMeshGL(), {});
#endif
}

TEST(Samples, FrameReduced) {
  Manifold frame = RoundedFrame(100, 10, 4);
  EXPECT_EQ(frame.NumDegenerateTris(), 0);
  EXPECT_EQ(frame.Genus(), 5);
  auto prop = frame.GetProperties();
  EXPECT_NEAR(prop.volume, 227333, 10);
  EXPECT_NEAR(prop.surfaceArea, 62635, 1);
  CheckGL(frame);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("roundedFrameReduced.glb", frame.GetMeshGL(), {});
#endif
}

TEST(Samples, Frame) {
  Manifold frame = RoundedFrame(100, 10);
  EXPECT_EQ(frame.NumDegenerateTris(), 0);
  EXPECT_EQ(frame.Genus(), 5);
  CheckGL(frame);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("roundedFrame.glb", frame.GetMeshGL(), {});
#endif
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

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("bracelet.glb", bracelet.GetMeshGL(), {});
#endif
}

TEST(Samples, GyroidModule) {
  const double size = 20;
  Manifold gyroid = GyroidModule(size);
  EXPECT_LE(gyroid.NumDegenerateTris(), 4);
  EXPECT_EQ(gyroid.Genus(), 15);
  CheckGL(gyroid);

  const Box bounds = gyroid.BoundingBox();
  const double precision = gyroid.Precision();
  EXPECT_NEAR(bounds.min.z, 0, precision);
  EXPECT_NEAR(bounds.max.z, size * std::sqrt(2.0), precision);

  CrossSection slice(gyroid.Slice(5));
  EXPECT_EQ(slice.NumContour(), 4);
  EXPECT_NEAR(slice.Area(), 121.9, 0.1);
  Manifold extrusion = Manifold::Extrude(slice.ToPolygons(), 1);
  EXPECT_EQ(extrusion.Genus(), -3);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("gyroidModule.glb", gyroid.GetMeshGL(), {});
#endif
}
#endif

TEST(Samples, Sponge1) {
  Manifold sponge = MengerSponge(1);
  EXPECT_EQ(sponge.NumDegenerateTris(), 0);
  EXPECT_EQ(sponge.NumVert(), 40);
  EXPECT_EQ(sponge.Genus(), 5);
  CheckGL(sponge);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("mengerSponge1.glb", sponge.GetMeshGL(), {});
#endif
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
  projection = projection.Simplify(cutSponge.first.Precision());
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

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportMesh("mengerHalf.glb", cutSponge.first.GetMeshGL(), {});

    const MeshGL out = sponge.GetMeshGL();
    ExportOptions options;
    options.faceted = true;
    options.mat.roughness = 0.2;
    options.mat.metalness = 1.0;
    for (size_t i = 0; i < out.vertProperties.size(); i += out.numProp) {
      vec3 pos = {out.vertProperties[i], out.vertProperties[i + 1],
                  out.vertProperties[i + 2]};
      options.mat.vertColor.push_back(vec4(0.5 * (pos + 0.5), 1.0));
    }
    ExportMesh("mengerSponge.glb", out, options);
  }
#endif
}
#endif
#endif

TEST(Samples, CondensedMatter16) {
  Manifold cm = CondensedMatter(16);
  CheckGL(cm);
}

TEST(Samples, CondensedMatter64) {
  Manifold cm = CondensedMatter(64);
  CheckGL(cm);
}
