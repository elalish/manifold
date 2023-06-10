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

#include "polygon.h"
#include "test.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

using namespace manifold;

std::vector<int> EdgePairs(const Mesh in) {
  const int numHalfedge = 3 * in.triVerts.size();
  std::vector<int> edgePair(numHalfedge);

  std::map<std::pair<int, int>, int> halfedgeLink;
  for (int i = 0; i < numHalfedge; ++i) {
    std::pair<int, int> key = std::make_pair(in.triVerts[i / 3][i % 3],
                                             in.triVerts[i / 3][(i + 1) % 3]);
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

// If you print this knot (with support), you can snap a half-inch marble into
// it and it'll roll around (dimensions in mm).
TEST(Samples, Knot13) {
  Manifold knot13 = TorusKnot(1, 3, 25, 10, 3.75);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("knot13.glb", knot13.GetMesh(), {});
#endif
  CheckNormals(knot13);
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
  if (options.exportModels) ExportMesh("knot42.glb", knot42.GetMesh(), {});
#endif
  CheckNormals(knot42);
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

TEST(Samples, Scallop) {
  Manifold scallop = Scallop();

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    Mesh in = scallop.GetMesh();
    std::vector<int> edgePair = EdgePairs(in);

    ExportOptions options;
    const int numVert = scallop.NumVert();
    const int numHalfedge = 3 * scallop.NumTri();
    for (int i = 0; i < scallop.NumVert(); ++i) {
      options.mat.vertColor.push_back({0, 0, 1, 1});
    }
    for (int i = 0; i < numHalfedge; ++i) {
      const int vert = in.triVerts[i / 3][i % 3];
      in.vertPos.push_back(in.vertPos[vert] + glm::vec3(in.halfedgeTangent[i]) *
                                                  in.halfedgeTangent[i].w);
      options.mat.vertColor.push_back({0.5, 0.5, 0, 1});
      const int j = edgePair[i % 3 == 0 ? i + 2 : i - 1];
      in.triVerts.push_back({vert, numVert + i, numVert + j});
    }
    options.faceted = true;
    options.mat.roughness = 0.5;
    ExportMesh("scallopFacets.glb", in, options);
  }
#endif

  auto colorCurvature = [](float* newProp, glm::vec3 pos,
                           const float* oldProp) {
    const float curvature = oldProp[0];
    const glm::vec3 red(1, 0, 0);
    const glm::vec3 blue(0, 0, 1);
    const float limit = 15;
    glm::vec3 color =
        glm::mix(blue, red, glm::smoothstep(-limit, limit, curvature));
    for (const int i : {0, 1, 2}) {
      newProp[i] = color[i];
    }
  };

  scallop = scallop.Refine(50).CalculateCurvature(-1, 0).SetProperties(
      3, colorCurvature);
  CheckNormals(scallop);
  auto prop = scallop.GetProperties();
  EXPECT_NEAR(prop.volume, 41.3, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 78.1, 0.1);
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
  CheckNormals(puzzle);
  EXPECT_LE(puzzle.NumDegenerateTris(), 2);
  CheckGL(puzzle);

  Manifold puzzle2 = puzzle.Rotate(0, 0, 180);
  EXPECT_TRUE((puzzle ^ puzzle2).IsEmpty());
  puzzle = puzzle.Transform(RotateUp({1, -1, -1}));
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("tetPuzzle.glb", puzzle.GetMesh(), {});
#endif
}

TEST(Samples, FrameReduced) {
  Manifold frame = RoundedFrame(100, 10, 4);
  CheckNormals(frame);
  EXPECT_EQ(frame.NumDegenerateTris(), 0);
  EXPECT_EQ(frame.Genus(), 5);
  auto prop = frame.GetProperties();
  EXPECT_NEAR(prop.volume, 227333, 10);
  EXPECT_NEAR(prop.surfaceArea, 62635, 1);
  CheckGL(frame);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("roundedFrameReduced.glb", frame.GetMesh(), {});
#endif
}

TEST(Samples, Frame) {
  Manifold frame = RoundedFrame(100, 10);
  CheckNormals(frame);
  EXPECT_EQ(frame.NumDegenerateTris(), 0);
  EXPECT_EQ(frame.Genus(), 5);
  CheckGL(frame);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("roundedFrame.glb", frame.GetMesh(), {});
#endif
}

// This creates a bracelet sample which involves many operations between shapes
// that are not in general position, e.g. coplanar faces.
TEST(Samples, Bracelet) {
  Manifold bracelet = StretchyBracelet();
  CheckNormals(bracelet);
  EXPECT_LE(bracelet.NumDegenerateTris(), 22);
  EXPECT_EQ(bracelet.Genus(), 1);
  CheckGL(bracelet);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("bracelet.glb", bracelet.GetMesh(), {});
#endif
}

TEST(Samples, GyroidModule) {
  const float size = 20;
  Manifold gyroid = GyroidModule(size);
  CheckNormals(gyroid);
  EXPECT_LE(gyroid.NumDegenerateTris(), 2);
  EXPECT_EQ(gyroid.Genus(), 15);
  CheckGL(gyroid);

  const Box bounds = gyroid.BoundingBox();
  const float precision = gyroid.Precision();
  EXPECT_NEAR(bounds.min.z, 0, precision);
  EXPECT_NEAR(bounds.max.z, size * glm::sqrt(2.0f), precision);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("gyroidModule.gltf", gyroid.GetMesh(), {});
#endif
}

TEST(Samples, Sponge1) {
  Manifold sponge = MengerSponge(1);
  CheckNormals(sponge);
  EXPECT_EQ(sponge.NumDegenerateTris(), 0);
  EXPECT_EQ(sponge.NumVert(), 40);
  EXPECT_EQ(sponge.Genus(), 5);
  CheckGL(sponge);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("mengerSponge1.glb", sponge.GetMesh(), {});
#endif
}

// This sample needs a lot of memory to run and is therefore disabled for
// emscripten.
#ifndef __EMSCRIPTEN__
// A fractal with many degenerate intersections, which also tests exact 90
// degree rotations.
TEST(Samples, Sponge4) {
  Manifold sponge = MengerSponge(4);
  CheckNormals(sponge);
  // FIXME: limit NumDegenerateTris
  EXPECT_LE(sponge.NumDegenerateTris(), 40000);
  EXPECT_EQ(sponge.Genus(), 26433);  // should be 1:5, 2:81, 3:1409, 4:26433
  CheckGL(sponge);

  std::pair<Manifold, Manifold> cutSponge = sponge.SplitByPlane({1, 1, 1}, 0);
  EXPECT_EQ(cutSponge.first.Genus(), 13394);
  EXPECT_EQ(cutSponge.second.Genus(), 13394);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportMesh("mengerHalf.glb", cutSponge.first.GetMesh(), {});

    const Mesh out = sponge.GetMesh();
    ExportOptions options;
    options.faceted = true;
    options.mat.roughness = 0.2;
    options.mat.metalness = 1.0;
    for (const glm::vec3 pos : out.vertPos) {
      options.mat.vertColor.push_back(glm::vec4(0.5f * (pos + 0.5f), 1.0f));
    }
    ExportMesh("mengerSponge.glb", out, options);
  }
#endif
}
#endif
