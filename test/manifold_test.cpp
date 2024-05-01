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

#include "manifold.h"

#include <algorithm>

#include "cross_section.h"
#include "samples.h"
#include "sdf.h"
#include "test.h"
#include "tri_dist.h"

using namespace manifold;

/**
 * This tests that turning a mesh into a manifold and returning it to a mesh
 * produces a consistent result.
 */
TEST(Manifold, GetMesh) {
  Manifold manifold = Manifold::Sphere(1);
  Mesh mesh_out = manifold.GetMesh();
  Manifold manifold2(mesh_out);
  Mesh mesh_out2 = manifold2.GetMesh();
  Identical(mesh_out, mesh_out2);
}

TEST(Manifold, GetMeshGL) {
  Manifold manifold = Manifold::Sphere(1);
  Mesh mesh_out = manifold.GetMesh();
  MeshGL meshGL_out = manifold.GetMeshGL();
  ASSERT_EQ(meshGL_out.NumVert(), mesh_out.vertPos.size());
  ASSERT_EQ(meshGL_out.NumTri(), mesh_out.triVerts.size());
  for (int i = 0; i < meshGL_out.NumVert(); ++i) {
    for (const int j : {0, 1, 2}) {
      ASSERT_EQ(meshGL_out.vertProperties[3 * i + j], mesh_out.vertPos[i][j]);
    }
  }
  for (int i = 0; i < meshGL_out.NumTri(); ++i) {
    for (const int j : {0, 1, 2})
      ASSERT_EQ(meshGL_out.triVerts[3 * i + j], mesh_out.triVerts[i][j]);
  }
}

TEST(Manifold, Empty) {
  Mesh emptyMesh;
  Manifold empty(emptyMesh);

  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_EQ(empty.Status(), Manifold::Error::NoError);
}

TEST(Manifold, ValidInput) {
  std::vector<float> propTol = {0.1, 0.2};
  MeshGL tetGL = TetGL();
  Manifold tet(tetGL, propTol);
  EXPECT_FALSE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NoError);
}

TEST(Manifold, InvalidInput1) {
  Mesh in = Tet();
  in.vertPos[2][1] = NAN;
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, InvalidInput2) {
  Mesh in = Tet();
  std::swap(in.triVerts[2][1], in.triVerts[2][2]);
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NotManifold);
}

TEST(Manifold, InvalidInput3) {
  Mesh in = Tet();
  for (glm::ivec3& tri : in.triVerts) {
    for (int i : {0, 1, 2}) {
      if (tri[i] == 2) tri[i] = -2;
    }
  }
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

TEST(Manifold, InvalidInput4) {
  Mesh in = Tet();
  for (glm::ivec3& tri : in.triVerts) {
    for (int i : {0, 1, 2}) {
      if (tri[i] == 2) tri[i] = 4;
    }
  }
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

TEST(Manifold, InvalidInput5) {
  MeshGL tetGL = TetGL();
  tetGL.mergeFromVert[tetGL.mergeFromVert.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::MergeIndexOutOfBounds);
}

TEST(Manifold, InvalidInput7) {
  MeshGL tetGL = TetGL();
  tetGL.triVerts[tetGL.triVerts.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

/**
 * ExpectMeshes performs a decomposition, so this test ensures that compose and
 * decompose are inverse operations.
 */
TEST(Manifold, Decompose) {
  std::vector<Manifold> manifoldList;
  manifoldList.emplace_back(Manifold::Tetrahedron());
  manifoldList.emplace_back(Manifold::Cube());
  manifoldList.emplace_back(Manifold::Sphere(1, 4));
  Manifold manifolds = Manifold::Compose(manifoldList);

  ExpectMeshes(manifolds, {{8, 12}, {6, 8}, {4, 4}});

  std::vector<MeshGL> input;

  for (const Manifold& manifold : manifoldList) {
    EXPECT_GE(manifold.OriginalID(), 0);
    input.emplace_back(manifold.GetMeshGL());
  }

  RelatedGL(manifolds, input);
}

TEST(Manifold, DecomposeProps) {
  std::vector<MeshGL> input;
  std::vector<Manifold> manifoldList;
  auto tet = WithPositionColors(Manifold::Tetrahedron());
  manifoldList.emplace_back(tet);
  input.emplace_back(tet);
  auto cube = WithPositionColors(Manifold::Cube());
  manifoldList.emplace_back(cube);
  input.emplace_back(cube);
  auto sphere = WithPositionColors(Manifold::Sphere(1, 4));
  manifoldList.emplace_back(sphere);
  input.emplace_back(sphere);
  Manifold manifolds = Manifold::Compose(manifoldList);

  ExpectMeshes(manifolds, {{8, 12, 3}, {6, 8, 3}, {4, 4, 3}});

  RelatedGL(manifolds, input);

  for (const Manifold& manifold : manifolds.Decompose()) {
    RelatedGL(manifold, input);
  }
}

/**
 * These tests check the various manifold constructors.
 */
TEST(Manifold, Sphere) {
  int n = 25;
  Manifold sphere = Manifold::Sphere(1.0f, 4 * n);
  EXPECT_EQ(sphere.NumTri(), n * n * 8);
}

TEST(Manifold, Normals) {
  Mesh cube = Manifold::Cube(glm::vec3(1), true).GetMesh();
  const int nVert = cube.vertPos.size();
  for (int i = 0; i < nVert; ++i) {
    glm::vec3 v = glm::normalize(cube.vertPos[i]);
    glm::vec3& n = cube.vertNormal[i];
    EXPECT_FLOAT_EQ(v.x, n.x);
    EXPECT_FLOAT_EQ(v.y, n.y);
    EXPECT_FLOAT_EQ(v.z, n.z);
  }
}

TEST(Manifold, Extrude) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0f, 3);
  EXPECT_EQ(donut.Genus(), 1);
  auto prop = donut.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 12.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 48.0f);
}

TEST(Manifold, ExtrudeCone) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0f, 0, 0, glm::vec2(0.0f));
  EXPECT_EQ(donut.Genus(), 0);
  EXPECT_FLOAT_EQ(donut.GetProperties().volume, 4.0f);
}

Polygons RotatePolygons(Polygons polys, const int index) {
  Polygons rotatedPolys;
  for (auto& polygon : polys) {
    auto rotatedPolygon = polygon;
    std::rotate(rotatedPolygon.begin(), rotatedPolygon.begin() + index,
                rotatedPolygon.end());
    rotatedPolys.push_back(rotatedPolygon);
  }
  return rotatedPolys;
}

TEST(Manifold, Revolve) {
  Polygons polys = SquareHole();
  Manifold vug;
  for (int i = 0; i < polys[0].size(); i++) {
    Polygons rotatedPolys = RotatePolygons(polys, i);
    vug = Manifold::Revolve(rotatedPolys, 48);
    EXPECT_EQ(vug.Genus(), -1);
    auto prop = vug.GetProperties();
    EXPECT_NEAR(prop.volume, 14.0f * glm::pi<float>(), 0.2f);
    EXPECT_NEAR(prop.surfaceArea, 30.0f * glm::pi<float>(), 0.2f);
  }
}

TEST(Manifold, Revolve2) {
  Polygons polys = SquareHole(2.0f);
  Manifold donutHole = Manifold::Revolve(polys, 48);
  EXPECT_EQ(donutHole.Genus(), 0);
  auto prop = donutHole.GetProperties();
  EXPECT_NEAR(prop.volume, 48.0f * glm::pi<float>(), 1.0f);
  EXPECT_NEAR(prop.surfaceArea, 96.0f * glm::pi<float>(), 1.0f);
}

TEST(Manifold, Revolve3) {
  CrossSection circle = CrossSection::Circle(1, 32);
  Manifold sphere = Manifold::Revolve(circle, 32);
  auto prop = sphere.GetProperties();
  EXPECT_NEAR(prop.volume, 4.0f / 3.0f * glm::pi<float>(), 0.1);
  EXPECT_NEAR(prop.surfaceArea, 4 * glm::pi<float>(), 0.15);
}

TEST(Manifold, PartialRevolveOnYAxis) {
  Polygons polys = SquareHole(2.0f);
  Polygons offsetPolys = SquareHole(10.0f);

  Manifold revolute;
  for (int i = 0; i < polys[0].size(); i++) {
    Polygons rotatedPolys = RotatePolygons(polys, i);
    revolute = Manifold::Revolve(rotatedPolys, 48, 180);
    EXPECT_EQ(revolute.Genus(), 1);
    auto prop = revolute.GetProperties();
    EXPECT_NEAR(prop.volume, 24.0f * glm::pi<float>(), 1.0f);
    EXPECT_NEAR(
        prop.surfaceArea,
        48.0f * glm::pi<float>() + 4.0f * 4.0f * 2.0f - 2.0f * 2.0f * 2.0f,
        1.0f);
  }
}

TEST(Manifold, PartialRevolveOffset) {
  Polygons polys = SquareHole(10.0f);

  Manifold revolute;
  for (int i = 0; i < polys[0].size(); i++) {
    Polygons rotatedPolys = RotatePolygons(polys, i);
    revolute = Manifold::Revolve(rotatedPolys, 48, 180);
    auto prop = revolute.GetProperties();
    EXPECT_EQ(revolute.Genus(), 1);
    EXPECT_NEAR(prop.surfaceArea, 777.0f, 1.0f);
    EXPECT_NEAR(prop.volume, 376.0f, 1.0f);
  }
}

TEST(Manifold, Warp) {
  CrossSection square = CrossSection::Square({1, 1});
  Manifold shape = Manifold::Extrude(square, 2, 10).Warp([](glm::vec3& v) {
    v.x += v.z * v.z;
  });
  auto propBefore = shape.GetProperties();

  Manifold simplified = Manifold::Compose({shape});
  auto propAfter = simplified.GetProperties();

  EXPECT_NEAR(propBefore.volume, propAfter.volume, 0.0001);
  EXPECT_NEAR(propBefore.surfaceArea, propAfter.surfaceArea, 0.0001);
  EXPECT_NEAR(propBefore.volume, 2, 0.0001);
}

TEST(Manifold, Warp2) {
  CrossSection circle =
      CrossSection::Circle(5, 20).Translate(glm::vec2(10.0, 10.0));

  Manifold shape = Manifold::Extrude(circle, 2, 10).Warp([](glm::vec3& v) {
    int nSegments = 10;
    double angleStep = 2.0 / 3.0 * glm::pi<float>() / nSegments;
    int zIndex = nSegments - 1 - std::round(v.z);
    double angle = zIndex * angleStep;
    v.z = v.y;
    v.y = v.x * sin(angle);
    v.x = v.x * cos(angle);
  });

  auto propBefore = shape.GetProperties();

  Manifold simplified = Manifold::Compose({shape});
  auto propAfter = simplified.GetProperties();

  EXPECT_NEAR(propBefore.volume, propAfter.volume, 0.0001);
  EXPECT_NEAR(propBefore.surfaceArea, propAfter.surfaceArea, 0.0001);
  EXPECT_NEAR(propBefore.volume, 321, 1);
}

TEST(Manifold, WarpBatch) {
  Manifold shape1 =
      Manifold::Cube({2, 3, 4}).Warp([](glm::vec3& v) { v.x += v.z * v.z; });
  auto prop1 = shape1.GetProperties();

  Manifold shape2 =
      Manifold::Cube({2, 3, 4}).WarpBatch([](VecView<glm::vec3> vecs) {
        for (glm::vec3& v : vecs) {
          v.x += v.z * v.z;
        }
      });
  auto prop2 = shape2.GetProperties();

  EXPECT_EQ(prop1.volume, prop2.volume);
  EXPECT_EQ(prop1.surfaceArea, prop2.surfaceArea);
}

glm::vec4 CircularTangent(const glm::vec3& tangent, const glm::vec3& edgeVec) {
  const glm::vec3 dir = glm::normalize(tangent);

  float weight = glm::abs(glm::dot(dir, glm::normalize(edgeVec)));
  if (weight == 0) {
    weight = 1;
  }
  // Quadratic weighted bezier for circular interpolation
  const glm::vec4 bz2 =
      weight * glm::vec4(dir * glm::length(edgeVec) / (2 * weight), 1);
  // Equivalent cubic weighted bezier
  const glm::vec4 bz3 = glm::mix(glm::vec4(0, 0, 0, 1), bz2, 2 / 3.0f);
  // Convert from homogeneous form to geometric form
  return glm::vec4(glm::vec3(bz3) / bz3.w, bz3.w);
}

TEST(Manifold, SmoothTorus) {
  Mesh torusMesh =
      Manifold::Revolve(CrossSection::Circle(1, 8).Translate({2, 0}), 6)
          .GetMesh();
  const int numTri = torusMesh.triVerts.size();

  // Create correct toroidal halfedge tangents - SmoothOut() is too generic to
  // do this perfectly.
  torusMesh.halfedgeTangent.resize(3 * numTri);
  for (int tri = 0; tri < numTri; ++tri) {
    for (const int i : {0, 1, 2}) {
      glm::vec4& tangent = torusMesh.halfedgeTangent[3 * tri + i];
      const glm::vec3 v = torusMesh.vertPos[torusMesh.triVerts[tri][i]];
      const glm::vec3 edge =
          torusMesh.vertPos[torusMesh.triVerts[tri][(i + 1) % 3]] - v;
      if (edge.z == 0) {
        glm::vec3 tan(v.y, -v.x, 0);
        tan *= glm::sign(glm::dot(tan, edge));
        tangent = CircularTangent(tan, edge);
      } else if (glm::abs(glm::determinant(
                     glm::mat2(glm::vec2(v), glm::vec2(edge)))) < kTolerance) {
        const float theta = glm::asin(v.z);
        glm::vec2 xy(v);
        const float r = glm::length(xy);
        xy = xy / r * v.z * (r > 2 ? -1.0f : 1.0f);
        glm::vec3 tan(xy.x, xy.y, glm::cos(theta));
        tan *= glm::sign(glm::dot(tan, edge));
        tangent = CircularTangent(tan, edge);
      } else {
        tangent = {0, 0, 0, -1};
      }
    }
  }

  Manifold smooth = Manifold(torusMesh).RefineToLength(0.1).CalculateNormals(0);
  Mesh out = smooth.GetMesh();
  for (glm::vec3 v : out.vertPos) {
    glm::vec3 p(v.x, v.y, 0);
    p = glm::normalize(p) * 2.0f;
    float r = glm::length(v - p);
    ASSERT_NEAR(r, 1, 0.005);
  }

#ifdef MANIFOLD_EXPORT
  ExportOptions options2;
  options2.faceted = false;
  options2.mat.normalChannels = {3, 4, 5};
  options2.mat.roughness = 0;
  if (options.exportModels) ExportMesh("smoothTorus.glb", out, options2);
#endif
}

TEST(Manifold, SineSurface) {
  MeshGL surface = LevelSet(
      [](glm::vec3 p) {
        float mid = glm::sin(p.x) + glm::sin(p.y);
        return (p.z > mid - 0.5 && p.z < mid + 0.5) ? 1 : -1;
      },
      {glm::vec3(-2 * glm::pi<float>() + 0.2),
       glm::vec3(0 * glm::pi<float>() - 0.2)},
      1);
  Manifold smoothed = Manifold(surface).SmoothOut(50).Refine(8);
  auto prop = smoothed.GetProperties();
  EXPECT_NEAR(prop.volume, 8.08, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 30.85, 0.01);
  EXPECT_EQ(smoothed.Genus(), 0);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportOptions options2;
    // options2.mat.colorChannels = {3, 4, 5, -1};
    ExportMesh("sinesurface.glb", smoothed.GetMeshGL(), options2);
  }
#endif
}