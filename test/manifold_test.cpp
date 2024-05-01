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

namespace {

using namespace manifold;

template <typename T>
int NumUnique(const std::vector<T>& in) {
  std::set<int> unique;
  for (const T& v : in) {
    unique.emplace(v);
  }
  return unique.size();
}

}  // namespace

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

TEST(Manifold, Smooth) {
  Manifold tet = Manifold::Tetrahedron();
  Manifold smooth = Manifold::Smooth(tet.GetMesh());
  int n = 100;
  smooth = smooth.Refine(n);
  ExpectMeshes(smooth, {{2 * n * n + 2, 4 * n * n}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 18.7, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 34.5, 0.1);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("smoothTet.glb", smooth.GetMesh(), {});
#endif
}

#ifdef MANIFOLD_EXPORT
TEST(Manifold, HullFail) {
  Manifold body = ReadMesh("hull-body.glb");
  Manifold mask = ReadMesh("hull-mask.glb");
  Manifold ret = body - mask;
  MeshGL mesh = ret.GetMesh();
}
#endif

TEST(Manifold, RefineQuads) {
  Manifold cylinder =
      Manifold(WithPositionColors(Manifold::Cylinder(2, 1, -1, 12).SmoothOut()))
          .RefineToLength(0.05);
  EXPECT_EQ(cylinder.NumTri(), 16892);
  auto prop = cylinder.GetProperties();
  EXPECT_NEAR(prop.volume, 2 * glm::pi<float>(), 0.003);
  EXPECT_NEAR(prop.surfaceArea, 6 * glm::pi<float>(), 0.004);
  const MeshGL out = cylinder.GetMeshGL();
  CheckGL(out);

  const MeshGL baseline = WithPositionColors(cylinder);
  float maxDiff = 0;
  for (int i = 0; i < out.vertProperties.size(); ++i) {
    maxDiff = glm::max(
        maxDiff, glm::abs(out.vertProperties[i] - baseline.vertProperties[i]));
  }
  // This has a wide tolerance because the triangle colors on the ends are still
  // being stretched out into circular arcs, which introduces unavoidable error.
  EXPECT_LE(maxDiff, 0.07);

#ifdef MANIFOLD_EXPORT
  ExportOptions options2;
  options2.mat.metalness = 0;
  options2.mat.roughness = 0.5;
  options2.mat.colorChannels = {3, 4, 5, -1};
  if (options.exportModels) ExportMesh("refinedCylinder.glb", out, options2);
#endif
}

TEST(Manifold, SmoothFlat) {
  Manifold cone = Manifold::Cylinder(5, 10, 5, 12).SmoothOut();
  Manifold smooth = cone.RefineToLength(0.5).CalculateNormals(0);
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 1165.77, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 769.88, 0.01);
  MeshGL out = smooth.GetMeshGL();
  CheckGL(out);

#ifdef MANIFOLD_EXPORT
  ExportOptions options2;
  options2.faceted = false;
  options2.mat.normalChannels = {3, 4, 5};
  options2.mat.roughness = 0;
  if (options.exportModels) ExportMesh("smoothCone.glb", out, options2);
#endif
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

TEST(Manifold, Smooth2Length) {
  Manifold cone = Manifold::Extrude(
      CrossSection::Circle(10, 10).Translate({10, 0}), 2, 0, 0, {0, 0});
  cone += cone.Scale({1, 1, -5});
  Manifold smooth = Manifold::Smooth(cone.GetMesh());
  smooth = smooth.RefineToLength(0.1);
  ExpectMeshes(smooth, {{85250, 170496}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 4842, 1);
  EXPECT_NEAR(prop.surfaceArea, 1400, 1);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("smoothCones.glb", smooth.GetMesh(), {});
#endif
}

TEST(Manifold, SmoothSphere) {
  int n[5] = {4, 8, 16, 32, 64};
  float precision[5] = {0.006, 0.003, 0.003, 0.0005, 0.00006};
  for (int i = 0; i < 5; ++i) {
    Manifold sphere = Manifold::Sphere(1, n[i]);
    // Refine(3*x) puts a center point in the triangle, which is the worst
    // case.
    Manifold smoothed = Manifold::Smooth(sphere.GetMesh()).Refine(6);
    Mesh out = smoothed.GetMesh();
    auto bounds =
        std::minmax_element(out.vertPos.begin(), out.vertPos.end(),
                            [](const glm::vec3& a, const glm::vec3& b) {
                              return glm::dot(a, a) < glm::dot(b, b);
                            });
    float min = glm::length(*bounds.first);
    float max = glm::length(*bounds.second);
    EXPECT_NEAR(min, 1, precision[i]);
    EXPECT_NEAR(max, 1, precision[i]);
  }
}

TEST(Manifold, SmoothNormals) {
  Manifold cylinder = Manifold::Cylinder(10, 5, 5, 8);
  Manifold out = cylinder.SmoothOut().RefineToLength(0.1);
  Manifold byNormals =
      cylinder.CalculateNormals(0).SmoothByNormals(0).RefineToLength(0.1);
  auto outProp = out.GetProperties();
  auto byNormalsProp = byNormals.GetProperties();
  EXPECT_FLOAT_EQ(outProp.volume, byNormalsProp.volume);
  EXPECT_FLOAT_EQ(outProp.surfaceArea, byNormalsProp.surfaceArea);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("smoothCylinder.glb", byNormals.GetMesh(), {});
#endif
}

TEST(Manifold, ManualSmooth) {
  // Unit Octahedron
  const Mesh oct = Manifold::Sphere(1, 4).GetMesh();
  Mesh smooth = Manifold::Smooth(oct).GetMesh();
  // Sharpen the edge from vert 4 to 5
  smooth.halfedgeTangent[6].w = 0;
  smooth.halfedgeTangent[22].w = 0;
  smooth.halfedgeTangent[16].w = 0;
  smooth.halfedgeTangent[18].w = 0;
  Manifold interp(smooth);
  interp = interp.Refine(100);

  ExpectMeshes(interp, {{40002, 80000}});
  auto prop = interp.GetProperties();
  EXPECT_NEAR(prop.volume, 3.83, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 11.95, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    interp = interp.CalculateCurvature(-1, 0).SetProperties(
        3, [](float* newProp, glm::vec3 pos, const float* oldProp) {
          const glm::vec3 red(1, 0, 0);
          const glm::vec3 purple(1, 0, 1);
          glm::vec3 color =
              glm::mix(purple, red, glm::smoothstep(0.0f, 2.0f, oldProp[0]));
          for (const int i : {0, 1, 2}) newProp[i] = color[i];
        });
    const MeshGL out = interp.GetMeshGL();
    ExportOptions options;
    options.mat.roughness = 0.1;
    options.mat.colorChannels = {3, 4, 5, -1};
    ExportMesh("sharpenedSphere.glb", out, options);
  }
#endif
}

TEST(Manifold, SmoothMirrored) {
  const Mesh tet = Manifold::Tetrahedron().Scale({1, 2, 3}).GetMesh();
  Manifold smooth = Manifold::Smooth(tet);
  Manifold mirror = smooth.Scale({-2, 2, 2}).Refine(10);
  smooth = smooth.Refine(10).Scale({2, 2, 2});

  auto prop0 = smooth.GetProperties();
  auto prop1 = mirror.GetProperties();
  EXPECT_NEAR(prop0.volume, prop1.volume, 0.1);
  EXPECT_NEAR(prop0.surfaceArea, prop1.surfaceArea, 0.1);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("mirrored.glb", mirror.GetMesh(), {});
#endif
}

TEST(Manifold, Csaszar) {
  Manifold csaszar = Manifold::Smooth(Csaszar());
  csaszar = csaszar.Refine(100);
  ExpectMeshes(csaszar, {{70000, 140000}});
  auto prop = csaszar.GetProperties();
  EXPECT_NEAR(prop.volume, 89873, 10);
  EXPECT_NEAR(prop.surfaceArea, 15301, 10);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    const Mesh out = csaszar.GetMesh();
    ExportOptions options;
    options.faceted = false;
    options.mat.roughness = 0.1;

    options.mat.vertColor.resize(csaszar.NumVert());
    const glm::vec4 blue(0, 0, 1, 1);
    const glm::vec4 yellow(1, 1, 0, 1);
    for (int tri = 0; tri < csaszar.NumTri(); ++tri) {
      for (int i : {0, 1, 2}) {
        const glm::vec3& uvw = {0.5, 0.5, 0.0};
        const float alpha = glm::min(uvw[0], glm::min(uvw[1], uvw[2]));
        options.mat.vertColor[out.triVerts[tri][i]] =
            glm::mix(yellow, blue, glm::smoothstep(0.0f, 0.2f, alpha));
      }
    }
    ExportMesh("smoothCsaszar.glb", out, options);
  }
#endif
}

/**
 * These tests verify the calculation of a manifold's geometric properties.
 */
TEST(Manifold, GetProperties) {
  Manifold cube = Manifold::Cube();
  auto prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0f);

  cube = cube.Scale(glm::vec3(-1.0f));
  prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0f);
}

TEST(Manifold, Precision) {
  Manifold cube = Manifold::Cube();
  EXPECT_FLOAT_EQ(cube.Precision(), kTolerance);
  cube = cube.Scale({0.1, 1, 10});
  EXPECT_FLOAT_EQ(cube.Precision(), 10 * kTolerance);
  cube = cube.Translate({-100, -10, -1});
  EXPECT_FLOAT_EQ(cube.Precision(), 100 * kTolerance);
}

TEST(Manifold, Precision2) {
  Manifold cube = Manifold::Cube();
  cube = cube.Translate({-0.5, 0, 0}).Scale({2, 1, 1});
  EXPECT_FLOAT_EQ(cube.Precision(), 2 * kTolerance);
}

TEST(Manifold, Precision3) {
  Manifold cylinder = Manifold::Cylinder(1, 1, 1, 1000);
  const auto prop = cylinder.GetProperties();

  MeshGL mesh = cylinder.GetMeshGL();
  mesh.precision = 0.001;
  mesh.faceID.clear();
  Manifold cylinder2(mesh);

  const auto prop2 = cylinder2.GetProperties();
  EXPECT_NEAR(prop.volume, prop2.volume, 0.001);
  EXPECT_NEAR(prop.surfaceArea, prop2.surfaceArea, 0.001);
}

/**
 * Curvature is the inverse of the radius of curvature, and signed such that
 * positive is convex and negative is concave. There are two orthogonal
 * principal curvatures at any point on a manifold, with one maximum and the
 * other minimum. Gaussian curvature is their product, while mean
 * curvature is their sum. Here we check our discrete approximations
 * calculated at each vertex against the constant expected values of spheres
 * of different radii and at different mesh resolutions.
 */
TEST(Manifold, CalculateCurvature) {
  const float precision = 0.015;
  for (int n = 4; n < 100; n *= 2) {
    const int gaussianIdx = 3;
    const int meanIdx = 4;
    Manifold sphere = Manifold::Sphere(1, 64).CalculateCurvature(
        gaussianIdx - 3, meanIdx - 3);
    MeshGL sphereGL = sphere.GetMeshGL();
    ASSERT_EQ(sphereGL.numProp, 5);
    EXPECT_NEAR(GetMinProperty(sphereGL, meanIdx), 2, 2 * precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, meanIdx), 2, 2 * precision);
    EXPECT_NEAR(GetMinProperty(sphereGL, gaussianIdx), 1, precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, gaussianIdx), 1, precision);

    sphere = sphere.Scale(glm::vec3(2.0f))
                 .CalculateCurvature(gaussianIdx - 3, meanIdx - 3);
    sphereGL = sphere.GetMeshGL();
    ASSERT_EQ(sphereGL.numProp, 5);
    EXPECT_NEAR(GetMinProperty(sphereGL, meanIdx), 1, precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, meanIdx), 1, precision);
    EXPECT_NEAR(GetMinProperty(sphereGL, gaussianIdx), 0.25, 0.25 * precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, gaussianIdx), 0.25, 0.25 * precision);
  }
}

/**
 * Testing more advanced Manifold operations.
 */

TEST(Manifold, Transform) {
  Manifold cube = Manifold::Cube({1, 2, 3});
  Manifold cube2 = cube;
  cube = cube.Rotate(30, 40, 50).Scale({6, 5, 4}).Translate({1, 2, 3});

  glm::mat3 rX(1.0f, 0.0f, 0.0f,          //
               0.0f, cosd(30), sind(30),  //
               0.0f, -sind(30), cosd(30));
  glm::mat3 rY(cosd(40), 0.0f, -sind(40),  //
               0.0f, 1.0f, 0.0f,           //
               sind(40), 0.0f, cosd(40));
  glm::mat3 rZ(cosd(50), sind(50), 0.0f,   //
               -sind(50), cosd(50), 0.0f,  //
               0.0f, 0.0f, 1.0f);
  glm::mat3 s = glm::mat3(1.0f);
  s[0][0] = 6;
  s[1][1] = 5;
  s[2][2] = 4;
  glm::mat4x3 transform = glm::mat4x3(s * rZ * rY * rX);
  transform[3] = glm::vec3(1, 2, 3);
  cube2 = cube2.Transform(transform);

  Identical(cube.GetMesh(), cube2.GetMesh());
}

TEST(Manifold, Slice) {
  Manifold cube = Manifold::Cube();
  CrossSection bottom = cube.Slice();
  CrossSection top = cube.Slice(1);
  EXPECT_EQ(bottom.Area(), 1);
  EXPECT_EQ(top.Area(), 0);
}

TEST(Manifold, MeshRelation) {
  Mesh gyroidMesh = Gyroid();
  MeshGL gyroidMeshGL = WithIndexColors(gyroidMesh);
  Manifold gyroid(gyroidMeshGL);

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorChannels = glm::ivec4(3, 4, 5, -1);
  if (options.exportModels) ExportMesh("gyroid.glb", gyroid.GetMeshGL(), opt);
#endif

  RelatedGL(gyroid, {gyroidMeshGL});
}

TEST(Manifold, MeshRelationTransform) {
  const Manifold cube = Manifold::Cube();
  const MeshGL cubeGL = cube.GetMeshGL();
  const Manifold turned = cube.Rotate(45, 90);

  RelatedGL(turned, {cubeGL});
}

TEST(Manifold, MeshRelationRefine) {
  const Mesh in = Csaszar();
  MeshGL inGL = WithIndexColors(in);
  Manifold csaszar(inGL);

  RelatedGL(csaszar, {inGL});
  csaszar = csaszar.RefineToLength(1);
  ExpectMeshes(csaszar, {{9019, 18038, 3}});
  RelatedGL(csaszar, {inGL});

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorChannels = glm::ivec4(3, 4, 5, -1);
  if (options.exportModels) ExportMesh("csaszar.glb", csaszar.GetMeshGL(), opt);
#endif
}

TEST(Manifold, MeshGLRoundTrip) {
  const Manifold cylinder = Manifold::Cylinder(2, 1);
  ASSERT_GE(cylinder.OriginalID(), 0);
  MeshGL inGL = cylinder.GetMeshGL();
  const Manifold cylinder2(inGL);
  const MeshGL outGL = cylinder2.GetMeshGL();

  ASSERT_EQ(inGL.runOriginalID.size(), 1);
  ASSERT_EQ(outGL.runOriginalID.size(), 1);
  ASSERT_EQ(outGL.runOriginalID[0], inGL.runOriginalID[0]);

  RelatedGL(cylinder2, {inGL});
}

void CheckCube(const MeshGL& cubeSTL) {
  Manifold cube(cubeSTL);
  EXPECT_EQ(cube.NumTri(), 12);
  EXPECT_EQ(cube.NumVert(), 8);
  EXPECT_EQ(cube.NumPropVert(), 24);

  auto prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0f);
}

TEST(Manifold, Merge) {
  MeshGL cubeSTL = CubeSTL();
  EXPECT_EQ(cubeSTL.NumTri(), 12);
  EXPECT_EQ(cubeSTL.NumVert(), 36);

  Manifold cubeBad(cubeSTL);
  EXPECT_TRUE(cubeBad.IsEmpty());
  EXPECT_EQ(cubeBad.Status(), Manifold::Error::NotManifold);

  EXPECT_TRUE(cubeSTL.Merge());
  CheckCube(cubeSTL);

  EXPECT_FALSE(cubeSTL.Merge());
  EXPECT_EQ(cubeSTL.mergeFromVert.size(), 28);
  cubeSTL.mergeFromVert.resize(14);
  cubeSTL.mergeToVert.resize(14);

  EXPECT_TRUE(cubeSTL.Merge());
  EXPECT_EQ(cubeSTL.mergeFromVert.size(), 28);
  CheckCube(cubeSTL);
}

TEST(Manifold, FaceIDRoundTrip) {
  const Manifold cube = Manifold::Cube();
  ASSERT_GE(cube.OriginalID(), 0);
  MeshGL inGL = cube.GetMeshGL();
  ASSERT_EQ(NumUnique(inGL.faceID), 6);
  inGL.faceID = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  const Manifold cube2(inGL);
  const MeshGL outGL = cube2.GetMeshGL();
  ASSERT_EQ(NumUnique(outGL.faceID), 12);
}

TEST(Manifold, MirrorUnion) {
  auto a = Manifold::Cube({5., 5., 5.}, true);
  auto b = a.Translate({2.5, 2.5, 2.5});
  auto result = a + b + b.Mirror({1, 1, 0});

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("manifold_mirror_union.glb", result.GetMesh(), {});
#endif

  auto vol_a = a.GetProperties().volume;
  EXPECT_FLOAT_EQ(vol_a * 2.75, result.GetProperties().volume);
  EXPECT_TRUE(a.Mirror(glm::vec3(0)).IsEmpty());
}

TEST(Manifold, MirrorUnion2) {
  auto a = Manifold::Cube();
  auto result = Manifold::Compose({a.Mirror({1, 0, 0})});
  EXPECT_TRUE(result.MatchesTriNormals());
}

TEST(Manifold, Invalid) {
  auto invalid = Manifold::Error::InvalidConstruction;
  auto circ = CrossSection::Circle(10.);
  auto empty_circ = CrossSection::Circle(-2.);
  auto empty_sq = CrossSection::Square(glm::vec2(0.0f));

  EXPECT_EQ(Manifold::Sphere(0).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(0, 5).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(2, -5).Status(), invalid);
  EXPECT_EQ(Manifold::Cube(glm::vec3(0.0f)).Status(), invalid);
  EXPECT_EQ(Manifold::Cube({-1, 1, 1}).Status(), invalid);
  EXPECT_EQ(Manifold::Extrude(circ, 0.).Status(), invalid);
  EXPECT_EQ(Manifold::Extrude(empty_circ, 10.).Status(), invalid);
  EXPECT_EQ(Manifold::Revolve(empty_sq).Status(), invalid);
}

TEST(Manifold, MultiCompose) {
  auto part = Manifold::Compose({Manifold::Cube({10, 10, 10})});
  auto finalAssembly =
      Manifold::Compose({part, part.Translate({0, 10, 0}),
                         part.Mirror({1, 0, 0}).Translate({10, 0, 0}),
                         part.Mirror({1, 0, 0}).Translate({10, 10, 0})});
  EXPECT_FLOAT_EQ(finalAssembly.GetProperties().volume, 4000);
}

TEST(Manifold, MergeDegenerates) {
  MeshGL cube = Manifold::Cube(glm::vec3(1), true).GetMeshGL();
  MeshGL squash;
  squash.vertProperties = cube.vertProperties;
  squash.triVerts = cube.triVerts;
  // Move one vert to the position of its neighbor and remove one triangle
  // linking them to break the manifold.
  squash.vertProperties[squash.vertProperties.size() - 1] *= -1;
  squash.triVerts.resize(squash.triVerts.size() - 3);
  // Rotate the degenerate triangle to the middle to catch more problems.
  std::rotate(squash.triVerts.begin(), squash.triVerts.begin() + 3 * 5,
              squash.triVerts.end());
  // Merge should remove the now duplicate vertex.
  EXPECT_TRUE(squash.Merge());
  // Manifold should remove the triangle with two references to the same vert.
  Manifold squashed = Manifold(squash);
  EXPECT_FALSE(squashed.IsEmpty());
  EXPECT_EQ(squashed.Status(), Manifold::Error::NoError);
}

TEST(Manifold, PinchedVert) {
  Mesh shape;
  shape.vertPos = {{0, 0, 0},         //
                   {1, 1, 0},         //
                   {1, -1, 0},        //
                   {-0.00001, 0, 0},  //
                   {-1, -1, -0},      //
                   {-1, 1, 0},        //
                   {0, 0, 2},         //
                   {0, 0, -2}};
  shape.triVerts = {{0, 2, 6},  //
                    {2, 1, 6},  //
                    {1, 0, 6},  //
                    {4, 3, 6},  //
                    {3, 5, 6},  //
                    {5, 4, 6},  //
                    {2, 0, 4},  //
                    {0, 3, 4},  //
                    {3, 0, 1},  //
                    {3, 1, 5},  //
                    {7, 2, 4},  //
                    {7, 4, 5},  //
                    {7, 5, 1},  //
                    {7, 1, 2}};
  Manifold touch(shape);
  EXPECT_FALSE(touch.IsEmpty());
  EXPECT_EQ(touch.Status(), Manifold::Error::NoError);
  EXPECT_EQ(touch.Genus(), 0);
}

TEST(Manifold, TictacHull) {
  const float tictacRad = 100;
  const float tictacHeight = 500;
  const int tictacSeg = 1000;
  const float tictacMid = tictacHeight - 2 * tictacRad;
  const auto sphere = Manifold::Sphere(tictacRad, tictacSeg);
  const std::vector<Manifold> spheres{sphere,
                                      sphere.Translate({0, 0, tictacMid})};
  const auto tictac = Manifold::Hull(spheres);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportMesh("tictac_hull.glb", tictac.GetMesh(), {});
  }
#endif

  EXPECT_EQ(sphere.NumVert() + tictacSeg, tictac.NumVert());
}

TEST(Manifold, HollowHull) {
  auto sphere = Manifold::Sphere(100, 360);
  auto hollow = sphere - sphere.Scale({0.8, 0.8, 0.8});
  const float sphere_vol = sphere.GetProperties().volume;
  EXPECT_FLOAT_EQ(hollow.Hull().GetProperties().volume, sphere_vol);
}

TEST(Manifold, CubeHull) {
  std::vector<glm::vec3> cubePts = {
      {0, 0, 0},       {1, 0, 0},   {0, 1, 0},      {0, 0, 1},  // corners
      {1, 1, 0},       {0, 1, 1},   {1, 0, 1},      {1, 1, 1},  // corners
      {0.5, 0.5, 0.5}, {0.5, 0, 0}, {0.5, 0.7, 0.2}  // internal points
  };
  auto cube = Manifold::Hull(cubePts);
  EXPECT_FLOAT_EQ(cube.GetProperties().volume, 1);
}

TEST(Manifold, EmptyHull) {
  const std::vector<glm::vec3> tooFew{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  EXPECT_TRUE(Manifold::Hull(tooFew).IsEmpty());

  const std::vector<glm::vec3> coplanar{
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
  EXPECT_TRUE(Manifold::Hull(coplanar).IsEmpty());
}

TEST(Manifold, MinGapCubeCube) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({2, 2, 0});

  float distance = a.MinGap(b, 1.5f);

  EXPECT_FLOAT_EQ(distance, sqrt(2));
}

TEST(Manifold, MinGapCubeCube2) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({3, 3, 0});

  float distance = a.MinGap(b, 3);

  EXPECT_FLOAT_EQ(distance, sqrt(2) * 2);
}

TEST(Manifold, MinGapCubeSphereOverlapping) {
  auto a = Manifold::Cube();
  auto b = Manifold::Sphere(1);

  float distance = a.MinGap(b, 0.1f);

  EXPECT_FLOAT_EQ(distance, 0);
}

TEST(Manifold, MinGapSphereSphere) {
  auto a = Manifold::Sphere(1);
  auto b = Manifold::Sphere(1).Translate({2, 2, 0});

  float distance = a.MinGap(b, 0.85f);

  EXPECT_FLOAT_EQ(distance, 2 * sqrt(2) - 2);
}

TEST(Manifold, MinGapSphereSphereOutOfBounds) {
  auto a = Manifold::Sphere(1);
  auto b = Manifold::Sphere(1).Translate({2, 2, 0});

  float distance = a.MinGap(b, 0.8f);

  EXPECT_FLOAT_EQ(distance, 0.8f);
}

TEST(Manifold, MinGapClosestPointOnEdge) {
  auto a = Manifold::Cube({1, 1, 1}, true).Rotate(0, 0, 45);
  auto b =
      Manifold::Cube({1, 1, 1}, true).Rotate(0, 45, 0).Translate({2, 0, 0});

  float distance = a.MinGap(b, 0.7f);

  EXPECT_FLOAT_EQ(distance, 2 - sqrt(2));
}

TEST(Manifold, MinGapClosestPointOnTriangleFace) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Scale({10, 10, 10}).Translate({2, -5, -1});

  float distance = a.MinGap(b, 1.1f);

  EXPECT_FLOAT_EQ(distance, 1);
}

TEST(Manifold, MingapAfterTransformations) {
  auto a = Manifold::Sphere(1, 512).Rotate(30, 30, 30);
  auto b =
      Manifold::Sphere(1, 512).Scale({3, 1, 1}).Rotate(0, 90, 45).Translate(
          {3, 0, 0});

  float distance = a.MinGap(b, 1.1f);

  ASSERT_NEAR(distance, 1, 0.001f);
}

TEST(Manifold, MingapStretchyBracelet) {
  auto a = StretchyBracelet();
  auto b = StretchyBracelet().Translate({0, 0, 20});

  float distance = a.MinGap(b, 10);

  ASSERT_NEAR(distance, 5, 0.001f);
}

TEST(Manifold, MinGapAfterTransformationsOutOfBounds) {
  auto a = Manifold::Sphere(1, 512).Rotate(30, 30, 30);
  auto b =
      Manifold::Sphere(1, 512).Scale({3, 1, 1}).Rotate(0, 90, 45).Translate(
          {3, 0, 0});

  float distance = a.MinGap(b, 0.95f);

  ASSERT_NEAR(distance, 0.95f, 0.001f);
}
TEST(Manifold, TriangleDistanceClosestPointsOnVertices) {
  std::array<glm::vec3, 3> p = {glm::vec3{-1, 0, 0}, glm::vec3{1, 0, 0},
                                glm::vec3{0, 1, 0}};

  std::array<glm::vec3, 3> q = {glm::vec3{2, 0, 0}, glm::vec3{4, 0, 0},
                                glm::vec3{3, 1, 0}};

  float distance = DistanceTriangleTriangleSquared(p, q);

  EXPECT_FLOAT_EQ(distance, 1);
}

TEST(Manifold, TriangleDistanceClosestPointOnEdge) {
  std::array<glm::vec3, 3> p = {glm::vec3{-1, 0, 0}, glm::vec3{1, 0, 0},
                                glm::vec3{0, 1, 0}};

  std::array<glm::vec3, 3> q = {glm::vec3{-1, 2, 0}, glm::vec3{1, 2, 0},
                                glm::vec3{0, 3, 0}};

  float distance = DistanceTriangleTriangleSquared(p, q);

  EXPECT_FLOAT_EQ(distance, 1);
}

TEST(Manifold, TriangleDistanceClosestPointOnEdge2) {
  std::array<glm::vec3, 3> p = {glm::vec3{-1, 0, 0}, glm::vec3{1, 0, 0},
                                glm::vec3{0, 1, 0}};

  std::array<glm::vec3, 3> q = {glm::vec3{1, 1, 0}, glm::vec3{3, 1, 0},
                                glm::vec3{2, 2, 0}};

  float distance = DistanceTriangleTriangleSquared(p, q);

  EXPECT_FLOAT_EQ(distance, 0.5f);
}

TEST(Manifold, TriangleDistanceClosestPointOnFace) {
  std::array<glm::vec3, 3> p = {glm::vec3{-1, 0, 0}, glm::vec3{1, 0, 0},
                                glm::vec3{0, 1, 0}};

  std::array<glm::vec3, 3> q = {glm::vec3{-1, 2, -0.5f}, glm::vec3{1, 2, -0.5f},
                                glm::vec3{0, 2, 1.5f}};

  float distance = DistanceTriangleTriangleSquared(p, q);

  EXPECT_FLOAT_EQ(distance, 1);
}

TEST(Manifold, TriangleDistanceOverlapping) {
  std::array<glm::vec3, 3> p = {glm::vec3{-1, 0, 0}, glm::vec3{1, 0, 0},
                                glm::vec3{0, 1, 0}};

  std::array<glm::vec3, 3> q = {glm::vec3{-1, 0, 0}, glm::vec3{1, 0.5f, 0},
                                glm::vec3{0, 1, 0}};

  float distance = DistanceTriangleTriangleSquared(p, q);

  EXPECT_FLOAT_EQ(distance, 0);
}
