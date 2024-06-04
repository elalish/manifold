// Copyright 2024 The Manifold Authors.
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

#include <algorithm>

#include "cross_section.h"
#include "manifold.h"
#include "samples.h"
#include "sdf.h"
#include "test.h"
#include "tri_dist.h"

using namespace manifold;

TEST(Smooth, Tetrahedron) {
  Manifold tet = Manifold::Tetrahedron();
  Manifold smooth = Manifold::Smooth(tet.GetMesh());
  int n = 100;
  smooth = smooth.Refine(n);
  ExpectMeshes(smooth, {{2 * n * n + 2, 4 * n * n}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 17.0, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 32.9, 0.1);

  MeshGL out = smooth.CalculateCurvature(-1, 0).GetMeshGL();
  float maxMeanCurvature = 0;
  for (int i = 3; i < out.vertProperties.size(); i += 4) {
    maxMeanCurvature =
        glm::max(maxMeanCurvature, glm::abs(out.vertProperties[i]));
  }
  EXPECT_NEAR(maxMeanCurvature, 4.73, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("smoothTet.glb", smooth.GetMesh(), {});
#endif
}

TEST(Smooth, RefineQuads) {
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
  if (options.exportModels) ExportMesh("refineQuads.glb", out, options2);
#endif
}

TEST(Smooth, TruncatedCone) {
  Manifold cone = Manifold::Cylinder(5, 10, 5, 12).SmoothOut();
  Manifold smooth = cone.RefineToLength(0.5).CalculateNormals(0);
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 1062.27, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 751.46, 0.01);
  MeshGL out = smooth.GetMeshGL();
  CheckGL(out);

#ifdef MANIFOLD_EXPORT
  ExportOptions options2;
  options2.faceted = false;
  options2.mat.normalChannels = {3, 4, 5};
  options2.mat.roughness = 0;
  if (options.exportModels)
    ExportMesh("smoothTruncatedCone.glb", out, options2);
#endif
}

TEST(Smooth, ToLength) {
  Manifold cone = Manifold::Extrude(
      CrossSection::Circle(10, 10).Translate({10, 0}), 2, 0, 0, {0, 0});
  cone += cone.Scale({1, 1, -5});
  Manifold smooth = Manifold::Smooth(cone.GetMesh());
  smooth = smooth.RefineToLength(0.1);
  ExpectMeshes(smooth, {{85250, 170496}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 4604, 1);
  EXPECT_NEAR(prop.surfaceArea, 1356, 1);

  MeshGL out = smooth.CalculateCurvature(-1, 0).GetMeshGL();
  float maxMeanCurvature = 0;
  for (int i = 3; i < out.vertProperties.size(); i += 4) {
    maxMeanCurvature =
        glm::max(maxMeanCurvature, glm::abs(out.vertProperties[i]));
  }
  EXPECT_NEAR(maxMeanCurvature, 1.67, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("smoothToLength.glb", smooth.GetMesh(), {});
#endif
}

TEST(Smooth, Sphere) {
  int n[5] = {4, 8, 16, 32, 64};
  float precision[5] = {0.04, 0.003, 0.003, 0.0005, 0.00006};
  for (int i = 0; i < 5; ++i) {
    Manifold sphere = Manifold::Sphere(1, n[i]);
    // Refine(odd) puts a center point in the triangle, which is the worst case.
    Manifold smoothed = Manifold::Smooth(sphere.GetMesh()).Refine(6);
    // Refine(3*x) puts a center point in the triangle, which is the worst
    // case.
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

TEST(Smooth, Normals) {
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
    ExportMesh("smoothNormals.glb", byNormals.GetMesh(), {});
#endif
}

TEST(Smooth, Manual) {
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
  EXPECT_NEAR(prop.volume, 3.74, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 11.78, 0.01);

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
    ExportMesh("manual.glb", out, options);
  }
#endif
}

TEST(Smooth, Mirrored) {
  const Mesh tet = Manifold::Tetrahedron().Scale({1, 2, 3}).GetMesh();
  Manifold smooth = Manifold::Smooth(tet);
  Manifold mirror = smooth.Scale({-2, 2, 2}).Refine(10);
  smooth = smooth.Refine(10).Scale({2, 2, 2});

  auto prop0 = smooth.GetProperties();
  auto prop1 = mirror.GetProperties();
  EXPECT_NEAR(prop0.volume, prop1.volume, 0.1);
  EXPECT_NEAR(prop0.surfaceArea, prop1.surfaceArea, 0.1);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("smoothMirrored.glb", mirror.GetMesh(), {});
#endif
}

TEST(Smooth, Csaszar) {
  Manifold csaszar = Manifold::Smooth(Csaszar());
  csaszar = csaszar.Refine(100);
  ExpectMeshes(csaszar, {{70000, 140000}});
  auto prop = csaszar.GetProperties();
  EXPECT_NEAR(prop.volume, 79890, 10);
  EXPECT_NEAR(prop.surfaceArea, 11950, 10);

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

TEST(Smooth, Torus) {
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

  Manifold smooth = Manifold(torusMesh)
                        .RefineToLength(0.1)
                        .CalculateCurvature(-1, 0)
                        .CalculateNormals(1);
  MeshGL out = smooth.GetMeshGL();
  float maxMeanCurvature = 0;
  for (int i = 0; i < out.vertProperties.size(); i += 7) {
    glm::vec3 v(out.vertProperties[i], out.vertProperties[i + 1],
                out.vertProperties[i + 2]);
    glm::vec3 p(v.x, v.y, 0);
    p = glm::normalize(p) * 2.0f;
    float r = glm::length(v - p);
    ASSERT_NEAR(r, 1, 0.006);
    maxMeanCurvature =
        glm::max(maxMeanCurvature, glm::abs(out.vertProperties[i + 3]));
  }
  EXPECT_NEAR(maxMeanCurvature, 1.63, 0.01);

#ifdef MANIFOLD_EXPORT
  ExportOptions options2;
  options2.faceted = false;
  options2.mat.normalChannels = {4, 5, 6};
  options2.mat.roughness = 0;
  if (options.exportModels) ExportMesh("smoothTorus.glb", out, options2);
#endif
}

TEST(Smooth, SineSurface) {
  MeshGL surface = LevelSet(
      [](glm::vec3 p) {
        float mid = glm::sin(p.x) + glm::sin(p.y);
        return (p.z > mid - 0.5 && p.z < mid + 0.5) ? 1 : -1;
      },
      {glm::vec3(-2 * glm::pi<float>() + 0.2),
       glm::vec3(0 * glm::pi<float>() - 0.2)},
      1);
  Manifold smoothed =
      Manifold(surface).CalculateNormals(0, 50).SmoothByNormals(0).Refine(8);
  auto prop = smoothed.GetProperties();
  EXPECT_NEAR(prop.volume, 7.89, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 30.60, 0.01);
  EXPECT_EQ(smoothed.Genus(), 0);

  Manifold smoothed1 = Manifold(surface).SmoothOut(50).Refine(8);
  auto prop1 = smoothed1.GetProperties();
  EXPECT_FLOAT_EQ(prop1.volume, prop.volume);
  EXPECT_FLOAT_EQ(prop1.surfaceArea, prop.surfaceArea);
  EXPECT_EQ(smoothed1.Genus(), 0);

  Manifold smoothed2 = Manifold(surface).SmoothOut(180, 1).Refine(8);
  auto prop2 = smoothed2.GetProperties();
  EXPECT_NEAR(prop2.volume, 9.02, 0.01);
  EXPECT_NEAR(prop2.surfaceArea, 33.56, 0.01);
  EXPECT_EQ(smoothed2.Genus(), 0);

  Manifold smoothed3 = Manifold(surface).SmoothOut(50, 0.5).Refine(8);
  auto prop3 = smoothed3.GetProperties();
  EXPECT_NEAR(prop3.volume, 8.46, 0.01);
  EXPECT_NEAR(prop3.surfaceArea, 31.66, 0.01);
  EXPECT_EQ(smoothed3.Genus(), 0);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportOptions options2;
    // options2.faceted = false;
    // options2.mat.normalChannels = {3, 4, 5};
    ExportMesh("smoothSineSurface.glb", smoothed.GetMeshGL(), options2);
  }
#endif
}