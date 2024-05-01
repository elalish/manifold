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

#include <algorithm>

#include "cross_section.h"
#include "manifold.h"
#include "samples.h"
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
  EXPECT_NEAR(prop.volume, 18.7, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 34.5, 0.1);

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
  if (options.exportModels) ExportMesh("refinedCylinder.glb", out, options2);
#endif
}

TEST(Smooth, SmoothFlat) {
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

TEST(Smooth, Smooth2Length) {
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

TEST(Smooth, SmoothSphere) {
  int n[5] = {4, 8, 16, 32, 64};
  float precision[5] = {0.006, 0.003, 0.003, 0.0005, 0.00006};
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

TEST(Smooth, SmoothNormals) {
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

TEST(Smooth, SmoothMirrored) {
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

TEST(Smooth, Csaszar) {
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
