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

TEST(Manifold, Smooth) {
  Manifold tet = Manifold::Tetrahedron();
  Manifold smooth = Manifold::Smooth(tet.GetMesh());
  int n = 100;
  smooth = smooth.Refine(n);
  ExpectMeshes(smooth, {{2 * n * n + 2, 4 * n * n}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 17.38, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 33.38, 0.1);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("smoothTet.glb", smooth.GetMesh(), {});
#endif
}

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
  Manifold smooth = cone.RefineToLength(0.1).CalculateNormals(0);
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 1142.9, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 764.28, 0.01);
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

TEST(Manifold, Smooth2Length) {
  Manifold cone = Manifold::Extrude(
      CrossSection::Circle(10, 10).Translate({10, 0}), 2, 0, 0, {0, 0});
  cone += cone.Scale({1, 1, -5});
  Manifold smooth = Manifold::Smooth(cone.GetMesh());
  smooth = smooth.RefineToLength(0.1);
  ExpectMeshes(smooth, {{85250, 170496}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 4688, 1);
  EXPECT_NEAR(prop.surfaceArea, 1369, 1);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("smoothCones.glb", smooth.GetMesh(), {});
#endif
}

TEST(Manifold, SmoothSphere) {
  int n[5] = {4, 8, 16, 32, 64};
  float precision[5] = {0.04, 0.003, 0.003, 0.0005, 0.00006};
  for (int i = 0; i < 5; ++i) {
    Manifold sphere = Manifold::Sphere(1, n[i]);
    // Refine(odd) puts a center point in the triangle, which is the worst case.
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
  smooth.halfedgeTangent[6] = {0, 0, 0, 1};
  smooth.halfedgeTangent[22] = {0, 0, 0, 1};
  smooth.halfedgeTangent[16] = {0, 0, 0, 1};
  smooth.halfedgeTangent[18] = {0, 0, 0, 1};
  Manifold interp(smooth);
  interp = interp.Refine(100);

  ExpectMeshes(interp, {{40002, 80000}});
  auto prop = interp.GetProperties();
  EXPECT_NEAR(prop.volume, 3.53, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 11.39, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    const Mesh out = interp.GetMesh();
    ExportOptions options;
    options.faceted = false;
    options.mat.roughness = 0.1;

    options.mat.vertColor.resize(interp.NumVert());
    const glm::vec4 red(1, 0, 0, 1);
    const glm::vec4 purple(1, 0, 1, 1);
    for (int tri = 0; tri < interp.NumTri(); ++tri) {
      for (int i : {0, 1, 2}) {
        const glm::vec3& uvw = {0.5, 0.5, 0.0};
        const float alpha = glm::min(uvw[0], glm::min(uvw[1], uvw[2]));
        options.mat.vertColor[out.triVerts[tri][i]] =
            glm::mix(purple, red, glm::smoothstep(0.0f, 0.2f, alpha));
      }
    }
    ExportMesh("sharpenedSphere.glb", out, options);
  }
#endif
}

TEST(Manifold, SmoothMirrored) {
  const Mesh tet = Manifold::Tetrahedron().GetMesh();
  Manifold smooth = Manifold::Smooth(tet);
  Manifold mirror = smooth.Scale({-1, 1, 2}).Refine(10);
  smooth = smooth.Refine(10).Scale({1, 1, 2});

  auto prop0 = smooth.GetProperties();
  auto prop1 = mirror.GetProperties();
  EXPECT_NEAR(prop0.volume, prop1.volume, 0.1);
  EXPECT_NEAR(prop0.surfaceArea, prop1.surfaceArea, 0.1);
}

TEST(Manifold, Csaszar) {
  Manifold csaszar = Manifold::Smooth(Csaszar());
  csaszar = csaszar.Refine(100);
  ExpectMeshes(csaszar, {{70000, 140000}});
  auto prop = csaszar.GetProperties();
  EXPECT_NEAR(prop.volume, 84699, 10);
  EXPECT_NEAR(prop.surfaceArea, 14796, 10);

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
