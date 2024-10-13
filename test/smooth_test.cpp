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

#include "../src/utils.h"
#ifdef MANIFOLD_CROSS_SECTION
#include "manifold/cross_section.h"
#endif
#include "manifold/manifold.h"
#include "test.h"

using namespace manifold;

TEST(Smooth, Tetrahedron) {
  Manifold tet = Manifold::Tetrahedron();
  Manifold smooth = Manifold::Smooth(tet.GetMeshGL());
  int n = 100;
  smooth = smooth.Refine(n);
  ExpectMeshes(smooth, {{2 * n * n + 2, 4 * n * n}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 17.0, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 32.9, 0.1);

  MeshGL out = smooth.CalculateCurvature(-1, 0).GetMeshGL();
  float maxMeanCurvature = 0;
  for (size_t i = 3; i < out.vertProperties.size(); i += 4) {
    maxMeanCurvature =
        std::max(maxMeanCurvature, std::abs(out.vertProperties[i]));
  }
  EXPECT_NEAR(maxMeanCurvature, 4.73, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("smoothTet.glb", smooth.GetMeshGL(), {});
#endif
}

TEST(Smooth, RefineQuads) {
  Manifold cylinder =
      Manifold(WithPositionColors(Manifold::Cylinder(2, 1, -1, 12)))
          .SmoothOut()
          .RefineToLength(0.05);
  EXPECT_EQ(cylinder.NumTri(), 16892);
  auto prop = cylinder.GetProperties();
  EXPECT_NEAR(prop.volume, 2 * kPi, 0.003);
  EXPECT_NEAR(prop.surfaceArea, 6 * kPi, 0.004);
  const MeshGL out = cylinder.GetMeshGL();
  CheckGL(out);

  const MeshGL baseline = WithPositionColors(cylinder);
  EXPECT_EQ(out.NumVert(), baseline.NumVert());
  float maxDiff = 0;
  for (size_t i = 0; i < out.vertProperties.size(); ++i) {
    maxDiff = std::max(
        maxDiff, std::abs(out.vertProperties[i] - baseline.vertProperties[i]));
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
  Manifold cone = Manifold::Cylinder(5, 10, 5, 12);
  Manifold smooth = cone.SmoothOut().RefineToLength(0.5).CalculateNormals(0);
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 1158.61, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 768.12, 0.01);
  MeshGL out = smooth.GetMeshGL();
  CheckGL(out);

  Manifold smooth1 = cone.SmoothOut(180, 1).RefineToLength(0.5);
  auto prop1 = smooth1.GetProperties();

  Manifold smooth2 = cone.SmoothOut(180, 0).RefineToLength(0.5);
  auto prop2 = smooth2.GetProperties();
  EXPECT_NEAR(prop2.volume, prop1.volume, 0.01);
  EXPECT_NEAR(prop2.surfaceArea, prop1.surfaceArea, 0.01);

#ifdef MANIFOLD_EXPORT
  ExportOptions options2;
  options2.faceted = false;
  options2.mat.normalChannels = {3, 4, 5};
  options2.mat.roughness = 0;
  if (options.exportModels)
    ExportMesh("smoothTruncatedCone.glb", out, options2);
#endif
}

#ifdef MANIFOLD_CROSS_SECTION
TEST(Smooth, ToLength) {
  Manifold cone = Manifold::Extrude(
      CrossSection::Circle(10, 10).Translate({10, 0}).ToPolygons(), 2, 0, 0,
      {0, 0});
  cone += cone.Scale({1, 1, -5});
  Manifold smooth = cone.AsOriginal().SmoothOut(180).RefineToLength(0.1);
  ExpectMeshes(smooth, {{85250, 170496}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 4604, 1);
  EXPECT_NEAR(prop.surfaceArea, 1356, 1);

  MeshGL out = smooth.CalculateCurvature(-1, 0).GetMeshGL();
  float maxMeanCurvature = 0;
  for (size_t i = 3; i < out.vertProperties.size(); i += 4) {
    maxMeanCurvature =
        std::max(maxMeanCurvature, std::abs(out.vertProperties[i]));
  }
  EXPECT_NEAR(maxMeanCurvature, 1.67, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("smoothToLength.glb", smooth.GetMeshGL(), {});
#endif
}
#endif

TEST(Smooth, Sphere) {
  int n[5] = {4, 8, 16, 32, 64};
  // Tests vertex precision of interpolation
  double precision[5] = {0.04, 0.003, 0.003, 0.0005, 0.00006};
  for (int i = 0; i < 5; ++i) {
    Manifold sphere = Manifold::Sphere(1, n[i]);
    // Refine(3*x) makes a center point, which is the worst case.
    Manifold smoothed = Manifold::Smooth(sphere.GetMeshGL()).Refine(6);
    // Refine(3*x) puts a center point in the triangle, which is the worst
    // case.
    MeshGL64 out = smoothed.GetMeshGL64();
    const int numVert = out.NumVert();
    double maxR2 = 0;
    double minR2 = 2;
    for (int v = 0; v < numVert; ++v) {
      const vec3 a = out.GetVertPos(v);
      const double r2 = dot(a, a);
      maxR2 = std::max(maxR2, r2);
      minR2 = std::min(minR2, r2);
    }
    EXPECT_NEAR(std::sqrt(minR2), 1, precision[i]);
    EXPECT_NEAR(std::sqrt(maxR2), 1, precision[i]);
  }
}

TEST(Smooth, Precision) {
  // Tests face precision of refinement
  const double precision = 0.001;
  const double radius = 10;
  const double height = 10;
  Manifold cylinder = Manifold::Cylinder(height, radius, radius, 8);
  Manifold smoothed = cylinder.SmoothOut().RefineToPrecision(precision);
  // Makes an edge bisector, which is the worst case.
  MeshGL64 out = smoothed.Refine(2).GetMeshGL64();
  const int numVert = out.NumVert();
  double maxR2 = 0;
  double minR2 = 2 * radius * radius;
  for (int v = 0; v < numVert; ++v) {
    const vec3 a = out.GetVertPos(v);
    const vec2 a1(a);
    // Ignore end caps.
    const double r2 = (std::abs(a.z) < 0.001 || std::abs(a.z - height) < 0.001)
                          ? radius * radius
                          : la::dot(a1, a1);
    maxR2 = std::max(maxR2, r2);
    minR2 = std::min(minR2, r2);
  }
  EXPECT_NEAR(std::sqrt(minR2), radius - precision, 1e-4);
  EXPECT_NEAR(std::sqrt(maxR2), radius, 1e-8);
  EXPECT_EQ(smoothed.NumTri(), 7984);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("refineCylinder.glb", smoothed.GetMeshGL(), {});
#endif
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
    ExportMesh("smoothNormals.glb", byNormals.GetMeshGL(), {});
#endif
}

TEST(Smooth, Manual) {
  // Unit Octahedron
  const auto oct = Manifold::Sphere(1, 4).GetMeshGL();
  MeshGL smooth = Manifold::Smooth(oct).GetMeshGL();
  // Sharpen the edge from vert 4 to 5
  smooth.halfedgeTangent[4 * 6 + 3] = 0;
  smooth.halfedgeTangent[4 * 22 + 3] = 0;
  smooth.halfedgeTangent[4 * 16 + 3] = 0;
  smooth.halfedgeTangent[4 * 18 + 3] = 0;
  Manifold interp(smooth);
  interp = interp.Refine(100);

  ExpectMeshes(interp, {{40002, 80000}});
  auto prop = interp.GetProperties();
  EXPECT_NEAR(prop.volume, 3.74, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 11.78, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    interp = interp.CalculateCurvature(-1, 0).SetProperties(
        3, [](double* newProp, vec3 pos, const double* oldProp) {
          const vec3 red(1, 0, 0);
          const vec3 purple(1, 0, 1);
          vec3 color = la::lerp(purple, red, smoothstep(0.0, 2.0, oldProp[0]));
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
  const auto tet = Manifold::Tetrahedron().Scale({1, 2, 3}).GetMeshGL();
  Manifold smooth = Manifold::Smooth(tet);
  Manifold mirror = smooth.Scale({-2, 2, 2}).Refine(10);
  smooth = smooth.Refine(10).Scale({2, 2, 2});

  auto prop0 = smooth.GetProperties();
  auto prop1 = mirror.GetProperties();
  EXPECT_NEAR(prop0.volume, prop1.volume, 0.1);
  EXPECT_NEAR(prop0.surfaceArea, prop1.surfaceArea, 0.1);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("smoothMirrored.glb", mirror.GetMeshGL(), {});
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
    const MeshGL out = csaszar.GetMeshGL();
    ExportOptions options;
    options.faceted = false;
    options.mat.roughness = 0.1;

    options.mat.vertColor.resize(csaszar.NumVert());
    const vec4 blue(0, 0, 1, 1);
    const vec4 yellow(1, 1, 0, 1);
    for (size_t tri = 0; tri < csaszar.NumTri(); ++tri) {
      for (int i : {0, 1, 2}) {
        const vec3& uvw = {0.5, 0.5, 0.0};
        const double alpha = std::min(uvw[0], std::min(uvw[1], uvw[2]));
        options.mat.vertColor[out.triVerts[3 * tri + i]] =
            la::lerp(yellow, blue, smoothstep(0.0, 0.2, alpha));
      }
    }
    ExportMesh("smoothCsaszar.glb", out, options);
  }
#endif
}

vec4 CircularTangent(const vec3& tangent, const vec3& edgeVec) {
  const vec3 dir = la::normalize(tangent);

  double weight = std::abs(la::dot(dir, la::normalize(edgeVec)));
  if (weight == 0) {
    weight = 1;
  }
  // Quadratic weighted bezier for circular interpolation
  const vec4 bz2 = weight * vec4(dir * la::length(edgeVec) / (2 * weight), 1);
  // Equivalent cubic weighted bezier
  const vec4 bz3 = la::lerp(vec4(0, 0, 0, 1), bz2, 2 / 3.0);
  // Convert from homogeneous form to geometric form
  return vec4(vec3(bz3) / bz3.w, bz3.w);
}

#ifdef MANIFOLD_CROSS_SECTION
TEST(Smooth, Torus) {
  MeshGL64 torusMesh =
      Manifold::Revolve(
          CrossSection::Circle(1, 8).Translate({2, 0}).ToPolygons(), 6)
          .GetMeshGL64();
  const int numTri = torusMesh.NumTri();
  const int numProp = torusMesh.numProp;

  // Create correct toroidal halfedge tangents - SmoothOut() is too generic to
  // do this perfectly.
  torusMesh.halfedgeTangent.resize(4 * 3 * numTri);
  for (int tri = 0; tri < numTri; ++tri) {
    const auto triVerts = torusMesh.GetTriVerts(tri);
    for (const int i : {0, 1, 2}) {
      vec4 tangent;
      const vec3 v = torusMesh.GetVertPos(triVerts[i]);
      const vec3 v1 = torusMesh.GetVertPos(triVerts[(i + 1) % 3]);
      const vec3 edge = v1 - v;
      if (edge.z == 0) {
        vec3 tan(v.y, -v.x, 0);
        tan *= la::dot(tan, edge) < 0 ? -1.0 : 1.0;
        tangent = CircularTangent(tan, edge);
      } else if (std::abs(la::determinant(mat2(vec2(v), vec2(edge)))) <
                 kTolerance) {
        const double theta = std::asin(v.z);
        vec2 xy(v);
        const double r = la::length(xy);
        xy = xy / r * v.z * (r > 2 ? -1.0 : 1.0);
        vec3 tan(xy.x, xy.y, std::cos(theta));
        tan *= la::dot(tan, edge) < 0 ? -1.0 : 1.0;
        tangent = CircularTangent(tan, edge);
      } else {
        tangent = {0, 0, 0, -1};
      }
      const int e = 3 * tri + i;
      for (const int j : {0, 1, 2, 3})
        torusMesh.halfedgeTangent[4 * e + j] = tangent[j];
    }
  }

  Manifold smooth = Manifold(torusMesh)
                        .RefineToLength(0.1)
                        .CalculateCurvature(-1, 0)
                        .CalculateNormals(1);
  MeshGL out = smooth.GetMeshGL();
  float maxMeanCurvature = 0;
  for (size_t i = 0; i < out.vertProperties.size(); i += 7) {
    vec3 v(out.vertProperties[i], out.vertProperties[i + 1],
           out.vertProperties[i + 2]);
    vec3 p(v.x, v.y, 0);
    p = la::normalize(p) * 2.0;
    double r = la::length(v - p);
    ASSERT_NEAR(r, 1, 0.006);
    maxMeanCurvature =
        std::max(maxMeanCurvature, std::abs(out.vertProperties[i + 3]));
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
#endif

TEST(Smooth, SineSurface) {
  Manifold surface =
      Manifold::LevelSet(
          [](vec3 p) {
            double mid = la::sin(p.x) + la::sin(p.y);
            return (p.z > mid - 0.5 && p.z < mid + 0.5) ? 1.0 : -1.0;
          },
          {vec3(-2 * kPi + 0.2), vec3(0 * kPi - 0.2)}, 1)
          .AsOriginal();

  Manifold smoothed =
      surface.CalculateNormals(0, 50).SmoothByNormals(0).Refine(8);
  auto prop = smoothed.GetProperties();
  EXPECT_NEAR(prop.volume, 8.09, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 30.93, 0.01);
  EXPECT_EQ(smoothed.Genus(), 0);
  EXPECT_NEAR(smoothed.TrimByPlane({0, 1, 1}, -3.19487).GetProperties().volume,
              prop.volume, 1e-5);

  Manifold smoothed1 = surface.SmoothOut(50).Refine(8);
  auto prop1 = smoothed1.GetProperties();
  EXPECT_FLOAT_EQ(prop1.volume, prop.volume);
  EXPECT_FLOAT_EQ(prop1.surfaceArea, prop.surfaceArea);
  EXPECT_EQ(smoothed1.Genus(), 0);
  EXPECT_NEAR(smoothed1.TrimByPlane({0, 1, 1}, -3.19487).GetProperties().volume,
              prop1.volume, 1e-5);

  Manifold smoothed2 = surface.SmoothOut(180, 1).Refine(8);
  auto prop2 = smoothed2.GetProperties();
  EXPECT_NEAR(prop2.volume, 9.00, 0.01);
  EXPECT_NEAR(prop2.surfaceArea, 33.52, 0.01);
  EXPECT_EQ(smoothed2.Genus(), 0);
  EXPECT_NEAR(smoothed2.TrimByPlane({0, 1, 1}, -3.19487).GetProperties().volume,
              prop2.volume, 1e-3);

  Manifold smoothed3 = surface.SmoothOut(50, 0.5).Refine(8);
  auto prop3 = smoothed3.GetProperties();
  EXPECT_NEAR(prop3.volume, 8.44, 0.01);
  EXPECT_NEAR(prop3.surfaceArea, 31.73, 0.02);
  EXPECT_EQ(smoothed3.Genus(), 0);
  EXPECT_NEAR(smoothed3.TrimByPlane({0, 1, 1}, -3.19487).GetProperties().volume,
              prop3.volume, 1e-5);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportOptions options2;
    // options2.faceted = false;
    // options2.mat.normalChannels = {3, 4, 5};
    ExportMesh("smoothSineSurface.glb", smoothed.GetMeshGL(), options2);
  }
#endif
}

TEST(Smooth, SDF) {
  const double r = 10;
  const double extra = 2;

  auto sphericalGyroid = [r](vec3 p) {
    const double gyroid =
        cos(p.x) * sin(p.y) + cos(p.y) * sin(p.z) + cos(p.z) * sin(p.x);
    const double d = la::min(0.0, r - la::length(p));
    return gyroid - d * d / 2;
  };

  auto gradient = [r](vec3 pos) {
    const double rad = la::length(pos);
    const double d = la::min(0.0, r - rad) / (rad > 0 ? rad : 1);
    const vec3 sphereGrad = d * pos;
    const vec3 gyroidGrad(cos(pos.z) * cos(pos.x) - sin(pos.x) * sin(pos.y),
                          cos(pos.x) * cos(pos.y) - sin(pos.y) * sin(pos.z),
                          cos(pos.y) * cos(pos.z) - sin(pos.z) * sin(pos.x));
    return gyroidGrad + sphereGrad;
  };

  auto error = [sphericalGyroid](double* newProp, vec3 pos,
                                 const double* oldProp) {
    newProp[0] = std::abs(sphericalGyroid(pos));
  };

  Manifold gyroid = Manifold::LevelSet(
      sphericalGyroid, {vec3(-r - extra), vec3(r + extra)}, 0.5, 0, 0.00001);

  EXPECT_LT(gyroid.NumTri(), 76000);

  Manifold interpolated = gyroid.Refine(3).SetProperties(1, error);

  Manifold smoothed =
      gyroid
          .SetProperties(
              3,
              [gradient](double* newProp, vec3 pos, const double* oldProp) {
                const vec3 normal = -la::normalize(gradient(pos));
                for (const int i : {0, 1, 2}) newProp[i] = normal[i];
              })
          .SmoothByNormals(0)
          .RefineToLength(0.1)
          .SetProperties(1, error);

  MeshGL out = smoothed.GetMeshGL();
  EXPECT_NEAR(GetMaxProperty(out, 3), 0, 0.026);
  EXPECT_NEAR(GetMaxProperty(interpolated.GetMeshGL(), 3), 0, 0.083);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportOptions options2;
    ExportMesh("smoothGyroid.glb", gyroid.GetMeshGL(), options2);
  }
#endif
}
