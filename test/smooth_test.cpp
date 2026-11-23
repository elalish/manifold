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
#include <cmath>
#include <limits>

#include "gtest/gtest.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"
#include "test.h"

using namespace manifold;

TEST(Smooth, Tetrahedron) {
  Manifold tet = Manifold::Tetrahedron();
  Manifold smooth = tet.CalculateNormals(180).SmoothByNormals();
  int n = 100;
  smooth = smooth.Refine(n);
  ExpectMeshes(smooth, {{2 * n * n + 2, 4 * n * n, 3}});
  EXPECT_NEAR(smooth.Volume(), 17.0, 0.1);
  EXPECT_NEAR(smooth.SurfaceArea(), 32.9, 0.1);

  MeshGL out = smooth.CalculateCurvature(-1, 3).GetMeshGL();
  float maxMeanCurvature = 0;
  for (size_t i = 6; i < out.vertProperties.size(); i += 7) {
    maxMeanCurvature =
        std::max(maxMeanCurvature, std::abs(out.vertProperties[i]));
  }
  EXPECT_NEAR(maxMeanCurvature, 4.73, 0.01);

  if (options.exportModels) WriteTestOBJ("smoothTet.obj", smooth);
}

TEST(Smooth, RefineQuads) {
  Manifold source = Manifold::Cylinder(2, 1, -1, {12}).CalculateNormals();
  const Box bbox = source.BoundingBox();
  const vec3 size = bbox.Size();
  Manifold cylinder =
      source
          .SetProperties(
              6,
              [bbox, size](double* newProp, vec3 pos, const double* oldProp) {
                for (const int i : {0, 1, 2}) {
                  newProp[i] = oldProp[i];
                  newProp[i + 3] = (pos[i] - bbox.min[i]) / size[i];
                }
              })
          .SmoothByNormals()
          .RefineToLength(0.05)
          .SetProperties(3,
                         [](double* newProp, vec3 pos, const double* oldProp) {
                           for (const int i : {0, 1, 2})
                             newProp[i] = oldProp[i + 3];
                         });
  EXPECT_EQ(cylinder.NumTri(), 16644);
  EXPECT_NEAR(cylinder.Volume(), 2 * kPi, 0.003);
  EXPECT_NEAR(cylinder.SurfaceArea(), 6 * kPi, 0.004);
  const MeshGL out = cylinder.GetMeshGL();
  CheckGL(out);

  const MeshGL baseline = WithPositionColors(cylinder).GetMeshGL();
  EXPECT_EQ(out.NumVert(), baseline.NumVert());
  float maxDiff = 0;
  for (size_t i = 0; i < out.vertProperties.size(); ++i) {
    maxDiff = std::max(
        maxDiff, std::abs(out.vertProperties[i] - baseline.vertProperties[i]));
  }
  // This has a wide tolerance because the triangle colors on the ends are still
  // being stretched out into circular arcs, which introduces unavoidable error.
  EXPECT_LE(maxDiff, 0.07);

  if (options.exportModels) WriteTestOBJ("refineQuads.obj", cylinder);
}

TEST(Smooth, TruncatedCone) {
  Manifold cone = Manifold::Cylinder(5, 10, 5, {12});
  Manifold smooth = cone.CalculateNormals()
                        .SmoothByNormals()
                        .RefineToLength(0.5)
                        .CalculateNormals();
  EXPECT_NEAR(smooth.Volume(), 1021.05, 0.01);
  EXPECT_NEAR(smooth.SurfaceArea(), 737.67, 0.01);
  CheckGL(smooth, false);

  if (options.exportModels) WriteTestOBJ("smoothTruncatedCone.obj", smooth);
}

TEST(Smooth, ToLength) {
  Manifold cone = Manifold::Extrude(
      CrossSection::Circle(10, {10}).Translate({10, 0}).ToPolygons(), 2, 0, 0,
      {0, 0});
  cone += cone.Scale({1, 1, -5});
  EXPECT_EQ(cone.NumVert(), 11);
  Manifold smooth =
      cone.CalculateNormals(180).SmoothByNormals().RefineToLength(0.1);
  ExpectMeshes(smooth, {{85250, 170496, 3}});
  EXPECT_NEAR(smooth.Volume(), 4570, 1);
  EXPECT_NEAR(smooth.SurfaceArea(), 1348, 1);

  MeshGL out = smooth.CalculateCurvature(-1, 3).GetMeshGL();
  float maxMeanCurvature = 0;
  for (size_t i = 6; i < out.vertProperties.size(); i += 7) {
    maxMeanCurvature =
        std::max(maxMeanCurvature, std::abs(out.vertProperties[i]));
  }
  EXPECT_NEAR(maxMeanCurvature, 1.63, 0.01);

  if (options.exportModels) WriteTestOBJ("smoothToLength.obj", smooth);
}

TEST(Smooth, Sphere) {
  int n[5] = {4, 8, 16, 32, 64};
  // Tests vertex precision of interpolation
  double precision[5] = {0.04, 0.015, 0.003, 0.0005, 0.00006};
  for (int i = 0; i < 5; ++i) {
    Manifold sphere = Manifold::Sphere(1, {n[i]});
    // Refine(3*x) makes a center point, which is the worst case.
    Manifold smoothed =
        sphere.CalculateNormals(180).SmoothByNormals().Refine(6);
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
  // Tests face tolerance of refinement
  const double tolerance = 0.001;
  const double radius = 10;
  const double height = 10;
  Manifold cylinder = Manifold::Cylinder(height, radius, radius, {8});
  Manifold smoothed =
      cylinder.CalculateNormals().SmoothByNormals().RefineToTolerance(
          tolerance);
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
  EXPECT_NEAR(std::sqrt(minR2), radius - tolerance, 1e-4);
  EXPECT_NEAR(std::sqrt(maxR2), radius, 1e-8);
  EXPECT_EQ(smoothed.NumTri(), 7984);
  if (options.exportModels) WriteTestOBJ("refineCylinder.obj", smoothed);
}

TEST(Smooth, FacetedNormals) {
  // calculateNormals with minSharpAngle=0 should produce a faceted result
  // where smoothing and refinement do not change the volume or surface area.
  // Regression test for https://github.com/elalish/manifold/issues/1577
  Manifold cylinder = Manifold::Cylinder(10, 10);
  Manifold faceted =
      cylinder.CalculateNormals(0).SmoothByNormals().RefineToLength(0.1);
  EXPECT_EQ(faceted.Status(), Manifold::Error::NoError);
  EXPECT_FLOAT_EQ(cylinder.Volume(), faceted.Volume());
  EXPECT_FLOAT_EQ(cylinder.SurfaceArea(), faceted.SurfaceArea());

  // The fix must survive rotation and translation, which change the face
  // normals and shift the dot products into different float neighborhoods.
  Manifold rotated = cylinder.Rotate(30, 45, 60).Translate({5, 10, 15});
  Manifold facetedRot =
      rotated.CalculateNormals(0).SmoothByNormals().RefineToLength(0.1);
  EXPECT_EQ(facetedRot.Status(), Manifold::Error::NoError);
  EXPECT_FLOAT_EQ(rotated.Volume(), facetedRot.Volume());
  EXPECT_FLOAT_EQ(rotated.SurfaceArea(), facetedRot.SurfaceArea());
}

TEST(Smooth, NormalTransform) {
  Manifold cube1 = Manifold::Cube().Rotate(30).CalculateNormals();
  Manifold cube2 =
      Manifold::Cube().CalculateNormals().Rotate(30).Translate({3, 0, 0});
  Manifold combo = cube1 + cube2;
  Manifold out1 = combo.SmoothByNormals().Refine(10);
  EXPECT_FLOAT_EQ(out1.Volume(), 2);
  EXPECT_FLOAT_EQ(out1.SurfaceArea(), 12);
  Manifold out2 = Manifold(combo.GetMeshGL()).SmoothByNormals().Refine(10);
  EXPECT_FLOAT_EQ(out2.Volume(), 2);
  EXPECT_FLOAT_EQ(out2.SurfaceArea(), 12);
}

TEST(Smooth, MissingNormals) {
  Manifold tetNorm = Manifold::Tetrahedron().CalculateNormals();
  Manifold diff = tetNorm - Manifold::Tetrahedron().Translate(vec3(0.5));
  Manifold out = diff.SmoothByNormals().Refine(10);
  EXPECT_NEAR(out.Volume(), 2.46, 0.01);
  EXPECT_NEAR(out.SurfaceArea(), 12.45, 0.01);
  if (options.exportModels) WriteTestOBJ("missingNormals.obj", out);
}

TEST(Smooth, MissingNormalChannels) {
  const Manifold cube = Manifold::Cube();
  // Fewer than three properties means no normal slot, equivalent to an
  // explicitly-zero normal slot. This guards GetNormal's short-circuit before
  // it reads properties 0..2.
  const double zeroNormals =
      cube.SetProperties(3, nullptr).SmoothByNormals().Refine(4).Volume();
  EXPECT_DOUBLE_EQ(cube.SmoothByNormals().Refine(4).Volume(), zeroNormals);
}

TEST(Smooth, GetMeshGLNormalizesNormals) {
  // Interpolation and a non-uniform transform both leave recorded normals
  // non-unit before GetMeshGL converts them into graphics output.
  const Manifold ellipsoid =
      Manifold::Sphere(1, {8}).CalculateNormals().Scale({2, 3, 4});
  const Manifold source = ellipsoid + ellipsoid.Translate({1, 0, 0});
  auto check = [](const auto& mesh, double tolerance) {
    ASSERT_GE(mesh.numProp, 6);
    ASSERT_GT(mesh.NumVert(), 0u);
    ASSERT_TRUE(mesh.HasNormals(0));
    for (size_t vert = 0; vert < mesh.NumVert(); ++vert) {
      const vec3 normal(mesh.vertProperties[vert * mesh.numProp + 3],
                        mesh.vertProperties[vert * mesh.numProp + 4],
                        mesh.vertProperties[vert * mesh.numProp + 5]);
      EXPECT_NEAR(la::length(normal), 1.0, tolerance);
    }
  };
  check(source.GetMeshGL(), 1e-5);
  check(source.GetMeshGL64(), 1e-12);
}

TEST(Smooth, MissingNormalsCone) {
  Manifold cone = Manifold::Cylinder(5, 10, 7, {6}).CalculateNormals(60);
  Manifold peak = Manifold::Cylinder(4, 7, 0, {6});
  Manifold biCone = cone + peak.Translate({0, 0, 5});
  Manifold out = biCone.SmoothByNormals().RefineToTolerance(0.01);
  EXPECT_NEAR(out.Volume(), 1480, 1);
  EXPECT_NEAR(out.SurfaceArea(), 828, 1);
  if (options.exportModels) WriteTestOBJ("missingNormalsCone.obj", out);
}

TEST(Smooth, OperandOrder) {
  auto mk = [](bool unique) {
    Manifold a = Manifold::Sphere(10, {16}).Scale({3, 1, 1});
    Manifold b = Manifold::Sphere(6, {16});
    if (unique) {
      a = a.AsOriginal();
      b = b.AsOriginal();
    }
    a = a.CalculateNormals(180).Rotate(0, 0, 45);
    b = b.Rotate(60, 0, 0).Translate({0, 0, 9});
    return std::make_pair(a, b);
  };
  auto smooth = [](const Manifold& m) {
    return m.SmoothByNormals().RefineToTolerance(0.01);
  };

  auto [a, b] = mk(false);
  auto [ua, ub] = mk(true);

  Manifold ref = smooth(ua + ub);
  Manifold good = smooth(a + b);
  Manifold bad = smooth(b + a);

  EXPECT_NEAR(good.Volume(), ref.Volume(), 0.01 * ref.Volume());
  EXPECT_NEAR(good.SurfaceArea(), ref.SurfaceArea(), 0.01 * ref.SurfaceArea());
  EXPECT_NEAR(bad.Volume(), ref.Volume(), 0.01 * ref.Volume());
  EXPECT_NEAR(bad.SurfaceArea(), ref.SurfaceArea(), 0.01 * ref.SurfaceArea());
  if (options.exportModels) {
    WriteTestOBJ("operandOrderRef.obj", ref);
    WriteTestOBJ("operandOrderGood.obj", good);
    WriteTestOBJ("operandOrderBad.obj", bad);
  }
}

TEST(Smooth, Fillet) {
  float depth = 3;
  float radius = 10;
  float filletScale = 1 + depth / radius;
  Manifold cylinder = Manifold::Cylinder(20, radius, radius, {6}, true)
                          .AsOriginal()
                          .CalculateNormals(80)
                          .Rotate(20);
  CrossSection section = cylinder.Slice(0);
  Manifold chamfer =
      Manifold::Extrude(section.Simplify().ToPolygons(), depth, 0, 0,
                        vec2(filletScale, 1.1 * filletScale))
          .Mirror({0, 0, 1});
  EXPECT_EQ(chamfer.NumTri(), 20);
  Manifold base = Manifold::Cylinder(10, 15, 15, {6})
                      .Translate({0, 0, -10 - depth})
                      .Scale({1, 1.2, 1})
                      .CalculateNormals(80);
  Manifold chamfered = cylinder + chamfer + base;
  EXPECT_EQ(chamfered.NumTri(), 56);
  Manifold fillet = chamfered.SmoothByNormals().RefineToTolerance(0.01);
  EXPECT_EQ(fillet.Status(), Manifold::Error::NoError);
  EXPECT_NEAR(fillet.Volume(), 12805, 1);
  EXPECT_NEAR(fillet.SurfaceArea(), 3452, 1);
  if (options.exportModels) WriteTestOBJ("fillet.obj", fillet);
}

TEST(Smooth, Fillet2) {
  float depth = 3;
  float radius = 10;
  float filletScale = 1 + depth / radius;
  Manifold cylinder = Manifold::Cylinder(40, radius, radius, {6}, true)
                          .AsOriginal()
                          .Rotate(0, 30)
                          .CalculateNormals(80);
  CrossSection section = cylinder.Slice(0);

  Manifold chamfer = Manifold::Extrude(section.Simplify().ToPolygons(), depth,
                                       0, 0, vec2(1.3, 1.2))
                         .Mirror({0, 0, 1});
  EXPECT_EQ(chamfer.NumTri(), 20);
  Manifold base = Manifold::Cube(vec3(40), true)
                      .Translate({0, 0, -20 - depth})
                      .CalculateNormals();
  Manifold chamfered = cylinder + chamfer + base;
  EXPECT_EQ(chamfered.NumTri(), 48);
  Manifold fillet = chamfered.SmoothByNormals().RefineToTolerance(0.01);
  EXPECT_EQ(fillet.Status(), Manifold::Error::NoError);
  EXPECT_NEAR(fillet.Volume(), 71507, 1);
  EXPECT_NEAR(fillet.SurfaceArea(), 10936, 1);
  if (options.exportModels) WriteTestOBJ("fillet2.obj", fillet);
}

TEST(Smooth, Manual) {
  // Unit Octahedron
  const auto oct = Manifold::Sphere(1, {4});
  MeshGL smooth = oct.CalculateNormals(180).SmoothByNormals().GetMeshGL();
  // Sharpen the edge from vert 4 to 5
  smooth.halfedgeTangent[4 * 6 + 3] = 0;
  smooth.halfedgeTangent[4 * 22 + 3] = 0;
  smooth.halfedgeTangent[4 * 16 + 3] = 0;
  smooth.halfedgeTangent[4 * 18 + 3] = 0;
  Manifold interp(smooth);
  interp = interp.Refine(100);

  ExpectMeshes(interp, {{40002, 80000, 3}});
  EXPECT_NEAR(interp.Volume(), 3.74, 0.01);
  EXPECT_NEAR(interp.SurfaceArea(), 11.78, 0.01);

  if (options.exportModels) {
    interp = interp.CalculateCurvature(-1, 3).SetProperties(
        3, [](double* newProp, vec3 pos, const double* oldProp) {
          const vec3 red(1, 0, 0);
          const vec3 purple(1, 0, 1);
          vec3 color = la::lerp(purple, red, smoothstep(0.0, 2.0, oldProp[3]));
          for (const int i : {0, 1, 2}) newProp[i] = color[i];
        });
    WriteTestOBJ("manual.obj", interp);
  }
}

TEST(Smooth, ExportTangentsFollowTriangleRuns) {
  const Manifold sphere =
      Manifold::Sphere(1, {12}).CalculateNormals(180).SmoothByNormals();
  const Manifold ellipsoid = sphere.Scale({1, 2, 3}).Translate({6, 0, 0});
  const Manifold tetrahedron =
      Manifold::Tetrahedron().CalculateNormals(180).SmoothByNormals().Translate(
          {-6, 0, 0});
  const std::vector<Manifold> cases = {
      sphere,
      ellipsoid,
      Manifold::BatchBoolean({sphere, ellipsoid}, OpType::Add),
      Manifold::BatchBoolean({ellipsoid, sphere}, OpType::Add),
      Manifold::BatchBoolean({ellipsoid, tetrahedron, sphere}, OpType::Add),
      Manifold::BatchBoolean(
          {ellipsoid, Manifold::Cube().Translate({-5, 0, 0})}, OpType::Add)};

  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    const Manifold& source = cases[i];
    const Manifold expected = source.Refine(3);
    ASSERT_EQ(expected.Status(), Manifold::Error::NoError);
    const auto check = [&](const auto& mesh, double tolerance) {
      const Manifold restored(mesh);
      ASSERT_EQ(restored.Status(), Manifold::Error::NoError);
      const Manifold actual = restored.Refine(3);
      ASSERT_EQ(actual.Status(), Manifold::Error::NoError);
      EXPECT_EQ(actual.NumTri(), expected.NumTri());
      EXPECT_NEAR(actual.Volume(), expected.Volume(), tolerance);
      EXPECT_NEAR(actual.SurfaceArea(), expected.SurfaceArea(), tolerance);
      const Manifold extra = actual - expected;
      const Manifold missing = expected - actual;
      ASSERT_EQ(extra.Status(), Manifold::Error::NoError);
      ASSERT_EQ(missing.Status(), Manifold::Error::NoError);
      EXPECT_NEAR(extra.Volume(), 0, tolerance);
      EXPECT_NEAR(missing.Volume(), 0, tolerance);
    };
    check(source.GetMeshGL(), 1e-4);
    check(source.GetMeshGL64(), 1e-9);
  }
}

TEST(Smooth, InvalidTangents) {
  Manifold cube = Manifold::Cube().CalculateNormals(180).SmoothByNormals();
  MeshGL withTangents = cube.GetMeshGL();
  size_t sizeHalfedges = withTangents.halfedgeTangent.size();
  for (size_t i = (sizeHalfedges / 8) * 4 + 3; i < sizeHalfedges; i += 4) {
    // mark the second half of the tangents as missing, which is incorrect.
    withTangents.halfedgeTangent[i] = -1;
  }
  Manifold cube2(withTangents);
  Manifold smooth = cube2.Refine(10);
  EXPECT_EQ(smooth.Status(), Manifold::Error::InvalidTangents);
}

TEST(Smooth, Mirrored) {
  const auto tet = Manifold::Tetrahedron().Scale({1, 2, 3});
  Manifold smooth = tet.CalculateNormals().SmoothByNormals();
  Manifold mirror = smooth.Scale({-2, 2, 2}).Refine(10);
  smooth = smooth.Refine(10).Scale({2, 2, 2});

  EXPECT_NEAR(smooth.Volume(), mirror.Volume(), 0.1);
  EXPECT_NEAR(smooth.SurfaceArea(), mirror.SurfaceArea(), 0.1);

  if (options.exportModels) WriteTestOBJ("smoothMirrored.obj", mirror);
}

TEST(Smooth, Csaszar) {
  Manifold csaszar(Csaszar());
  csaszar = csaszar.CalculateNormals(180).SmoothByNormals().Refine(100);
  ExpectMeshes(csaszar, {{70000, 140000, 3}});
  EXPECT_NEAR(csaszar.Volume(), 74757, 10);
  EXPECT_NEAR(csaszar.SurfaceArea(), 11313, 10);

  if (options.exportModels) WriteTestOBJ("smoothCsaszar.obj", csaszar);
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

TEST(Smooth, Torus) {
  MeshGL64 torusMesh =
      Manifold::Revolve(
          CrossSection::Circle(1, {8}).Translate({2, 0}).ToPolygons(), {6})
          .GetMeshGL64();
  const int numTri = torusMesh.NumTri();

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
      } else if (std::abs(la::determinant(mat2(vec2(v), vec2(edge)))) < 1e-5) {
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
                        .CalculateNormals()
                        .CalculateCurvature(-1, 3);
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
        std::max(maxMeanCurvature, std::abs(out.vertProperties[i + 6]));
  }
  EXPECT_NEAR(maxMeanCurvature, 1.63, 0.01);

  if (options.exportModels) WriteTestOBJ("smoothTorus.obj", smooth);
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
          .SmoothByNormals()
          .RefineToLength(0.1)
          .SetProperties(1, error);

  MeshGL out = smoothed.GetMeshGL();
  EXPECT_NEAR(GetMaxProperty(out, 3), 0, 0.035);
  EXPECT_NEAR(GetMaxProperty(interpolated.GetMeshGL(), 3), 0, 0.083);

  if (options.exportModels) WriteTestOBJ("smoothGyroid.obj", smoothed);
}

TEST(Smooth, RefineInvalid) {
  const Manifold cube = Manifold::Cube();
  for (const double length : {0.0, -0.0, std::nan("")}) {
    EXPECT_EQ(cube.RefineToLength(length).Status(),
              Manifold::Error::InvalidConstruction);
  }
  EXPECT_EQ(
      cube.RefineToLength(std::numeric_limits<double>::infinity()).NumTri(),
      cube.NumTri());

  const Manifold smooth = cube.CalculateNormals(30).SmoothByNormals();
  for (const double tolerance : {0.0, -0.0, std::nan("")}) {
    EXPECT_EQ(smooth.RefineToTolerance(tolerance).Status(),
              Manifold::Error::InvalidConstruction);
  }
  EXPECT_EQ(smooth.RefineToTolerance(std::numeric_limits<double>::infinity())
                .NumTri(),
            smooth.NumTri());
}

TEST(Smooth, RefineTooLarge) {
  const Manifold cube = Manifold::Cube();
  EXPECT_EQ(cube.RefineToLength(1e-9).Status(),
            Manifold::Error::ResultTooLarge);
  EXPECT_EQ(cube.RefineToLength(1e-300).Status(),
            Manifold::Error::ResultTooLarge);
  EXPECT_EQ(cube.Refine(1 << 20).Status(), Manifold::Error::ResultTooLarge);
  const Manifold smooth =
      Manifold::Cylinder(1, 1, 1, {8}).CalculateNormals().SmoothByNormals();
  ASSERT_GT(smooth.RefineToTolerance(0.01).NumTri(), smooth.NumTri());
  EXPECT_EQ(smooth.RefineToTolerance(1e-30).Status(),
            Manifold::Error::ResultTooLarge);
}

TEST(Smooth, RefinePropertiesTooLarge) {
  const Manifold props =
      Manifold::Cube().SetProperties(1000, [](double*, vec3, const double*) {});
  EXPECT_EQ(props.Refine(1000).Status(), Manifold::Error::ResultTooLarge);
}

TEST(Smooth, SphereTooLarge) {
  EXPECT_EQ(Manifold::Sphere(1, {1 << 20}).Status(),
            Manifold::Error::ResultTooLarge);
}

TEST(Smooth, RefineFlatTriangles) {
  // The wedge's caps are long, nearly flat triangles.
  const Manifold wedge = Manifold::Extrude({{{0, 0}, {2, 0}, {1, 1e-3}}}, 1e-3);
  const Manifold refined = wedge.RefineToLength(7e-5);
  EXPECT_EQ(refined.Status(), Manifold::Error::NoError);
  EXPECT_GT(refined.NumTri(), 1000000);
}

TEST(Smooth, RefineLongEdges) {
  // 46342 divisions per long edge, where squaring an int division count
  // overflows.
  const Manifold refined =
      Manifold::Cube({46342, 1, 1}, true).RefineToLength(1);
  EXPECT_EQ(refined.Status(), Manifold::Error::NoError);
  EXPECT_NEAR(refined.Volume(), 46342, 1e-6);
  EXPECT_EQ(refined.Genus(), 0);
  EXPECT_GT(refined.NumTri(), 1112208);
}
