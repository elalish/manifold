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

#include "manifold/manifold.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "../src/execution_impl.h"
#include "gtest/gtest.h"
#include "manifold/cross_section.h"
#include "test.h"

namespace {

using namespace manifold;

bool SameMeshPayload(const MeshGL64& a, const MeshGL64& b) {
  return std::tie(a.numProp, a.vertProperties, a.triVerts, a.mergeFromVert,
                  a.mergeToVert, a.runIndex, a.runOriginalID, a.runTransform,
                  a.runFlags, a.faceID, a.halfedgeTangent) ==
         std::tie(b.numProp, b.vertProperties, b.triVerts, b.mergeFromVert,
                  b.mergeToVert, b.runIndex, b.runOriginalID, b.runTransform,
                  b.runFlags, b.faceID, b.halfedgeTangent);
}

template <typename T>
int NumUnique(const std::vector<T>& in) {
  std::set<int> unique;
  for (const T& v : in) {
    unique.emplace(v);
  }
  return unique.size();
}

// A disjoint Union of a CubeSTL (no hasNormals) and a Sphere with
// CalculateNormals. The result has mixed meshIDs - some with hasNormals,
// some without - the exact shape the per-meshID handling in
// Impl::Transform / Compose / CreateProperties was added to support.
Manifold MixedNormalsCubePlusSphere(double sphereRadius = 5.0) {
  return Manifold(CubeSTL().Merge()).Translate({20, 0, 0}) +
         Manifold::Sphere(sphereRadius, {32}).CalculateNormals();
}

// Count verts on the sphere surface (|pos| ~ sphereRadius) whose stored
// normal at slot 3..5+offset aligns with `expected(pos)` (dot > 0.9).
// Returns (good, bad).
template <typename ExpectedFn>
std::pair<int, int> CountSphereNormalAlignment(const MeshGL& gl,
                                               double sphereRadius,
                                               int normalSlotOffset,
                                               ExpectedFn expected) {
  int good = 0, bad = 0;
  for (size_t v = 0; v < gl.NumVert(); ++v) {
    const vec3 pos(gl.vertProperties[v * gl.numProp + 0],
                   gl.vertProperties[v * gl.numProp + 1],
                   gl.vertProperties[v * gl.numProp + 2]);
    if (std::abs(la::length(pos) - sphereRadius) > 0.1) continue;
    const vec3 n(gl.vertProperties[v * gl.numProp + normalSlotOffset + 0],
                 gl.vertProperties[v * gl.numProp + normalSlotOffset + 1],
                 gl.vertProperties[v * gl.numProp + normalSlotOffset + 2]);
    if (la::dot(n, expected(pos)) > 0.9)
      ++good;
    else
      ++bad;
  }
  return {good, bad};
}

}  // namespace

/**
 * This tests that turning a mesh into a manifold and returning it to a mesh
 * produces a consistent result.
 */
TEST(Manifold, GetMeshGL) {
  Manifold manifold = Manifold::Sphere(0.01);
  auto mesh_out = manifold.GetMeshGL();
  Manifold manifold2(mesh_out);
  auto mesh_out2 = manifold2.GetMeshGL();
  Identical(mesh_out, mesh_out2);
}

TEST(Manifold, SharedTopologyCopyIsolation) {
  const Manifold sphere =
      Manifold::Sphere(1, {16}).CalculateNormals(180).SmoothByNormals();
  const Manifold cube = Manifold::Cube().Refine(2).CalculateNormals(30);
  for (const Manifold& source : {sphere, cube}) {
    const MeshGL64 original = source.GetMeshGL64();
    const std::vector<std::function<Manifold(const Manifold&)>> operations = {
        [](const Manifold& m) { return m.CalculateNormals(30); },
        [](const Manifold& m) { return m.Refine(2); },
        [](const Manifold& m) { return m.Simplify(0.02); },
        [](const Manifold& m) { return m.RemoveDegenerates(); },
        [](const Manifold& m) { return m.Mirror({1, 2, 3}); },
        [](const Manifold& m) { return m.AsOriginal(); },
        [](const Manifold& m) { return m.CalculateCurvature(3, 4); },
        [](const Manifold& m) { return m.SetProperties(1, nullptr); },
        [](const Manifold& m) { return m.SmoothByNormals(); },
        [](const Manifold& m) {
          return m.Warp([](vec3& p) { p.x += 0.1 * p.z * p.z; });
        },
        [](const Manifold& m) {
          return m.WarpBatch([](VecView<vec3> points) {
            for (vec3& p : points) p.y += 0.1 * p.x * p.x;
          });
        },
        [](const Manifold& m) {
          return m - Manifold::Cube({0.5, 0.5, 0.5}, true);
        }};
    for (size_t i = 0; i < operations.size(); ++i) {
      SCOPED_TRACE(i);
      const Manifold result = operations[i](source);
      EXPECT_EQ(result.Status(), Manifold::Error::NoError);
      EXPECT_TRUE(SameMeshPayload(source.GetMeshGL64(), original));
    }
    const Manifold invalid = source.WarpBatch([](VecView<vec3> points) {
      points[0].x = std::numeric_limits<double>::quiet_NaN();
    });
    EXPECT_EQ(invalid.Status(), Manifold::Error::NonFiniteVertex);
    EXPECT_TRUE(SameMeshPayload(source.GetMeshGL64(), original));
  }
}

#if (defined(__cpp_exceptions) || defined(_CPPUNWIND)) && \
    !defined(__EMSCRIPTEN__)
TEST(Manifold, SharedTopologyThrowingCallbacks) {
  const Manifold source = Manifold::Cube().CalculateNormals(30);
  const MeshGL64 original = source.GetMeshGL64();
  EXPECT_THROW(source.WarpBatch([](VecView<vec3> points) {
    points[0].x += 1;
    throw std::runtime_error("warp callback");
  }),
               std::runtime_error);
  EXPECT_TRUE(SameMeshPayload(source.GetMeshGL64(), original));
  EXPECT_THROW(
      source.SetProperties(3,
                           [](double* prop, vec3, const double*) {
                             prop[0] = 42;
                             throw std::runtime_error("property callback");
                           }),
      std::runtime_error);
  EXPECT_TRUE(SameMeshPayload(source.GetMeshGL64(), original));
}
#endif

#ifndef __EMSCRIPTEN__
TEST(Manifold, SharedTopologyConcurrentCopies) {
  const Manifold source = Manifold::Sphere(1, {16}).CalculateNormals(180);
  const MeshGL64 original = source.GetMeshGL64();
  std::atomic<bool> intact = true;
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&] {
      for (int iteration = 0; iteration < 10; ++iteration) {
        const Manifold result = source.CalculateNormals(30)
                                    .Refine(2)
                                    .Mirror({1, 2, 3})
                                    .SetProperties(1, nullptr);
        if (result.Status() != Manifold::Error::NoError ||
            !SameMeshPayload(source.GetMeshGL64(), original)) {
          intact = false;
        }
      }
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_TRUE(intact);
  EXPECT_TRUE(SameMeshPayload(source.GetMeshGL64(), original));
}
#endif

TEST(Manifold, WarpNonFiniteResults) {
  const Manifold cube = Manifold::Cube();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double value :
       {inf, -inf, std::numeric_limits<double>::quiet_NaN()}) {
    for (int axis : {0, 1, 2}) {
      SCOPED_TRACE(testing::Message()
                   << "value " << value << ", axis " << axis);
      const Manifold warped = cube.Warp([=](vec3& p) { p[axis] = value; });
      EXPECT_EQ(warped.Status(), Manifold::Error::NonFiniteVertex);
      for (bool all : {false, true}) {
        SCOPED_TRACE(all);
        const Manifold batch = cube.WarpBatch([=](VecView<vec3> points) {
          for (size_t i = 0; i < (all ? points.size() : 1); ++i)
            points[i][axis] = value;
        });
        EXPECT_EQ(batch.Status(), Manifold::Error::NonFiniteVertex);
      }
    }
  }
  const Manifold finite = cube.Warp([](vec3& p) { p *= 2.0; });
  EXPECT_EQ(finite.Status(), Manifold::Error::NoError);
  EXPECT_NEAR(finite.Volume(), 8, 1e-12);
  EXPECT_EQ(cube.Status(), Manifold::Error::NoError);
  EXPECT_NEAR(cube.Volume(), 1, 1e-12);
  const Manifold empty = Manifold().WarpBatch([](VecView<vec3>) {});
  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_EQ(empty.Status(), Manifold::Error::NoError);
}

TEST(Manifold, MeshDeterminism) {
  Manifold cube1 = Manifold::Cube(vec3(2.0, 2.0, 2.0), true);
  Manifold cube2 = Manifold::Cube(vec3(2.0, 2.0, 2.0), true)
                       .Translate(vec3(-1.1091, 0.88509, 1.3099));

  Manifold result = cube1 - cube2;
  MeshGL out = result.GetMeshGL();

  uint32_t triVerts[]{0, 2,  7,  0,  10, 1,  0,  6,  10, 0,  1,  2, 1, 3,  2,
                      1, 5,  3,  4,  2,  3,  1,  11, 5,  4,  3,  5, 0, 7,  6,
                      6, 7,  8,  4,  7,  2,  4,  5,  8,  4,  8,  7, 6, 8,  13,
                      9, 8,  5,  10, 12, 11, 1,  10, 11, 11, 13, 5, 6, 12, 10,
                      6, 13, 12, 13, 9,  5,  13, 8,  9,  11, 12, 13};

  float vertProperties[]{-1,      -1,       -1,     -1,      -1,       1,
                         -1,      -0.11491, 0.3099, -1,      -0.11491, 1,
                         -0.1091, -0.11491, 0.3099, -0.1091, -0.11491, 1,
                         -1,      1,        -1,     -1,      1,        0.3099,
                         -0.1091, 1,        0.3099, -0.1091, 1,        1,
                         1,       -1,       -1,     1,       -1,       1,
                         1,       1,        -1,     1,       1,        1};

  bool flag = true;
  for (size_t i = 0; i != out.triVerts.size(); i++) {
    if (out.triVerts[i] != triVerts[i]) {
      flag = false;
      break;
    }
  }

  for (size_t i = 0; flag && i != out.vertProperties.size(); i++) {
    if (out.vertProperties[i] != vertProperties[i]) {
      flag = false;
      break;
    }
  }

  EXPECT_TRUE(flag);
}

TEST(Manifold, Empty) {
  MeshGL emptyMesh;
  Manifold empty(emptyMesh);

  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_EQ(empty.Status(), Manifold::Error::NoError);
}

TEST(Manifold, ErrorStrings) {
  using Error = Manifold::Error;
  const std::string unknownName = ErrorName(static_cast<Error>(-1));
  const std::string unknown = ToString(static_cast<Error>(-1));
  std::set<std::string> names;
  std::set<std::string> messages;
  for (int i = 0; i <= static_cast<int>(Error::Cancelled); ++i) {
    const std::string name = ErrorName(static_cast<Error>(i));
    const std::string message = ToString(static_cast<Error>(i));
    EXPECT_FALSE(name.empty());
    EXPECT_FALSE(message.empty());
    EXPECT_NE(name, unknownName);
    EXPECT_NE(message, unknown);
    names.insert(name);
    messages.insert(message);
  }
  EXPECT_EQ(names.size(), static_cast<size_t>(Error::Cancelled) + 1);
  EXPECT_EQ(messages.size(), static_cast<size_t>(Error::Cancelled) + 1);
  EXPECT_STREQ(ErrorName(Error::Cancelled), "Cancelled");
  const auto past = static_cast<Error>(static_cast<int>(Error::Cancelled) + 1);
  EXPECT_EQ(ErrorName(past), unknownName);
  EXPECT_EQ(ToString(past), unknown);
}

TEST(Manifold, ValidInput) {
  MeshGL tetGL = TetGL();
  Manifold tet(tetGL);
  EXPECT_FALSE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NoError);
}

TEST(Manifold, ValidInputOneRunIndex) {
  MeshGL emptyMesh;
  emptyMesh.runIndex = {0};
  Manifold empty(emptyMesh);
  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_EQ(empty.Status(), Manifold::Error::NoError);
}

TEST(Manifold, InvalidInput1) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, InvalidInput2) {
  MeshGL in = TetGL();
  std::swap(in.triVerts[2 * 3 + 1], in.triVerts[2 * 3 + 2]);
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NotManifold);
}

TEST(Manifold, InvalidInput3) {
  MeshGL in = TetGL();
  for (uint32_t& triVert : in.triVerts) {
    if (triVert == 2) triVert = -2;
  }
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

TEST(Manifold, InvalidInput4) {
  MeshGL in = TetGL();
  for (uint32_t& triVert : in.triVerts) {
    if (triVert == 2) triVert = 4;
  }
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NotManifold);
}

TEST(Manifold, InvalidInput5) {
  MeshGL tetGL = TetGL();
  tetGL.mergeFromVert[tetGL.mergeFromVert.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::MergeIndexOutOfBounds);
}

TEST(Manifold, InvalidInputOddHalfedgeCount) {
  // An odd number of triangles leaves one halfedge without a partner; the
  // mesh must be rejected, not paired through an uninitialized index.
  MeshGL in = TetGL();
  in.triVerts.insert(in.triVerts.end(), {2, 0, 1});
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NotManifold);
}

TEST(Manifold, InvalidInput6) {
  MeshGL tetGL = TetGL();
  tetGL.triVerts[tetGL.triVerts.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

TEST(Manifold, InvalidInput64BitIndex) {
  // Indices past 2^32 must not wrap around to a valid 32-bit index.
  const MeshGL tetGL = TetGL();
  MeshGL64 tet;
  tet.numProp = tetGL.numProp;
  tet.vertProperties.assign(tetGL.vertProperties.begin(),
                            tetGL.vertProperties.end());
  tet.triVerts.assign(tetGL.triVerts.begin(), tetGL.triVerts.end());
  tet.mergeFromVert.assign(tetGL.mergeFromVert.begin(),
                           tetGL.mergeFromVert.end());
  tet.mergeToVert.assign(tetGL.mergeToVert.begin(), tetGL.mergeToVert.end());
  ASSERT_EQ(Manifold(tet).Status(), Manifold::Error::NoError);

  const uint64_t wrap = uint64_t(1) << 32;
  MeshGL64 badTri = tet;
  badTri.triVerts[1] += wrap;
  EXPECT_EQ(Manifold(badTri).Status(), Manifold::Error::VertexOutOfBounds);

  MeshGL64 badMergeTo = tet;
  badMergeTo.mergeToVert[0] += wrap;
  EXPECT_EQ(Manifold(badMergeTo).Status(),
            Manifold::Error::MergeIndexOutOfBounds);

  MeshGL64 badMergeFrom = tet;
  badMergeFrom.mergeFromVert[0] += wrap;
  EXPECT_EQ(Manifold(badMergeFrom).Status(),
            Manifold::Error::MergeIndexOutOfBounds);
}

TEST(Manifold, InvalidInputPropertiesWrongLength) {
  // vertProperties written with 4 floats per vertex, but numProp says 3.
  const MeshGL tet = Manifold::Tetrahedron().GetMeshGL();
  ASSERT_EQ(tet.numProp, 3);
  MeshGL stride = tet;
  stride.vertProperties.clear();
  for (size_t v = 0; v < tet.NumVert(); ++v) {
    for (int j = 0; j < 3; ++j)
      stride.vertProperties.push_back(tet.vertProperties[3 * v + j]);
    stride.vertProperties.push_back(0.5f);
  }
  EXPECT_EQ(Manifold(stride).Status(), Manifold::Error::PropertiesWrongLength);

  // A trailing partial vertex, with numProp = 5.
  MeshGL partial = TetGL();
  partial.vertProperties.insert(partial.vertProperties.end(), {1, 1});
  EXPECT_EQ(Manifold(partial).Status(), Manifold::Error::PropertiesWrongLength);

  MeshGL64 partial64;
  partial64.numProp = partial.numProp;
  partial64.vertProperties.assign(partial.vertProperties.begin(),
                                  partial.vertProperties.end());
  partial64.triVerts.assign(partial.triVerts.begin(), partial.triVerts.end());
  partial64.mergeFromVert.assign(partial.mergeFromVert.begin(),
                                 partial.mergeFromVert.end());
  partial64.mergeToVert.assign(partial.mergeToVert.begin(),
                               partial.mergeToVert.end());
  EXPECT_EQ(Manifold(partial64).Status(),
            Manifold::Error::PropertiesWrongLength);

  // Merge leaves a malformed MeshGL unchanged instead of reading the
  // trailing float as a coordinate.
  MeshGL unmerged = CubeSTL();
  ASSERT_FALSE(unmerged.Merge().mergeFromVert.empty());
  unmerged.vertProperties.push_back(1e7f);
  EXPECT_TRUE(unmerged.Merge().mergeFromVert.empty());
}

TEST(Manifold, InvalidInput7) {
  MeshGL cube = CubeUV();
  cube.runIndex = {0, 1, static_cast<uint32_t>(cube.triVerts.size())};
  Manifold tet(cube);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::RunIndexWrongLength);
}

TEST(Manifold, InvalidInputRunIndexPastEnd) {
  MeshGL tetGL = TetGL();
  tetGL.runIndex = {0, 3000};
  tetGL.runOriginalID = {1};
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::RunIndexWrongLength);
}

TEST(Manifold, InvalidInputRunIndexDecreasing) {
  MeshGL tetGL = TetGL();
  tetGL.runIndex = {0, 9, 6, static_cast<uint32_t>(tetGL.triVerts.size())};
  tetGL.runOriginalID = {1, 2, 3};
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::RunIndexWrongLength);
}

TEST(Manifold, InvalidInputRunIndexStart) {
  MeshGL tetGL = TetGL();
  tetGL.runIndex = {6};
  tetGL.runOriginalID = {1};
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::RunIndexWrongLength);
}

TEST(Manifold, InvalidInputRunIDsWithoutIndex) {
  MeshGL tetGL = TetGL();
  tetGL.runIndex.clear();
  tetGL.runOriginalID = {1, 2};
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::RunIndexWrongLength);
}

TEST(Manifold, InvalidInputShortTangents) {
  MeshGL tetGL = TetGL();
  tetGL.halfedgeTangent = {1, 1, 1, 1};
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::InvalidConstruction);
}

TEST(Manifold, LongTangents) {
  MeshGL tetGL = TetGL();
  tetGL.halfedgeTangent.resize(4 * tetGL.triVerts.size() + 4, 0);
  EXPECT_EQ(Manifold(tetGL).Status(), Manifold::Error::NoError);
}

TEST(Manifold, InvalidInputNoProperties) {
  MeshGL in;
  in.numProp = 0;
  in.vertProperties.resize(15);
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::MissingPositionProperties);
}

TEST(Manifold, ErrorPropagationDecompose) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  auto parts = errored.Decompose();
  ASSERT_EQ(parts.size(), 1);
  EXPECT_EQ(parts[0].Status(), Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationHull) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.Hull().Status(), Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationHullMulti) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  Manifold good = Manifold::Cube();
  EXPECT_EQ(Manifold::Hull({good, errored}).Status(),
            Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationSetProperties) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.SetProperties(1, nullptr).Status(),
            Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationCalculateCurvature) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.CalculateCurvature(0, 1).Status(),
            Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationCalculateNormals) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.CalculateNormals().Status(),
            Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, NormalsCavity) {
  // The #1712 repro: inner-sphere normals from a Boolean diff should point
  // toward the origin (outward from the surrounding solid).
  MeshGL mesh = (Manifold::Sphere(10.0, {32}) - Manifold::Sphere(3.0, {32}))
                    .CalculateNormals()
                    .GetMeshGL();
  ASSERT_GE(mesh.numProp, 6);
  auto [good, bad] = CountSphereNormalAlignment(
      mesh, 3.0, 3, [](vec3 pos) { return la::normalize(-pos); });
  EXPECT_GT(good, 0);
  EXPECT_EQ(bad, 0);
}

TEST(Manifold, NormalsRotateBeforeCalc) {
  // Rotation before CalculateNormals: SetNormals computes from already-
  // rotated faceNormal_ and stores world-frame at slot 0.
  MeshGL mesh = Manifold::Sphere(10.0, {32})
                    .Rotate(45, 0, 0)
                    .CalculateNormals()
                    .GetMeshGL();
  auto [_, bad] = CountSphereNormalAlignment(
      mesh, 10.0, 3, [](vec3 pos) { return la::normalize(pos); });
  EXPECT_EQ(bad, 0);
}

TEST(Manifold, LargeRotationsPreserveVolume) {
  const Manifold cube = Manifold::Cube({2, 3, 4});
  for (const double angle : {1e18, -1e18, 1e20, -1e20}) {
    for (int axis = 0; axis < 3; ++axis) {
      vec3 angles(0.0);
      angles[axis] = angle;
      const Manifold rotated = cube.Rotate(angles.x, angles.y, angles.z);
      ASSERT_EQ(rotated.Status(), Manifold::Error::NoError);
      EXPECT_NEAR(rotated.Volume(), cube.Volume(), 1e-10);
    }
  }
}

TEST(Manifold, NormalsRotateAfterCalc) {
  // Rotation *after* CalculateNormals: Impl::Transform eager-transforms the
  // stored slot 0..2 so it tracks the new orientation.
  MeshGL mesh = Manifold::Sphere(10.0, {32})
                    .CalculateNormals()
                    .Rotate(45, 0, 0)
                    .GetMeshGL();
  auto [_, bad] = CountSphereNormalAlignment(
      mesh, 10.0, 3, [](vec3 pos) { return la::normalize(pos); });
  EXPECT_EQ(bad, 0);
}

TEST(Manifold, NormalsAutoSubstitute) {
  // No-arg invocation: defaults to slot 0 and sets the per-run hasNormals
  // bit on every output run.
  MeshGL mesh = Manifold::Sphere(10.0, {32}).CalculateNormals().GetMeshGL();
  ASSERT_GE(mesh.numProp, 6);
  ASSERT_GT(mesh.NumRun(), 0u);
  EXPECT_TRUE(mesh.HasNormals(0));
}

TEST(Manifold, NormalsRoundTrip) {
  // GetMeshGL -> Manifold(mesh) -> GetMeshGL preserves the per-run flag, so
  // the second GetMeshGL still emits world-frame normals.
  Manifold round = (Manifold::Sphere(10.0, {32}) - Manifold::Sphere(3.0, {32}))
                       .CalculateNormals();
  MeshGL out1 = round.GetMeshGL();
  EXPECT_TRUE(out1.HasNormals(0));
  MeshGL out2 = Manifold(out1).GetMeshGL();
  EXPECT_TRUE(out2.HasNormals(0));
  auto [good, bad] = CountSphereNormalAlignment(
      out2, 3.0, 3, [](vec3 pos) { return la::normalize(-pos); });
  EXPECT_GT(good, 0);
  EXPECT_EQ(bad, 0);
}

TEST(Manifold, NormalsRefinePreserved) {
  // Refine keeps the recording: linearly-interpolated normals at the new
  // verts are less precise than recomputed ones but still meaningful.
  MeshGL mesh =
      Manifold::Sphere(10.0, {32}).CalculateNormals().Refine(2).GetMeshGL();
  ASSERT_GT(mesh.NumRun(), 0u);
  EXPECT_TRUE(mesh.HasNormals(0));
}

TEST(Manifold, NormalsSmoothByNormalsNoArg) {
  // Smoke test that the no-arg SmoothByNormals reads from the recorded
  // slot 0 and produces a valid manifold.
  Manifold smoothed =
      Manifold::Sphere(10.0, {32}).CalculateNormals().SmoothByNormals();
  EXPECT_EQ(smoothed.Status(), Manifold::Error::NoError);
}

TEST(Manifold, NormalsDroppedWithChannels) {
  // With fewer than three properties left, the recorded normals no longer
  // exist, so export must not write them into the next vertex.
  const Manifold m = (Manifold::Cube(vec3(2), true) + Manifold::Sphere(5, {8}))
                         .CalculateNormals()
                         .SetProperties(1, nullptr);
  const MeshGL64 mesh = m.GetMeshGL64();
  ASSERT_EQ(mesh.numProp, 4);
  for (size_t run = 0; run < mesh.runOriginalID.size(); ++run)
    EXPECT_FALSE(mesh.HasNormals(run));
  for (size_t v = 0; v < mesh.NumVert(); ++v)
    EXPECT_EQ(mesh.vertProperties[4 * v + 3], 0) << v;
  EXPECT_NEAR(Manifold(mesh).Volume(), m.Volume(), 1e-9);
}

TEST(Manifold, NormalsNormalizedPerRun) {
  // GetMeshGL normalizes each run that records normals, even when other runs
  // don't. The overlap makes the Boolean interpolate the sphere's normals to
  // less than unit length. The cube's unflagged slot 0..2 is left alone.
  const Manifold cube =
      Manifold::Cube(vec3(6), true)
          .Translate({4, 0, 0})
          .SetProperties(3, [](double* newProp, vec3, const double*) {
            newProp[0] = 0;
            newProp[1] = 0;
            newProp[2] = 2;
          });
  const Manifold mixed = Manifold::Sphere(5, {32}).CalculateNormals() + cube;
  auto check = [](const auto& gl) {
    int flagged = 0;
    int unflagged = 0;
    for (size_t run = 0; run < gl.NumRun(); ++run) {
      for (size_t i = gl.runIndex[run]; i < gl.runIndex[run + 1]; ++i) {
        const size_t v = gl.triVerts[i];
        const vec3 n(gl.vertProperties[v * gl.numProp + 3],
                     gl.vertProperties[v * gl.numProp + 4],
                     gl.vertProperties[v * gl.numProp + 5]);
        if (gl.HasNormals(run)) {
          ++flagged;
          EXPECT_NEAR(la::length(n), 1, 1e-6) << v;
        } else {
          ++unflagged;
          EXPECT_NEAR(la::length(n - vec3(0, 0, 2)), 0, 1e-6) << v;
        }
      }
    }
    EXPECT_GT(flagged, 0);
    EXPECT_GT(unflagged, 0);
  };
  check(mixed.GetMeshGL());
  check(mixed.GetMeshGL64());
}

TEST(Manifold, NormalsSharedPropVertMixedFlagsUndefined) {
  // hasNormals is per-run, but a single propVert holds one slot 0..2
  // value. If a hand-built MeshGL shares a propVert between a
  // hasNormals=true run and a hasNormals=false run, a Transform rotates
  // the slot on behalf of the hasNormals=true camp; the other camp's
  // interpretation (e.g. color) is collateral damage. Standard
  // CalculateNormals and Boolean outputs never share propVerts
  // this way - this test pins the documented behaviour for the only
  // input shape that can produce it.
  MeshGL gl;
  gl.numProp = 6;
  const float pts[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int v = 0; v < 4; ++v) {
    for (int j : {0, 1, 2}) gl.vertProperties.push_back(pts[v][j]);
    // Slot 0..2: +Z. Run 0 treats as normal, Run 1 treats as color.
    gl.vertProperties.push_back(0);
    gl.vertProperties.push_back(0);
    gl.vertProperties.push_back(1);
  }
  gl.triVerts = {0, 1, 3, 0, 2, 1, 0, 3, 2, 1, 2, 3};
  gl.runOriginalID = {Manifold::ReserveIDs(1), Manifold::ReserveIDs(1)};
  gl.runIndex = {0, 6, 12};
  gl.runFlags = {0x02, 0x00};  // run 0 hasNormals, run 1 plain color

  Manifold m(gl);
  ASSERT_EQ(m.Status(), Manifold::Error::NoError);

  MeshGL out = m.Rotate(90, 0, 0).GetMeshGL();

  int run0Bad = 0;  // hasNormals camp: slot rotates as expected.
  int run1Bad = 0;  // no-normals camp: collateral, slot is now rotated.
  for (size_t run = 0; run < out.NumRun(); ++run) {
    const bool isNormalsRun = out.HasNormals(run);
    for (uint32_t i = out.runIndex[run]; i < out.runIndex[run + 1]; ++i) {
      const uint32_t v = out.triVerts[i];
      const vec3 val(out.vertProperties[v * out.numProp + 3],
                     out.vertProperties[v * out.numProp + 4],
                     out.vertProperties[v * out.numProp + 5]);
      if (isNormalsRun) {
        const vec3 expected(0, -1, 0);
        if (la::length(val - expected) > 0.01) ++run0Bad;
      } else {
        const vec3 expected(0, 0, 1);
        if (la::length(val - expected) > 0.01) ++run1Bad;
      }
    }
  }
  EXPECT_EQ(run0Bad, 0);
  EXPECT_GT(run1Bad, 0);
}

TEST(Manifold, GetNormalUnflaggedMeshFrame) {
  // A hand-built MeshGL without hasNormals stores slot 0..2 in the mesh frame.
  // Recompose those values with the run transform before smoothing.
  const Manifold rotated =
      Manifold::Cube({1, 1, 1}, true).Rotate(30, 45, 0).CalculateNormals();
  MeshGL gl_modern = rotated.GetMeshGL();
  ASSERT_GT(gl_modern.NumRun(), 0u);
  ASSERT_TRUE(gl_modern.HasNormals(0));

  MeshGL gl_mesh_frame = gl_modern;
  std::vector<bool> visited(gl_mesh_frame.NumVert(), false);
  for (size_t run = 0; run < gl_mesh_frame.NumRun(); ++run) {
    // Convert world-frame values to the input mesh frame for this run.
    const mat3 fwd =
        la::inverse(la::transpose(mat3(gl_mesh_frame.GetRunTransform(run)))) *
        (gl_mesh_frame.Backside(run) ? -1.0 : 1.0);
    const mat3 inv = la::inverse(fwd);
    for (uint32_t* itr =
             gl_mesh_frame.triVerts.data() + gl_mesh_frame.runIndex[run];
         itr < gl_mesh_frame.triVerts.data() + gl_mesh_frame.runIndex[run + 1];
         ++itr) {
      const uint32_t v = *itr;
      if (visited[v]) continue;
      visited[v] = true;
      vec3 n(gl_mesh_frame.vertProperties[v * gl_mesh_frame.numProp + 3],
             gl_mesh_frame.vertProperties[v * gl_mesh_frame.numProp + 4],
             gl_mesh_frame.vertProperties[v * gl_mesh_frame.numProp + 5]);
      n = inv * n;
      gl_mesh_frame.vertProperties[v * gl_mesh_frame.numProp + 3] = n.x;
      gl_mesh_frame.vertProperties[v * gl_mesh_frame.numProp + 4] = n.y;
      gl_mesh_frame.vertProperties[v * gl_mesh_frame.numProp + 5] = n.z;
    }
    gl_mesh_frame.runFlags[run] &= ~uint8_t(2);
  }
  ASSERT_FALSE(gl_mesh_frame.HasNormals(0));

  Manifold m_modern(gl_modern);
  Manifold m_mesh_frame(gl_mesh_frame);

  Manifold sm_modern = m_modern.SmoothByNormals().Refine(4);
  Manifold sm_mesh_frame = m_mesh_frame.SmoothByNormals().Refine(4);
  EXPECT_NEAR(sm_modern.Volume(), sm_mesh_frame.Volume(), 1e-4);
  EXPECT_NEAR(sm_modern.SurfaceArea(), sm_mesh_frame.SurfaceArea(), 1e-4);
}

TEST(Manifold, MixedNormalsDefaultIDRoundTrip) {
  // Both operands use the default originalID 0, but only the sphere records
  // normals. A MeshGL round trip must keep that per run, in either operand
  // order, so the sphere's normals still rotate afterward.
  const Manifold sphere = Manifold::Sphere(5, {32}).CalculateNormals();
  const Manifold overlap = Manifold::Cube(vec3(6), true).Translate({4, 0, 0});
  const Manifold apart = Manifold::Cube(vec3(6), true).Translate({20, 0, 0});
  for (const Manifold& mixed :
       {sphere + overlap, overlap + sphere, sphere + apart, apart + sphere}) {
    const MeshGL gl = mixed.GetMeshGL();
    const MeshGL round = Manifold(gl).GetMeshGL();
    ASSERT_EQ(round.NumRun(), gl.NumRun());
    for (size_t run = 0; run < gl.NumRun(); ++run) {
      EXPECT_EQ(round.runIndex[run + 1], gl.runIndex[run + 1]);
      EXPECT_EQ(round.HasNormals(run), gl.HasNormals(run));
    }
    // The overlapping cube has verts at the sphere's radius, so compare with
    // the result that skipped the round trip instead of expecting no misses.
    auto outward = [](vec3 pos) { return la::normalize(pos); };
    const auto expected = CountSphereNormalAlignment(
        mixed.Rotate(90, 0, 0).GetMeshGL(), 5.0, 3, outward);
    const auto actual = CountSphereNormalAlignment(
        Manifold(gl).Rotate(90, 0, 0).GetMeshGL(), 5.0, 3, outward);
    EXPECT_GT(expected.first, 0);
    EXPECT_EQ(actual, expected);
  }
}

TEST(Manifold, UntrackedRunsGroupedByNormals) {
  // Untracked runs that agree on normals share one relation, so only a
  // disagreeing run is split off.
  MeshGL gl = Manifold::Cube().CalculateNormals().GetMeshGL();
  ASSERT_EQ(gl.NumTri(), 12u);
  gl.runOriginalID = {0, 0, 0};
  gl.runIndex = {0, 12, 24, 36};
  gl.runFlags = {2, 0, 2};
  gl.runTransform.clear();
  const MeshGL out = Manifold(gl).GetMeshGL();
  ASSERT_EQ(out.NumRun(), 2u);
  for (size_t run = 0; run < out.NumRun(); ++run) {
    const uint32_t numTri = (out.runIndex[run + 1] - out.runIndex[run]) / 3;
    EXPECT_EQ(numTri, out.HasNormals(run) ? 8u : 4u);
  }
}

TEST(Manifold, TransformMixedNormalsPerMeshID) {
  // Mixed Boolean output (no-normals + with-normals): the with-normals
  // meshIDs still hold world-frame normals at slot 0..2 that must rotate
  // with a subsequent Transform. Impl::Transform must per-meshID iterate,
  // not skip on the whole-impl flag.
  MeshGL gl = MixedNormalsCubePlusSphere().Rotate(90, 0, 0).GetMeshGL();
  auto [good, bad] = CountSphereNormalAlignment(
      gl, 5.0, 3, [](vec3 pos) { return la::normalize(pos); });
  EXPECT_GT(good, 0);
  EXPECT_EQ(bad, 0);
}

TEST(Manifold, AsOriginalMixedNormalsKept) {
  // AsOriginal of a mixed result keeps two runs under one ID, one with the
  // sphere's normals and one for the cube without. The sphere's normals still
  // rotate, triID still matches the output order, and a MeshGL round trip
  // keeps both runs. The overlap interleaves the two kinds of triangles, and
  // the IDs cover untracked, reserved, above the reserved range, and equal to
  // the next reserved ID.
  const Manifold sphere = Manifold::Sphere(5, {32}).CalculateNormals();
  const Manifold cube = Manifold::Cube(vec3(6), true).Translate({4, 0, 0});
  for (const Manifold& mixed : {sphere + cube, cube + sphere}) {
    for (const int id : {-1, 0, 1 << 29, -2}) {
      const int useID =
          id == -2 ? static_cast<int>(Manifold::ReserveIDs(1)) + 1 : id;
      const Manifold original = mixed.AsOriginal(useID);
      const MeshGL gl = original.GetMeshGL();
      ASSERT_EQ(gl.NumRun(), 2u) << id;
      EXPECT_EQ(gl.runOriginalID[0], gl.runOriginalID[1]) << id;
      EXPECT_NE(gl.HasNormals(0), gl.HasNormals(1)) << id;
      const MeshGL round = Manifold(gl).GetMeshGL();
      ASSERT_EQ(round.NumRun(), 2u) << id;
      EXPECT_NE(round.HasNormals(0), round.HasNormals(1)) << id;
      auto outward = [](vec3 pos) { return la::normalize(pos); };
      const auto expected = CountSphereNormalAlignment(
          mixed.Rotate(90, 0, 0).GetMeshGL(), 5.0, 3, outward);
      const auto actual = CountSphereNormalAlignment(
          original.Rotate(90, 0, 0).GetMeshGL(), 5.0, 3, outward);
      EXPECT_GT(actual.first, 0) << id;
      EXPECT_EQ(actual, expected) << id;
      const MeshGL tri = original.TriID2FaceID().GetMeshGL();
      int mismatch = 0;
      for (size_t t = 0; t < tri.NumTri(); ++t) mismatch += tri.faceID[t] != t;
      EXPECT_EQ(mismatch, 0) << id;
    }
  }
}

TEST(Manifold, MinkowskiKeepsNormals) {
  // Minkowski unions the input with hulls that have no normals and calls
  // AsOriginal(0), so only the input's surviving triangles keep their normals.
  const Manifold nonConvex =
      (Manifold::Cube(vec3(2), true) - Manifold::Sphere(0.75, {8}))
          .CalculateNormals();
  const Manifold ball = Manifold::Sphere(0.1, {8});
  const MeshGL sum = nonConvex.MinkowskiSum(ball).GetMeshGL();
  ASSERT_EQ(sum.NumRun(), 2u);
  EXPECT_TRUE(sum.HasNormals(0));
  EXPECT_FALSE(sum.HasNormals(1));
  // Erosion keeps none of the input's triangles, so nothing records normals.
  const MeshGL difference = nonConvex.MinkowskiDifference(ball).GetMeshGL();
  ASSERT_EQ(difference.NumRun(), 1u);
  EXPECT_FALSE(difference.HasNormals(0));
  for (const MeshGL* gl : {&sum, &difference})
    for (float x : gl->vertProperties) EXPECT_TRUE(std::isfinite(x));
}

TEST(Manifold, ComposeMixedNormalsPerMeshID) {
  // Compose's per-node eager-transform check is per-Relation, not whole-node.
  // Pass a mixed Manifold with a pending Rotate into a 3-input BatchBoolean,
  // forcing the disjoint Compose path where node->transform_ is non-identity
  // and the node mixes meshIDs with and without normals. Sphere's normals
  // within the mixed input must still rotate.
  const Manifold mixed_rot = MixedNormalsCubePlusSphere().Rotate(90, 0, 0);
  const Manifold t1 = Manifold::Tetrahedron().Translate({-50, 0, 0});
  const Manifold t2 = Manifold::Tetrahedron().Translate({-100, 0, 0});
  MeshGL gl =
      Manifold::BatchBoolean({mixed_rot, t1, t2}, OpType::Add).GetMeshGL();
  auto [good, bad] = CountSphereNormalAlignment(
      gl, 5.0, 3, [](vec3 pos) { return la::normalize(pos); });
  EXPECT_GT(good, 0);
  EXPECT_EQ(bad, 0);
}

TEST(Manifold, BooleanSubtractMixedQPerMeshIDNegation) {
  // CreateProperties' cavity sign-flip must be per-source-meshID. Subtract a
  // mixed Q (some meshIDs with hasNormals, some without). Cavity verts from
  // the hasNormals meshIDs need their slot 0..2 flipped to point outward
  // from the result solid (into the cavity = toward sphere center).
  const Manifold A = Manifold::Cube({100, 100, 100}, true);
  MeshGL gl = (A - MixedNormalsCubePlusSphere()).GetMeshGL();
  auto [good, bad] = CountSphereNormalAlignment(
      gl, 5.0, 3, [](vec3 pos) { return la::normalize(-pos); });
  EXPECT_GT(good, 0);
  EXPECT_EQ(bad, 0);
}

TEST(Manifold, ErrorPropagationSmoothByNormals) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.SmoothByNormals().Status(),
            Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationRefine) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.Refine(2).Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.RefineToLength(0.1).Status(),
            Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.RefineToTolerance(0.1).Status(),
            Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationAsOriginal) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.AsOriginal().Status(), Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationWarp) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.Warp([](vec3& v) {}).Status(),
            Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.WarpBatch([](VecView<vec3>) {}).Status(),
            Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationMinkowski) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  Manifold good = Manifold::Cube();
  EXPECT_EQ(errored.MinkowskiSum(good).Status(),
            Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(good.MinkowskiSum(errored).Status(),
            Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.MinkowskiDifference(good).Status(),
            Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(good.MinkowskiDifference(errored).Status(),
            Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationSplitByPlane) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  auto parts = errored.SplitByPlane(vec3(0, 0, 1), 0);
  EXPECT_EQ(parts.first.Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(parts.second.Status(), Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationTrimByPlane) {
  MeshGL in = TetGL();
  std::swap(in.triVerts[1], in.triVerts[2]);
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NotManifold);
  for (vec3 normal : {vec3(0, 0, 1), vec3(0.0)}) {
    EXPECT_EQ(errored.TrimByPlane(normal, 0).Status(),
              Manifold::Error::NotManifold);
  }
}

TEST(Manifold, ErrorPropagationMirror) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.Mirror(vec3(1, 0, 0)).Status(),
            Manifold::Error::NonFiniteVertex);
  // Degenerate normal (zero vector) on errored input should still propagate.
  EXPECT_EQ(errored.Mirror(vec3(0, 0, 0)).Status(),
            Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, ErrorPropagationSimplify) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold errored(in);
  ASSERT_EQ(errored.Status(), Manifold::Error::NonFiniteVertex);
  EXPECT_EQ(errored.Simplify().Status(), Manifold::Error::NonFiniteVertex);
}

#ifndef MANIFOLD_NO_IOSTREAM
TEST(Manifold, ObjRoundTrip) {
  Manifold m = Manifold::Cube();
  std::stringstream ss;
  m.WriteOBJ(ss);
  ss.seekg(0);
  Manifold m2 = Manifold::ReadOBJ(ss);
  EXPECT_EQ(m2.Status(), Manifold::Error::NoError);
  EXPECT_EQ(m2.Volume(), 1);
  EXPECT_EQ(m2.GetPrecision(), kMinRelativePrecision);
}

// A unit tetrahedron, missing its last face line.
constexpr char kTetOBJ[] =
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\n";

TEST(Manifold, ObjNoTrailingNewline) {
  // The final line used to pick up the EOF value as an extra character. Put a
  // vertex last, since the face pattern happened to tolerate it.
  std::istringstream ss(
      "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1");
  const Manifold m = Manifold::ReadOBJ(ss);
  EXPECT_EQ(m.Status(), Manifold::Error::NoError);
  EXPECT_EQ(m.NumVert(), 4);
}

TEST(Manifold, ObjHugeFaceIndex) {
  // Indices too large for int, or for any integer type, must not throw.
  for (const char* index :
       {"2147483648", "4294967298", "99999999999999999999999"}) {
    std::istringstream ss(std::string(kTetOBJ) + "f 2 3 " + index + "\n");
    const Manifold m = Manifold::ReadOBJ(ss);
    EXPECT_EQ(m.Status(), Manifold::Error::VertexOutOfBounds) << index;
  }
}
#endif

TEST(Manifold, OppositeFace) {
  MeshGL gl;
  gl.vertProperties = {
      0, 0, 0,  //
      1, 0, 0,  //
      0, 1, 0,  //
      1, 1, 0,  //
      0, 0, 1,  //
      1, 0, 1,  //
      0, 1, 1,  //
      1, 1, 1,  //
      2, 0, 0,  //
      2, 1, 0,  //
      2, 0, 1,  //
      2, 1, 1,  //
  };
  gl.triVerts = {
      0, 1,  4,   //
      0, 2,  3,   //
      0, 3,  1,   //
      0, 4,  2,   //
      1, 3,  5,   //
      1, 3,  9,   //
      1, 5,  3,   //
      1, 5,  4,   //
      1, 8,  5,   //
      1, 9,  8,   //
      2, 4,  6,   //
      2, 6,  7,   //
      2, 7,  3,   //
      3, 5,  7,   //
      3, 7,  5,   //
      3, 7,  11,  //
      3, 11, 9,   //
      4, 5,  6,   //
      5, 7,  6,   //
      5, 8,  10,  //
      5, 10, 7,   //
      7, 10, 11,  //
      8, 9,  10,  //
      9, 11, 10,  //
  };
  Manifold man(gl);
  EXPECT_EQ(man.Status(), Manifold::Error::NoError);
  EXPECT_EQ(man.NumVert(), 12);
  EXPECT_FLOAT_EQ(man.Volume(), 2);
}

/**
 * ExpectMeshes performs a decomposition, so this test ensures that a disjoint
 * BatchBoolean Add and Decompose are inverse operations.
 */
TEST(Manifold, Decompose) {
  std::vector<Manifold> manifoldList;
  manifoldList.emplace_back(Manifold::Tetrahedron().AsOriginal());
  manifoldList.emplace_back(Manifold::Cube().Translate({2, 0, 0}).AsOriginal());
  manifoldList.emplace_back(
      Manifold::Sphere(1, {4}).Translate({4, 0, 0}).AsOriginal());
  Manifold manifolds = Manifold::BatchBoolean(manifoldList, OpType::Add);

  ExpectMeshes(manifolds, {{8, 12}, {6, 8}, {4, 4}});

  std::vector<MeshGL> input;

  for (const Manifold& manifold : manifoldList) {
    EXPECT_GE(manifold.OriginalID(), 0);
    input.emplace_back(manifold.GetMeshGL());
  }

  RelatedGL(manifolds, input);
}

TEST(Manifold, DecomposeGroupsEnclosedNegativeVolume) {
  const Manifold outer = Manifold::Cube({10, 10, 10}, true);
  const Manifold largeVug =
      Manifold::Cube({4, 4, 4}, true).Translate({-2, 0, 0});
  const Manifold siblingVug =
      Manifold::Cube({2, 2, 2}, true).Translate({2, 0, 0});
  const Manifold island = Manifold::Cube({2, 2, 2}, true).Translate({-2, 0, 0});
  const Manifold nestedVug =
      Manifold::Cube({1, 1, 1}, true).Translate({-2, 0, 0});
  const Manifold shape = outer - largeVug - siblingVug + island - nestedVug;

  const auto components = shape.Decompose();

  ASSERT_EQ(components.size(), 2);
  std::vector<double> componentVolumes;
  size_t componentTriangles = 0;
  for (const Manifold& component : components) {
    componentVolumes.push_back(component.Volume());
    componentTriangles += component.NumTri();
  }
  std::sort(componentVolumes.begin(), componentVolumes.end());
  EXPECT_NEAR(componentVolumes[0], 7, 1e-8);
  EXPECT_NEAR(componentVolumes[1], 928, 1e-8);
  EXPECT_NEAR(componentVolumes[0] + componentVolumes[1], shape.Volume(), 1e-8);
  EXPECT_EQ(componentTriangles, shape.NumTri());
}

TEST(Manifold, DecomposeDeeplyNestedVugsAndSeparateSolid) {
  const Manifold outer = Manifold::Cube({20, 20, 20}, true);
  const Manifold cavity = Manifold::Cube({16, 16, 16}, true);
  const Manifold island = Manifold::Cube({12, 12, 12}, true);
  const Manifold nestedCavity = Manifold::Cube({8, 8, 8}, true);
  const Manifold nestedIsland = Manifold::Cube({4, 4, 4}, true);
  const Manifold innerCavity = Manifold::Cube({2, 2, 2}, true);
  const Manifold separate =
      Manifold::Cube({3, 3, 3}, true).Translate({30, 0, 0});
  const Manifold shape = outer - cavity + island - nestedCavity + nestedIsland -
                         innerCavity + separate;

  const auto components = shape.Decompose();
  ASSERT_EQ(components.size(), 4);
  std::vector<double> volumes;
  size_t triangles = 0;
  for (const Manifold& component : components) {
    EXPECT_EQ(component.Status(), Manifold::Error::NoError);
    volumes.push_back(component.Volume());
    triangles += component.NumTri();
  }
  std::sort(volumes.begin(), volumes.end());
  EXPECT_NEAR(volumes[0], 27, 1e-8);
  EXPECT_NEAR(volumes[1], 56, 1e-8);
  EXPECT_NEAR(volumes[2], 1216, 1e-8);
  EXPECT_NEAR(volumes[3], 3904, 1e-8);
  EXPECT_EQ(triangles, shape.NumTri());
}

TEST(Manifold, DecomposeProps) {
  std::vector<MeshGL> input;
  std::vector<Manifold> manifoldList;
  auto tet = WithPositionColors(Manifold::Tetrahedron().AsOriginal());
  manifoldList.emplace_back(tet);
  input.emplace_back(tet.GetMeshGL());
  auto cube =
      WithPositionColors(Manifold::Cube().Translate({2, 0, 0}).AsOriginal());
  manifoldList.emplace_back(cube);
  input.emplace_back(cube.GetMeshGL());
  auto sphere = WithPositionColors(
      Manifold::Sphere(1, {4}).Translate({4, 0, 0}).AsOriginal());
  manifoldList.emplace_back(sphere);
  input.emplace_back(sphere.GetMeshGL());
  Manifold manifolds = Manifold::BatchBoolean(manifoldList, OpType::Add);

  ExpectMeshes(manifolds, {{8, 12, 3}, {6, 8, 3}, {4, 4, 3}});

  RelatedGL(manifolds, input);

  for (const Manifold& manifold : manifolds.Decompose()) {
    RelatedGL(manifold, input);
  }
}

TEST(Manifold, DecomposeManyUnequalComponents) {
  std::vector<Manifold> originals;
  std::vector<MeshGL> input;
  for (int i = 0; i < 24; ++i) {
    const Manifold shape = i % 3 == 0   ? Manifold::Cube()
                           : i % 3 == 1 ? Manifold::Tetrahedron()
                                        : Manifold::Sphere(1, {12});
    originals.push_back(
        WithPositionColors(shape.Translate({5.0 * i, 0, 0}).AsOriginal()));
    input.push_back(originals.back().GetMeshGL());
  }
  auto parts = Manifold::BatchBoolean(originals, OpType::Add).Decompose();
  ASSERT_EQ(parts.size(), originals.size());
  std::sort(parts.begin(), parts.end(),
            [](const Manifold& a, const Manifold& b) {
              return a.BoundingBox().min.x < b.BoundingBox().min.x;
            });
  for (size_t i = 0; i < parts.size(); ++i) {
    SCOPED_TRACE(i);
    ASSERT_EQ(parts[i].Status(), Manifold::Error::NoError);
    EXPECT_EQ(parts[i].NumVert(), originals[i].NumVert());
    EXPECT_EQ(parts[i].NumTri(), originals[i].NumTri());
    EXPECT_EQ(parts[i].NumProp(), originals[i].NumProp());
    EXPECT_NEAR(parts[i].Volume(), originals[i].Volume(), 1e-10);
    RelatedGL(parts[i], input);
  }
}

/**
 * These tests check the various manifold constructors.
 */
TEST(Manifold, Sphere) {
  int n = 25;
  Manifold sphere = Manifold::Sphere(1.0, {4 * n});
  EXPECT_EQ(sphere.NumTri(), n * n * 8);
}

// Native-only: the WASM build is single-threaded (no pthread), so std::thread
// construction throws, which aborts under -fno-exceptions.
#ifndef __EMSCRIPTEN__
TEST(Manifold, CircularQualityPerThread) {
  // Each thread builds with its own quality, so neither can see the other's.
  const CircularQuality coarse{0, 90.0};
  const CircularQuality fine{0, 5.0, 0.01};
  const size_t coarseTri = Manifold::Sphere(10, coarse).NumTri();
  const size_t fineTri = Manifold::Sphere(10, fine).NumTri();
  ASSERT_NE(coarseTri, fineTri);
  std::atomic<int> wrong{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&, i] {
      const CircularQuality& q = i % 2 ? fine : coarse;
      const size_t expected = i % 2 ? fineTri : coarseTri;
      for (int j = 0; j < 20; ++j) {
        if (Manifold::Sphere(10, q).NumTri() != expected) ++wrong;
      }
    });
  }
  for (auto& t : threads) t.join();
  EXPECT_EQ(wrong.load(), 0);
}
#endif

TEST(Manifold, CircularQualitySegmentsFor) {
  const CircularQuality defaults;
  EXPECT_EQ(defaults.segments, 0);
  EXPECT_DOUBLE_EQ(defaults.minAngle, 10.0);
  EXPECT_DOUBLE_EQ(defaults.minEdgeLength, 1.0);
  EXPECT_EQ(defaults.SegmentsFor(1.0), 8);
  EXPECT_EQ(defaults.SegmentsFor(100.0), 36);

  const CircularQuality q{0, 22.5, 0.125};
  EXPECT_EQ(q.SegmentsFor(2.0), 16);
  EXPECT_EQ((CircularQuality{12, 22.5, 0.125}.SegmentsFor(2.0)), 12);
  EXPECT_NE((CircularQuality{0, 5, 0.25}.SegmentsFor(2.0)), 16);
  // segments below 3 apply no constraint.
  for (const int segments : {-1, 1, 2}) {
    EXPECT_EQ((CircularQuality{segments, 22.5, 0.125}.SegmentsFor(2.0)), 16);
  }

  // A non-positive or NaN angle or edge length uses the default.
  for (const double invalid : {0.0, -1.0, static_cast<double>(NAN)}) {
    for (const double r : {1.0, 2.0, 100.0}) {
      EXPECT_EQ((CircularQuality{0, invalid, 0.125}.SegmentsFor(r)),
                (CircularQuality{0, 10.0, 0.125}.SegmentsFor(r)));
      EXPECT_EQ((CircularQuality{0, 22.5, invalid}.SegmentsFor(r)),
                (CircularQuality{0, 22.5, 1.0}.SegmentsFor(r)));
    }
  }
}

TEST(Manifold, CircularQualityFactories) {
  const CircularQuality q{0, 30.0, 0.1};
  const int n = q.SegmentsFor(5);
  EXPECT_EQ(n, 12);
  EXPECT_EQ(CrossSection::Circle(5, q).NumVert(), n);
  EXPECT_EQ(Manifold::Cylinder(2, 5, -1, q).NumVert(), 2 * n);
  EXPECT_EQ(Manifold::Sphere(5, q).NumTri(), Manifold::Sphere(5, {n}).NumTri());
  CrossSection ring = CrossSection::Square({1, 1}).Translate({4.5, 0});
  EXPECT_EQ(Manifold::Revolve(ring.ToPolygons(), q).NumVert(),
            Manifold::Revolve(ring.ToPolygons(), {n}).NumVert());
  // A round offset of a point-like square adds n vertices per 360 degrees.
  CrossSection tiny = CrossSection::Square({1e-3, 1e-3}, true);
  EXPECT_EQ(tiny.Offset(5, CrossSection::JoinType::Round, 2, q).NumVert(),
            tiny.Offset(5, CrossSection::JoinType::Round, 2, {n}).NumVert());

  // Sphere takes any positive count, rounding it up to a multiple of four.
  for (const int segments : {1, 2, 4}) {
    EXPECT_EQ(Manifold::Sphere(5, {segments}).NumTri(), 8);
    EXPECT_EQ(Manifold::Sphere(5, CircularQuality{segments, 30.0}).NumTri(), 8);
  }
  EXPECT_EQ(Manifold::Sphere(5, {5}).NumTri(), 32);

  // {} matches the defaults.
  EXPECT_EQ(Manifold::Sphere(5, {}).NumTri(), Manifold::Sphere(5).NumTri());
  EXPECT_EQ(CrossSection::Circle(5, {}).NumVert(),
            CircularQuality{}.SegmentsFor(5));
#if __cplusplus >= 202002L
  EXPECT_EQ(
      CrossSection::Circle(5, {.minAngle = 30, .minEdgeLength = 0.1}).NumVert(),
      n);
#endif
}

TEST(Manifold, RevolveDivisionsForAngle) {
  // A partial revolve of a ring off the axis has 4 vertices per slice, and a
  // full one shares its first and last slice.
  const Polygons ring =
      CrossSection::Square({1, 1}).Translate({4.5, 0}).ToPolygons();
  auto numDivisions = [&](CircularQuality q, double degrees) {
    const size_t numVert = Manifold::Revolve(ring, q, degrees).NumVert();
    return static_cast<int>(numVert / 4) - (std::abs(degrees) < 360 ? 1 : 0);
  };
  // A partial revolve rounds the matching fraction of a full circle up.
  EXPECT_EQ(numDivisions({30}, 90), 8);
  EXPECT_EQ(numDivisions({32}, 90), 8);
  EXPECT_EQ(numDivisions({30}, 100), 9);
  EXPECT_EQ(numDivisions({30}, 360), 30);
  EXPECT_EQ(numDivisions({30}, 1), 1);
  // The product underflows to zero for a subnormal angle.
  EXPECT_EQ(numDivisions({3}, 5e-324), 1);
  // Both sides of the boundaries where three segments need another division.
  EXPECT_EQ(numDivisions({3}, 120), 1);
  EXPECT_EQ(numDivisions({3}, 120.001), 2);
  EXPECT_EQ(numDivisions({3}, 240), 2);
  EXPECT_EQ(numDivisions({3}, 240.001), 3);
  // An excess of one ulp is treated as floating-point error, two is not. Two
  // and three ulps above 120 degrees give 30 segments one and two ulps above
  // 10.
  const double up1 = std::nextafter(120.0, 360.0);
  const double up2 = std::nextafter(up1, 360.0);
  const double up3 = std::nextafter(up2, 360.0);
  EXPECT_EQ(30 * up2 / 360, std::nextafter(10.0, 11.0));
  EXPECT_EQ(numDivisions({30}, up2), 10);
  EXPECT_EQ(30 * up3 / 360, std::nextafter(std::nextafter(10.0, 11.0), 11.0));
  EXPECT_EQ(numDivisions({30}, up3), 11);
  EXPECT_EQ(numDivisions({30}, 120.00000000001), 11);

  // An explicit count and a count derived from the angle and edge length
  // round the same way. The ring's outer radius 5.5 gets 36 by default.
  const CircularQuality derived{0, 12.5, 0.1};
  ASSERT_EQ(derived.SegmentsFor(5.5), 28);
  ASSERT_EQ(CircularQuality{}.SegmentsFor(5.5), 36);
  for (const double degrees : {10.0, 90.0, 100.0, 250.0, 359.0, 360.0}) {
    const int expected = static_cast<int>(std::ceil(28 * degrees / 360));
    EXPECT_EQ(numDivisions({28}, degrees), expected) << degrees;
    EXPECT_EQ(numDivisions(derived, degrees), expected) << degrees;
    EXPECT_EQ(numDivisions({}, degrees),
              static_cast<int>(std::ceil(36 * degrees / 360)))
        << degrees;
  }

  // A count that is whole in exact arithmetic is not rounded up by
  // floating-point error, here from k * 3.6 not being exact in binary (k = 19
  // gives 19.000000000000004 without the tolerance).
  for (int k = 1; k <= 100; ++k) {
    ASSERT_EQ(numDivisions({100}, k * 3.6), k) << k;
  }
  for (int k = 1; k <= 19; ++k) {
    ASSERT_EQ(numDivisions({19 * 7}, k * (360.0 / 19)), 7 * k) << k;
  }

  for (const double degrees :
       {1.0, 45.0, 90.0, 120.0, 179.0, 240.0, 241.0, 359.0, 359.999999, 360.0,
        0.7, 1e-4, 360.0 / 19, 7.5, 100.0 / 3, 0.1 * 3, -90.0, 720.0}) {
    for (int n = 1; n <= 40; ++n) {
      const double d = std::min(std::abs(degrees), 360.0);
      // The documented limits: more than 120 degrees needs 2, more than 240
      // needs 3.
      const int fewest = d > 240 ? 3 : d > 120 ? 2 : 1;
      EXPECT_EQ(
          numDivisions(CircularQuality::ForDivisions(n, degrees), degrees),
          std::max(n, fewest))
          << degrees << " degrees, " << n << " divisions";
    }
  }

  // Randomized angles and counts.
  std::mt19937 gen(12345);
  std::uniform_real_distribution<double> angle(0.01, 360);
  std::uniform_int_distribution<int> count(1, 300);
  for (int i = 0; i < 1000; ++i) {
    const double degrees = angle(gen);
    const int n = count(gen);
    const int fewest = degrees > 240 ? 3 : degrees > 120 ? 2 : 1;
    EXPECT_EQ(numDivisions(CircularQuality::ForDivisions(n, degrees), degrees),
              std::max(n, fewest))
        << degrees << " degrees, " << n << " divisions";
  }
  // A tiny angle gets no more divisions than the largest count gives.
  const int maxInt = std::numeric_limits<int>::max();
  EXPECT_EQ(CircularQuality::ForDivisions(40, 1e-6).segments, maxInt);
  EXPECT_EQ(numDivisions({maxInt}, 1e-6), 6);
  EXPECT_EQ(CircularQuality::ForDivisions(5, 0).segments, 0);
  EXPECT_EQ(CircularQuality::ForDivisions(5, NAN).segments, 0);
}

TEST(Manifold, CircularSegmentsTinyAngle) {
  // 360 / angle doesn't fit in an int here, so the edge length should decide.
  const CircularQuality tiny{0, 1e-9};
  EXPECT_EQ(tiny.SegmentsFor(1.0), 8);
  EXPECT_EQ(tiny.SegmentsFor(100.0), 628);
  EXPECT_GT(tiny.SegmentsFor(1e300), 1000000);
  EXPECT_EQ((CircularQuality{0, NAN, NAN}.SegmentsFor(1.0)), 8);
}

TEST(Manifold, AssignToMovedFrom) {
  Manifold a = Manifold::Cube();
  const Manifold b = std::move(a);
  a = b;
  EXPECT_EQ(a.Volume(), 1);

  Manifold c = Manifold::Cube();
  const Manifold d = std::move(c);
  Manifold e(c);
  e = d;
  EXPECT_EQ(e.Volume(), 1);

  Manifold f = Manifold::Sphere(1);
  f = c;
  f = d;
  EXPECT_EQ(f.Volume(), 1);
}

TEST(Manifold, Cylinder) {
  int n = 10000;
  Manifold cylinder = Manifold::Cylinder(2, 2, 2, {n});
  EXPECT_EQ(cylinder.NumTri(), 4 * n - 4);
}

TEST(Manifold, CylinderZeroRadiusLow) {
  // cylinder(h, 0, r) should produce a cone with apex at z=0 and base at z=h.
  // Use enough segments for accurate volume.
  const int n = 256;
  const double h = 5.0, r = 3.0;
  Manifold coneApexBottom = Manifold::Cylinder(h, 0.0, r, {n});
  Manifold coneApexTop = Manifold::Cylinder(h, r, 0.0, {n});

  EXPECT_EQ(coneApexBottom.Status(), Manifold::Error::NoError);
  EXPECT_FALSE(coneApexBottom.IsEmpty());

  // Both cones must have equal total volume.
  const double totalVol = coneApexTop.Volume();
  EXPECT_NEAR(coneApexBottom.Volume(), totalVol, 1e-6);

  // Differentiate orientation by intersecting with the bottom half (z in
  // [0, h/2]). A sub-cone scaled by 1/2 in all dimensions has 1/8 the volume,
  // so apex-at-bottom gives V/8 and apex-at-top gives 7*V/8.
  Manifold slicer = Manifold::Cube(vec3(2 * r + 1, 2 * r + 1, h / 2))
                        .Translate(vec3(-(r + 0.5), -(r + 0.5), 0.0));
  EXPECT_NEAR((coneApexBottom ^ slicer).Volume(), totalVol / 8.0, 0.01);
  EXPECT_NEAR((coneApexTop ^ slicer).Volume(), 7.0 * totalVol / 8.0, 0.01);
}

TEST(Manifold, CylinderCenter) {
  const int n = 256;
  const double h = 5.0, r = 3.0;
  const Manifold cylinder = Manifold::Cylinder(h, r, r, {n}, true);
  const Manifold coneApexTop = Manifold::Cylinder(h, r, 0.0, {n}, true);
  const Manifold coneApexBottom = Manifold::Cylinder(h, 0.0, r, {n}, true);

  for (const Manifold& m : {cylinder, coneApexTop, coneApexBottom}) {
    EXPECT_EQ(m.Status(), Manifold::Error::NoError);
    const Box box = m.BoundingBox();
    EXPECT_DOUBLE_EQ(box.min.z, -h / 2);
    EXPECT_DOUBLE_EQ(box.max.z, h / 2);
  }
  EXPECT_NEAR(cylinder.Volume(), Manifold::Cylinder(h, r, r, {n}).Volume(),
              1e-6);

  // Same orientation check as CylinderZeroRadiusLow, on the centered cones.
  const double totalVol = coneApexTop.Volume();
  EXPECT_NEAR(coneApexBottom.Volume(), totalVol, 1e-6);
  Manifold slicer = Manifold::Cube(vec3(2 * r + 1, 2 * r + 1, h / 2))
                        .Translate(vec3(-(r + 0.5), -(r + 0.5), -h / 2));
  EXPECT_NEAR((coneApexBottom ^ slicer).Volume(), totalVol / 8.0, 0.01);
  EXPECT_NEAR((coneApexTop ^ slicer).Volume(), 7.0 * totalVol / 8.0, 0.01);
}

TEST(Manifold, ConstructorsOriginalID) {
  // Every constructor returns an original with the shared ID 0, including
  // the centered cylinder and the apex-at-bottom cone.
  const Polygons square = {{{1, 0}, {2, 0}, {2, 1}, {1, 1}}};
  const std::vector<Manifold> shapes = {
      Manifold::Tetrahedron(),
      Manifold::Cube(vec3(1.0), true),
      Manifold::Sphere(1.0),
      Manifold::Cylinder(2, 1),
      Manifold::Cylinder(2, 1, 1, {}, true),
      Manifold::Cylinder(2, 0, 1),
      Manifold::Cylinder(2, 0, 1, {}, true),
      Manifold::Extrude(square, 1.0),
      Manifold::Revolve(square),
      Manifold::Hull({vec3(0.0), vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1)}),
      Manifold::Cube().MinkowskiSum(Manifold::Sphere(0.1, {8})),
  };
  for (const Manifold& m : shapes) {
    EXPECT_EQ(m.Status(), Manifold::Error::NoError);
    EXPECT_EQ(m.OriginalID(), 0);
    EXPECT_EQ(m.GetMeshGL().runOriginalID, std::vector<uint32_t>{0});
  }
}

TEST(Manifold, Extrude) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0, 3);
  EXPECT_EQ(donut.Genus(), 1);
  EXPECT_FLOAT_EQ(donut.Volume(), 12.0);
  EXPECT_FLOAT_EQ(donut.SurfaceArea(), 48.0);
}

TEST(Manifold, ExtrudeInvalid) {
  const Polygons square = {{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
  EXPECT_EQ(Manifold::Extrude(square, 1, 4, NAN).Status(),
            Manifold::Error::InvalidConstruction);
  EXPECT_EQ(Manifold::Extrude(square, 1, 4, 0, vec2(INFINITY)).Status(),
            Manifold::Error::InvalidConstruction);
  EXPECT_EQ(Manifold::Extrude({{{0, 0}, {1, 0}, {NAN, 1}}}, 1).Status(),
            Manifold::Error::InvalidConstruction);
}

TEST(Manifold, ShortContours) {
  const SimplePolygon square = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  EXPECT_DOUBLE_EQ(Manifold::Extrude({square, {{5, 5}}, {}}, 1).Volume(), 1);
  const SimplePolygon offset = {{1, 0}, {2, 0}, {2, 1}, {1, 1}};
  const Manifold revolved = Manifold::Revolve({offset}, {32});
  EXPECT_DOUBLE_EQ(
      Manifold::Revolve({offset, {{5, 5}}, {{1, 2}, {2, 2}}, {}}, {32})
          .Volume(),
      revolved.Volume());
}

TEST(Manifold, ExtrudeNegativeDivisions) {
  Manifold extrusion =
      Manifold::Extrude({{{0, 0}, {1, 0}, {1, 1}, {0, 1}}}, 1, -5);
  EXPECT_EQ(extrusion.Status(), Manifold::Error::InvalidConstruction);
  extrusion = Manifold::Extrude({{{0, 0}, {1, 0}, {1, 1}, {0, 1}}}, 1,
                                std::numeric_limits<int>::max());
  EXPECT_EQ(extrusion.Status(), Manifold::Error::ResultTooLarge);
}

TEST(Manifold, ExtrudeSquare) {
  const Polygons square = {{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
  const Manifold prism = Manifold::Extrude(square, 10.0);
  EXPECT_NEAR(prism.Volume(), 10.0, 0.001);
  EXPECT_NEAR(prism.SurfaceArea(), 42.0, 0.001);
}

TEST(Manifold, ExtrudeCone) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0, 0, 0, vec2(0.0));
  EXPECT_EQ(donut.Genus(), 0);
  EXPECT_FLOAT_EQ(donut.Volume(), 4.0);
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
  for (size_t i = 0; i < polys[0].size(); i++) {
    Polygons rotatedPolys = RotatePolygons(polys, i);
    vug = Manifold::Revolve(rotatedPolys, {48});
    EXPECT_EQ(vug.Genus(), -1);
    EXPECT_NEAR(vug.Volume(), 14.0 * kPi, 0.2);
    EXPECT_NEAR(vug.SurfaceArea(), 30.0 * kPi, 0.2);
  }
}

TEST(Manifold, Revolve2) {
  Polygons polys = SquareHole(2.0);
  Manifold donutHole = Manifold::Revolve(polys, {48});
  EXPECT_EQ(donutHole.Genus(), 0);
  EXPECT_NEAR(donutHole.Volume(), 48.0 * kPi, 1.0);
  EXPECT_NEAR(donutHole.SurfaceArea(), 96.0 * kPi, 1.0);
}

TEST(Manifold, Revolve3) {
  CrossSection circle = CrossSection::Circle(1, {32});
  Manifold sphere = Manifold::Revolve(circle.ToPolygons(), {32});
  EXPECT_NEAR(sphere.Volume(), 4.0 / 3.0 * kPi, 0.1);
  EXPECT_NEAR(sphere.SurfaceArea(), 4 * kPi, 0.15);
}

TEST(Manifold, RevolveClip) {
  Polygons polys = {{{-5, -10}, {5, 0}, {-5, 10}}};
  Polygons clipped = {{{0, -5}, {5, 0}, {0, 5}}};
  Manifold first = Manifold::Revolve(polys, {48});
  Manifold second = Manifold::Revolve(clipped, {48});
  EXPECT_EQ(first.Genus(), second.Genus());
  EXPECT_EQ(first.Volume(), second.Volume());
  EXPECT_EQ(first.SurfaceArea(), second.SurfaceArea());
}

TEST(Manifold, PartialRevolveOnYAxis) {
  Polygons polys = SquareHole(2.0);
  Polygons offsetPolys = SquareHole(10.0);

  Manifold revolute;
  for (size_t i = 0; i < polys[0].size(); i++) {
    Polygons rotatedPolys = RotatePolygons(polys, i);
    revolute = Manifold::Revolve(rotatedPolys, {48}, 180);
    EXPECT_EQ(revolute.Genus(), 1);
    EXPECT_NEAR(revolute.Volume(), 24.0 * kPi, 1.0);
    EXPECT_NEAR(revolute.SurfaceArea(),
                48.0 * kPi + 4.0 * 4.0 * 2.0 - 2.0 * 2.0 * 2.0, 1.0);
  }
}

TEST(Manifold, PartialRevolveOffset) {
  Polygons polys = SquareHole(10.0);

  Manifold revolute;
  for (size_t i = 0; i < polys[0].size(); i++) {
    Polygons rotatedPolys = RotatePolygons(polys, i);
    revolute = Manifold::Revolve(rotatedPolys, {48}, 180);
    EXPECT_EQ(revolute.Genus(), 1);
    EXPECT_NEAR(revolute.SurfaceArea(), 777.0, 1.0);
    EXPECT_NEAR(revolute.Volume(), 376.0, 1.0);
  }
}

TEST(Manifold, PartialRevolveSmallAngle) {
  // The default 12 segments for radius 2 cover 30 degrees each, so 10 degrees
  // is less than one segment.
  Polygons square = CrossSection::Square({1, 1}).Translate({1, 0}).ToPolygons();
  Manifold revolute = Manifold::Revolve(square, {}, 10);
  EXPECT_EQ(revolute.Genus(), 0);
  // Area 1 at mean radius 1.5, swept 10 degrees.
  EXPECT_NEAR(revolute.Volume(), 1.5 * kPi / 18, 0.01);
}

TEST(Manifold, RevolveInvalidAngle) {
  Polygons square = CrossSection::Square({1, 1}).Translate({1, 0}).ToPolygons();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double degrees :
       {0.0, std::numeric_limits<double>::quiet_NaN(), inf, -inf}) {
    for (const int segments : {0, 12}) {
      EXPECT_EQ(Manifold::Revolve(square, {segments}, degrees).Status(),
                Manifold::Error::InvalidConstruction)
          << "degrees " << degrees << ", segments " << segments;
    }
  }
}

TEST(Manifold, RevolveSliceCountOverflow) {
  const SimplePolygon profile = {{1, 0}, {2, 0}, {2, 1}, {1, 1}};
  const int maxInt = std::numeric_limits<int>::max();
  for (double degrees : {180.0, -180.0}) {
    EXPECT_EQ(Manifold::Revolve({profile}, {maxInt}, degrees).Status(),
              Manifold::Error::ResultTooLarge);
  }
  // A 15-gon with two triangular holes has 21 vertices and caps of 23
  // triangles each, so this many divisions over 180 degrees makes
  // 3 * (2 * 21 * 17043520 + 2 * 23) halfedges, just past INT_MAX.
  Polygons holed(1);
  for (int i = 0; i < 15; ++i) {
    holed[0].push_back(vec2(20, 0) + 10.0 * vec2(cosd(24 * i), sind(24 * i)));
  }
  holed.push_back({{14, -2}, {16, 2}, {18, -2}});
  holed.push_back({{22, -2}, {24, 2}, {26, -2}});
  EXPECT_EQ(Manifold::Revolve(holed, {6}, 180).NumTri(), 2 * 21 * 3 + 2 * 23);
  EXPECT_EQ(Manifold::Revolve(holed,
                              CircularQuality::ForDivisions(17043520, 180), 180)
                .Status(),
            Manifold::Error::ResultTooLarge);
  // Few enough vertices, but too many halfedges.
  const SimplePolygon triangle = {{0, 0}, {1, 1}, {0, 2}};
  EXPECT_EQ(Manifold::Revolve({triangle}, {maxInt - 2}).Status(),
            Manifold::Error::ResultTooLarge);
}

TEST(Manifold, RevolveAxisVerticesNotRepeated) {
  // Vertices on the axis appear once, so this stays well under the index
  // limit even though profile size times slices exceeds it.
  SimplePolygon profile;
  for (int i = 0; i < 10000; ++i) profile.push_back({0, i * 1e-4});
  profile.push_back({1, 0.5});
  const Manifold m = Manifold::Revolve({profile}, {220000});
  EXPECT_EQ(m.Status(), Manifold::Error::NoError);
  EXPECT_EQ(m.NumVert(), 10000 + 220000);
}

TEST(Manifold, RevolveNegativeAngle) {
  // One profile away from the axis and one touching it.
  for (const vec2 offset : {vec2(1, 0), vec2(0, 0)}) {
    Polygons square =
        CrossSection::Square({1, 1}).Translate(offset).ToPolygons();
    for (const int segments : {0, 12}) {
      for (const double degrees : {10.0, 90.0, 360.0, 400.0}) {
        Manifold positive = Manifold::Revolve(square, {segments}, degrees);
        Manifold negative = Manifold::Revolve(square, {segments}, -degrees);
        EXPECT_EQ(negative.Status(), Manifold::Error::NoError);
        EXPECT_GE(negative.OriginalID(), 0);
        EXPECT_EQ(negative.Genus(), positive.Genus());
        EXPECT_NEAR(negative.Volume(), positive.Volume(), 1e-9)
            << "degrees " << -degrees << ", segments " << segments;
        // The negative sweep goes the other way, mirroring the positive one
        // in y.
        const Box posBox = positive.BoundingBox();
        const Box negBox = negative.BoundingBox();
        EXPECT_NEAR(negBox.min.x, posBox.min.x, 1e-9);
        EXPECT_NEAR(negBox.max.x, posBox.max.x, 1e-9);
        EXPECT_NEAR(negBox.min.y, -posBox.max.y, 1e-9);
        EXPECT_NEAR(negBox.max.y, -posBox.min.y, 1e-9);
        EXPECT_NEAR(negBox.min.z, posBox.min.z, 1e-9);
        EXPECT_NEAR(negBox.max.z, posBox.max.z, 1e-9);
      }
    }
  }
}

TEST(Manifold, Warp) {
  CrossSection square = CrossSection::Square({1, 1});
  Manifold shape =
      Manifold::Extrude(square.ToPolygons(), 2, 10).Warp([](vec3& v) {
        v.x += v.z * v.z;
      });

  Manifold simplified = Manifold::BatchBoolean({shape}, OpType::Add);

  EXPECT_NEAR(shape.Volume(), simplified.Volume(), 0.0001);
  EXPECT_NEAR(shape.SurfaceArea(), simplified.SurfaceArea(), 0.0001);
  EXPECT_NEAR(shape.Volume(), 2, 0.0001);
}

TEST(Manifold, Warp2) {
  CrossSection circle =
      CrossSection::Circle(5, {20}).Translate(vec2(10.0, 10.0));

  Manifold shape =
      Manifold::Extrude(circle.ToPolygons(), 2, 10).Warp([](vec3& v) {
        int nSegments = 10;
        double angleStep = 2.0 / 3.0 * kPi / nSegments;
        int zIndex = nSegments - 1 - std::round(v.z);
        double angle = zIndex * angleStep;
        v.z = v.y;
        v.y = v.x * sin(angle);
        v.x = v.x * cos(angle);
      });

  Manifold simplified = Manifold::BatchBoolean({shape}, OpType::Add);

  EXPECT_NEAR(shape.Volume(), simplified.Volume(), 0.0001);
  EXPECT_NEAR(shape.SurfaceArea(), simplified.SurfaceArea(), 0.0001);
  EXPECT_NEAR(shape.Volume(), 321, 1);
}

TEST(Manifold, WarpBatch) {
  Manifold cube = Manifold::Cube({2, 3, 4});
  const int id = cube.OriginalID();

  Manifold shape1 = cube.Warp([](vec3& v) { v.x += v.z * v.z; });
  Manifold shape2 = cube.WarpBatch([](VecView<vec3> vecs) {
    for (vec3& v : vecs) {
      v.x += v.z * v.z;
    }
  });

  EXPECT_GE(id, 0);
  EXPECT_EQ(shape1.OriginalID(), -1);
  EXPECT_EQ(shape2.OriginalID(), -1);
  std::vector<uint32_t> runOriginalID1 = shape1.GetMeshGL().runOriginalID;
  EXPECT_EQ(runOriginalID1.size(), 1);
  EXPECT_EQ(runOriginalID1[0], id);
  std::vector<uint32_t> runOriginalID2 = shape2.GetMeshGL().runOriginalID;
  EXPECT_EQ(runOriginalID2.size(), 1);
  EXPECT_EQ(runOriginalID2[0], id);
  EXPECT_EQ(shape1.Volume(), shape2.Volume());
  EXPECT_EQ(shape1.SurfaceArea(), shape2.SurfaceArea());
}

TEST(Manifold, Project) {
  MeshGL input;
  input.numProp = 3;
  input.vertProperties = {0,    0,       0,     //
                          -2,   -0.7,    -0.1,  //
                          -2,   -0.7,    0,     //
                          -1.9, -0.7,    -0.1,  //
                          -1.9, -0.6901, -0.1,  //
                          -1.9, -0.7,    0,     //
                          -1.9, -0.6901, 0,     //
                          -2,   -1,      3,     //
                          -1.9, -1,      3,     //
                          -2,   -1,      4,     //
                          -1.9, -1,      4,     //
                          -1.9, -0.6901, 3,     //
                          -1.9, -0.6901, 4,     //
                          -1.7, -0.6901, 3,     //
                          -1.7, -0.6901, 3.2,   //
                          -2,   0,       -0.1,  //
                          -2,   0,       0,     //
                          -2,   0,       3,     //
                          -2,   0,       4,     //
                          -1.7, 0,       3,     //
                          -1.7, 0,       3.2,   //
                          -1,   -0.6901, -0.1,  //
                          -1,   -0.6901, 0,     //
                          -1,   -0.6901, 3.2,   //
                          -1,   -0.6901, 4,     //
                          -1,   0,       -0.1,  //
                          -1,   0,       0,     //
                          -1,   0,       3.2,   //
                          -1,   0,       4};
  input.triVerts = {1,  3,  2,   //
                    1,  4,  3,   //
                    2,  3,  5,   //
                    5,  6,  2,   //
                    3,  4,  6,   //
                    5,  3,  6,   //
                    6,  4,  21,  //
                    26, 22, 25,  //
                    21, 25, 22,  //
                    25, 15, 26,  //
                    26, 6,  22,  //
                    21, 4,  25,  //
                    21, 22, 6,   //
                    16, 26, 15,  //
                    16, 6,  26,  //
                    4,  15, 25,  //
                    15, 1,  16,  //
                    16, 2,  6,   //
                    4,  1,  15,  //
                    1,  2,  16,  //
                    12, 14, 23,  //
                    12, 13, 14,  //
                    12, 11, 13,  //
                    18, 9,  12,  //
                    11, 7,  17,  //
                    7,  9,  18,  //
                    17, 7,  18,  //
                    13, 11, 19,  //
                    17, 18, 20,  //
                    19, 11, 17,  //
                    19, 17, 20,  //
                    14, 13, 20,  //
                    18, 12, 24,  //
                    20, 13, 19,  //
                    20, 18, 27,  //
                    12, 10, 11,  //
                    24, 12, 23,  //
                    9,  10, 12,  //
                    9,  8,  10,  //
                    8,  11, 10,  //
                    8,  7,  11,  //
                    8,  9,  7,   //
                    14, 20, 27,  //
                    24, 28, 18,  //
                    27, 18, 28,  //
                    23, 14, 27,  //
                    24, 23, 28,  //
                    28, 23, 27};
  Manifold in(input);
  CrossSection projected = in.Project();
  EXPECT_NEAR(projected.Area(), 0.72, 0.01);
}

/**
 * Testing more advanced Manifold operations.
 */

TEST(Manifold, ProjectRefinedSphereContours) {
  // Cross the silhouette buffer's growth boundaries with regular polygons of
  // known area. Mirroring must preserve the winding of the projected solid.
  for (int segments : {128, 132, 256, 260}) {
    SCOPED_TRACE(segments);
    const Manifold sphere = Manifold::Sphere(1, {segments});
    const double area = 0.5 * segments * sind(360.0 / segments);
    for (const vec3 scale : {vec3(1), vec3(-2, 3, -1)}) {
      const Polygons projected = sphere.Scale(scale).Project();
      ASSERT_EQ(projected.size(), 1);
      ASSERT_EQ(projected.front().size(), segments);
      for (const vec2 point : projected.front()) {
        EXPECT_NEAR(point.x * point.x / (scale.x * scale.x) +
                        point.y * point.y / (scale.y * scale.y),
                    1, 1e-12);
      }
      EXPECT_NEAR(CrossSection(projected).Area(),
                  area * std::abs(scale.x * scale.y), 1e-10);
    }
  }
}

TEST(Manifold, ProjectGrowingContoursAndHoles) {
  EXPECT_TRUE(Manifold().Project().empty());
  const CrossSection section =
      CrossSection::Circle(3, {132}) - CrossSection::Circle(1, {132}) +
      CrossSection::Circle(0.5, {260}).Translate({10, 0});
  const Manifold prism =
      Manifold::Extrude(section.ToPolygons(), 2, 4).Refine(2);
  ASSERT_EQ(prism.Status(), Manifold::Error::NoError);
  for (const vec3 scale : {vec3(1), vec3(-2, 3, -1)}) {
    const CrossSection projected(prism.Scale(scale).Project());
    const CrossSection expected = section.Scale({scale.x, scale.y});
    EXPECT_EQ(projected.NumContour(), 3);
    EXPECT_NEAR(projected.Area(), expected.Area(), 1e-9);
    EXPECT_LT((projected - expected).Area() + (expected - projected).Area(),
              1e-9);
  }

  // Many small components exercise a much denser set of cusp edges.
  std::vector<Manifold> parts;
  for (int i = 0; i < 129; ++i) {
    parts.push_back(
        Manifold::Tetrahedron().Translate({4.0 * (i % 16), 4.0 * (i / 16), 0}));
  }
  const Polygons projected =
      Manifold::BatchBoolean(parts, OpType::Add).Project();
  ASSERT_EQ(projected.size(), parts.size());
  for (const SimplePolygon& contour : projected) {
    EXPECT_EQ(contour.size(), 4);
    EXPECT_NEAR(CrossSection(contour).Area(), 4, 1e-12);
  }
}

TEST(Manifold, Transform) {
  Manifold cube = Manifold::Cube({1, 2, 3});
  Manifold cube2 = cube;
  cube = cube.Rotate(30, 40, 50).Scale({6, 5, 4}).Translate({1, 2, 3});

  mat3 rX({1.0, 0.0, 0.0},            //
          {0.0, cosd(30), sind(30)},  //
          {0.0, -sind(30), cosd(30)});
  mat3 rY({cosd(40), 0.0, -sind(40)},  //
          {0.0, 1.0, 0.0},             //
          {sind(40), 0.0, cosd(40)});
  mat3 rZ({cosd(50), sind(50), 0.0},   //
          {-sind(50), cosd(50), 0.0},  //
          {0.0, 0.0, 1.0});
  mat3 s;
  s[0][0] = 6;
  s[1][1] = 5;
  s[2][2] = 4;
  mat3x4 transform = mat3x4(s * rZ * rY * rX, vec3(0.0));
  transform[3] = vec3(1, 2, 3);
  cube2 = cube2.Transform(transform);

  Identical(cube.GetMeshGL(), cube2.GetMeshGL());
}

TEST(Manifold, Slice) {
  Manifold cube = Manifold::Cube();
  CrossSection bottom = cube.Slice();
  CrossSection top = cube.Slice(1);
  EXPECT_EQ(bottom.Area(), 1);
  EXPECT_EQ(top.Area(), 0);
}

TEST(Manifold, SliceEmptyObject) {
  Manifold empty;
  EXPECT_TRUE(empty.IsEmpty());
  CrossSection bottom = empty.Slice();
}

TEST(Manifold, Simplify) {
  Polygons polyCircle =
      CrossSection::Circle(1, {20}).Translate({10, 0}).ToPolygons();
  Manifold torus = Manifold::Revolve(polyCircle, {100});
  Manifold simplified = torus.Simplify(0.4);
  EXPECT_NEAR(torus.Volume(), simplified.Volume(), 5);
  EXPECT_NEAR(torus.SurfaceArea(), simplified.SurfaceArea(), 12);

  if (options.exportModels) WriteTestOBJ("torus.obj", simplified);
}

TEST(Manifold, SimplifyNaNTolerance) {
  Polygons polyCircle =
      CrossSection::Circle(1, {20}).Translate({10, 0}).ToPolygons();
  const Manifold torus = Manifold::Revolve(polyCircle, {100});
  // NaN simplifies like 0, at the Manifold's epsilon.
  const Manifold simplified = torus.Simplify(NAN);
  EXPECT_EQ(simplified.Status(), Manifold::Error::NoError);
  EXPECT_EQ(simplified.NumTri(), torus.Simplify().NumTri());
}

TEST(Manifold, SimplifyToEmpty) {
  const Manifold cube = Manifold::Cube();
  const Manifold gone = cube.Translate({100, 0, 0}).Simplify(1e9);
  EXPECT_TRUE(gone.IsEmpty());
  EXPECT_FALSE(gone.BoundingBox().IsFinite());
  EXPECT_EQ((gone + cube).GetEpsilon(), cube.GetEpsilon());
}

TEST(Manifold, MeshID) {
  const Manifold cube = Manifold::Cube();
  MeshGL cubeGL = cube.GetMeshGL();
  cubeGL.runIndex.clear();
  cubeGL.runOriginalID.clear();
  Manifold cube1 = Manifold(cubeGL);
  EXPECT_EQ(cube1.GetMeshGL().runOriginalID[0], 0);
}

TEST(Manifold, MeshRelation) {
  Manifold gyroid = WithPositionColors(Gyroid());
  MeshGL gyroidMeshGL = gyroid.GetMeshGL();

  gyroid = gyroid.Simplify();

  if (options.exportModels) WriteTestOBJ("gyroid.obj", gyroid);

  RelatedGL(gyroid, {gyroidMeshGL});
}

TEST(Manifold, MeshRelationTransform) {
  const Manifold cube = Manifold::Cube();
  const MeshGL cubeGL = cube.GetMeshGL();
  const Manifold turned = cube.Rotate(45, 90);

  RelatedGL(turned, {cubeGL});
}

TEST(Manifold, MeshRelationRefine) {
  Manifold csaszar = WithPositionColors(Csaszar()).AsOriginal();
  MeshGL inGL = csaszar.GetMeshGL();

  RelatedGL(csaszar, {inGL});
  csaszar = csaszar.RefineToLength(1);
  ExpectMeshes(csaszar, {{9019, 18038, 3}});
  RelatedGL(csaszar, {inGL});

  if (options.exportModels) WriteTestOBJ("csaszar.obj", csaszar);
}

TEST(Manifold, MeshRelationRefinePrecision) {
  Manifold source = Manifold(Csaszar()).CalculateNormals(180);
  const Box bbox = source.BoundingBox();
  const vec3 size = bbox.Size();
  const Manifold input = source.SetProperties(
      6, [bbox, size](double* newProp, vec3 pos, const double* oldProp) {
        for (const int i : {0, 1, 2}) {
          newProp[i] = oldProp[i];
          newProp[i + 3] = (pos[i] - bbox.min[i]) / size[i];
        }
      });
  ASSERT_EQ(input.NumProp(), 6);
  MeshGL inGL = input.GetMeshGL();
  ASSERT_EQ(inGL.numProp, 9);
  const int id = inGL.runOriginalID[0];
  Manifold csaszar = Manifold(inGL).CalculateNormals(180).SmoothByNormals();
  ASSERT_EQ(csaszar.NumProp(), 6);

  csaszar = csaszar.RefineToTolerance(0.05);
  ExpectMeshes(csaszar, {{2135, 4270, 6}});
  std::vector<uint32_t> runOriginalID = csaszar.GetMeshGL().runOriginalID;
  EXPECT_EQ(runOriginalID.size(), 1);
  EXPECT_EQ(runOriginalID[0], id);

  if (options.exportModels) WriteTestOBJ("csaszarSmooth.obj", csaszar);
}

TEST(Manifold, MeshGLRoundTrip) {
  const Manifold cylinder = Manifold::Cylinder(2, 1);
  EXPECT_EQ(cylinder.OriginalID(), 0);
  MeshGL inGL = cylinder.TriID2FaceID().GetMeshGL();
  const Manifold cylinder2(inGL);
  const MeshGL outGL = cylinder2.GetMeshGL();

  EXPECT_EQ(inGL.runOriginalID.size(), 1);
  EXPECT_EQ(outGL.runOriginalID.size(), 1);
  EXPECT_EQ(outGL.runOriginalID[0], inGL.runOriginalID[0]);

  RelatedGL(cylinder2, {inGL});
}

void CheckCube(const MeshGL& cubeSTL) {
  Manifold cube(cubeSTL);
  EXPECT_EQ(cube.NumTri(), 12);
  EXPECT_EQ(cube.NumVert(), 8);
  EXPECT_EQ(cube.NumPropVert(), 24);

  EXPECT_FLOAT_EQ(cube.Volume(), 1.0);
  EXPECT_FLOAT_EQ(cube.SurfaceArea(), 6.0);
}

TEST(Manifold, Merge) {
  MeshGL cubeSTL = CubeSTL();
  EXPECT_EQ(cubeSTL.NumTri(), 12);
  EXPECT_EQ(cubeSTL.NumVert(), 36);

  Manifold cubeBad(cubeSTL);
  EXPECT_TRUE(cubeBad.IsEmpty());
  EXPECT_EQ(cubeBad.Status(), Manifold::Error::NotManifold);

  MeshGL merged = cubeSTL.Merge();
  EXPECT_TRUE(cubeSTL.mergeFromVert.empty());
  EXPECT_EQ(merged.mergeFromVert.size(), 28);
  CheckCube(merged);

  EXPECT_EQ(merged.Merge().mergeFromVert, merged.mergeFromVert);
  merged.mergeFromVert.resize(14);
  merged.mergeToVert.resize(14);

  merged = merged.Merge();
  EXPECT_EQ(merged.mergeFromVert.size(), 28);
  CheckCube(merged);
}

TEST(Manifold, MergeTolerance) {
  MeshGL mesh = CubeSTL();
  mesh.vertProperties[0] += 1e-4f;
  EXPECT_EQ(Manifold(mesh.Merge(1e-3f)).Status(), Manifold::Error::NoError);
}

TEST(Manifold, MergeNaNTolerance) {
  const MeshGL mesh = CubeSTL();
  MeshGL64 mesh64;
  mesh64.numProp = mesh.numProp;
  mesh64.vertProperties.assign(mesh.vertProperties.begin(),
                               mesh.vertProperties.end());
  mesh64.triVerts.assign(mesh.triVerts.begin(), mesh.triVerts.end());
  // NaN merges like 0, at the floor of each mesh type.
  EXPECT_FALSE(mesh.Merge().mergeFromVert.empty());
  EXPECT_EQ(mesh.Merge(NAN).mergeFromVert, mesh.Merge().mergeFromVert);
  EXPECT_EQ(mesh64.Merge(NAN).mergeFromVert, mesh64.Merge().mergeFromVert);
}

TEST(Precision, Merge) {
  // The merge floor is the ingest precision of the mesh type times the
  // bounding box scale, which is 0.5 for the centered cube.
  MeshGL mesh = CubeSTL();
  mesh.vertProperties[0] += 1e-4f;
  EXPECT_EQ(Manifold(mesh.Merge()).Status(), Manifold::Error::NotManifold);
  EXPECT_EQ(Manifold(mesh.Merge(1e-3f)).Status(), Manifold::Error::NoError);
  // Shifted to [0, 1], the scale is 1 and an offset from 0 is exact in float.
  mesh = CubeSTL();
  for (size_t v = 0; v < mesh.NumVert(); ++v) {
    for (const int i : {0, 1, 2})
      mesh.vertProperties[mesh.numProp * v + i] += 0.5f;
  }
  size_t atZero = 0;
  while (mesh.vertProperties[mesh.numProp * atZero] != 0) ++atZero;
  mesh.vertProperties[mesh.numProp * atZero] += 0.5 * kDefaultRelativePrecision;
  EXPECT_EQ(Manifold(mesh.Merge()).Status(), Manifold::Error::NoError);

  const MeshGL stl = CubeSTL();
  MeshGL64 mesh64;
  mesh64.numProp = stl.numProp;
  mesh64.vertProperties.assign(stl.vertProperties.begin(),
                               stl.vertProperties.end());
  mesh64.triVerts.assign(stl.triVerts.begin(), stl.triVerts.end());
  MeshGL64 coarse64 = mesh64;
  coarse64.vertProperties[0] += 0.1 * kDefaultRelativePrecision;
  EXPECT_EQ(Manifold(coarse64.Merge()).Status(), Manifold::Error::NotManifold);
  EXPECT_EQ(Manifold(coarse64.Merge(kDefaultRelativePrecision)).Status(),
            Manifold::Error::NoError);
  mesh64.vertProperties[0] += 0.1 * kMinRelativePrecision;
  EXPECT_EQ(Manifold(mesh64.Merge()).Status(), Manifold::Error::NoError);
}

TEST(Precision, Ingest) {
  // WithPrecision is the only way to set the precision.
  static_assert(!std::is_constructible_v<Manifold, const MeshGL&, double>);
  static_assert(!std::is_constructible_v<Manifold, const MeshGL64&, double>);
  const MeshGL64 mesh = Manifold::Sphere(1, {64}).GetMeshGL64();
  const MeshGL meshF = Manifold::Sphere(1, {64}).GetMeshGL();
  // The default follows the representation.
  EXPECT_EQ(kDefaultRelativePrecision, std::numeric_limits<float>::epsilon());
  EXPECT_EQ(Manifold(mesh).GetPrecision(), kMinRelativePrecision);
  EXPECT_EQ(Manifold(meshF).GetPrecision(), kDefaultRelativePrecision);
  ExecutionContext ctx;
  EXPECT_EQ(ctx.FromMeshGL(mesh).GetPrecision(), kMinRelativePrecision);
  EXPECT_EQ(ctx.FromMeshGL(meshF).GetPrecision(), kDefaultRelativePrecision);
  const Manifold coarse = Manifold(mesh).WithPrecision(1e-3);
  EXPECT_EQ(coarse.GetPrecision(), 1e-3);
  EXPECT_DOUBLE_EQ(coarse.GetEpsilon(), 1e-3 * coarse.BoundingBox().Scale());
  // 0, negative values, NaN and infinity give kDefaultRelativePrecision, even
  // for a Manifold made from MeshGL64.
  for (const double unset :
       {0.0, -0.0, -1e-3, -1e-15, -kMinRelativePrecision,
        static_cast<double>(NAN), static_cast<double>(INFINITY),
        static_cast<double>(-INFINITY)}) {
    EXPECT_EQ(Manifold(mesh).WithPrecision(unset).GetPrecision(),
              kDefaultRelativePrecision);
    EXPECT_EQ(Manifold(meshF).WithPrecision(unset).GetPrecision(),
              kDefaultRelativePrecision);
  }
  // Positive values below the floor use the floor.
  EXPECT_EQ(Manifold(meshF).WithPrecision(1e-15).GetPrecision(),
            kMinRelativePrecision);
  EXPECT_EQ(Manifold(meshF).WithPrecision(kMinRelativePrecision).GetPrecision(),
            kMinRelativePrecision);
  EXPECT_EQ(Manifold(meshF).WithPrecision(1e-9).GetPrecision(), 1e-9);
  // The mesh does not store the precision.
  EXPECT_EQ(Manifold(coarse.GetMeshGL64()).GetPrecision(),
            kMinRelativePrecision);
  EXPECT_EQ(Manifold(coarse.GetMeshGL()).GetPrecision(),
            kDefaultRelativePrecision);
  // Transforms and single-input operations keep the precision.
  EXPECT_EQ(coarse.Translate({10, 0, 0}).GetPrecision(), 1e-3);
  EXPECT_DOUBLE_EQ(coarse.Translate({10, 0, 0}).GetEpsilon(), 1e-3 * 11);
  EXPECT_EQ(coarse.Refine(2).GetPrecision(), 1e-3);
  EXPECT_EQ(coarse.Simplify().GetPrecision(), 1e-3);
  EXPECT_EQ(coarse.AsOriginal().GetPrecision(), 1e-3);
  EXPECT_EQ(coarse.CalculateNormals().GetPrecision(), 1e-3);
  EXPECT_EQ(coarse.Warp([](vec3& v) { v.x *= 2; }).GetPrecision(), 1e-3);
  EXPECT_EQ(coarse
                .SetProperties(4, [](double* newProp, vec3 pos,
                                     const double*) { newProp[3] = pos.x; })
                .GetPrecision(),
            1e-3);
  EXPECT_EQ(coarse.Hull().GetPrecision(), 1e-3);
  EXPECT_EQ(coarse.TrimByPlane({0, 0, 1}, 0).GetPrecision(), 1e-3);
  const auto halves = coarse.SplitByPlane({0, 0, 1}, 0);
  EXPECT_EQ(halves.first.GetPrecision(), 1e-3);
  EXPECT_EQ(halves.second.GetPrecision(), 1e-3);
  for (const Manifold& part :
       (coarse + coarse.Translate({5, 0, 0})).Decompose()) {
    EXPECT_EQ(part.GetPrecision(), 1e-3);
  }
  // A part with an inner void is rebuilt by a separate Decompose path.
  const Manifold hollow =
      (Manifold::Cube(vec3(4), true) - Manifold::Cube(vec3(2), true))
          .WithPrecision(1e-3);
  for (const Manifold& part :
       (hollow + coarse.Translate({10, 0, 0})).Decompose()) {
    EXPECT_EQ(part.GetPrecision(), 1e-3);
  }
}

TEST(Precision, IngestIgnoresUnusedVerts) {
  MeshGL64 mesh = Manifold::Cube(vec3(1)).Refine(2).GetMeshGL64();
  int raised = 0;
  for (size_t v = 0; v < mesh.NumVert(); ++v) {
    double* p = &mesh.vertProperties[mesh.numProp * v];
    if (p[0] == 0.5 && p[1] == 0.5 && p[2] == 1) {
      p[2] += 1e-4;
      ++raised;
    }
  }
  ASSERT_EQ(raised, 1);
  const size_t numVert =
      Manifold(mesh).WithPrecision(1e-6).Simplify().NumVert();

  // A far unused vertex must not widen the tolerance.
  mesh.vertProperties.push_back(1000);
  for (size_t i = 1; i < mesh.numProp; ++i) mesh.vertProperties.push_back(0);
  const Manifold m = Manifold(mesh).WithPrecision(1e-6);
  EXPECT_DOUBLE_EQ(m.GetEpsilon(), 1e-6 * m.BoundingBox().Scale());
  EXPECT_EQ(m.Simplify().NumVert(), numVert);

  ExecutionContext ctx;
  const Manifold viaCtx = ctx.FromMeshGL(mesh);
  EXPECT_DOUBLE_EQ(viaCtx.GetEpsilon(),
                   kMinRelativePrecision * viaCtx.BoundingBox().Scale());
}

TEST(Precision, BuiltShapesDefaultToFloat) {
  const CrossSection square = CrossSection::Square(vec2(1));
  const std::vector<Manifold> shapes = {
      Manifold(),
      Manifold::Tetrahedron(),
      Manifold::Cube(),
      Manifold::Cylinder(1, 1),
      Manifold::Sphere(1),
      Manifold::Extrude(square.ToPolygons(), 1),
      Manifold::Revolve(square.Translate({1, 0}).ToPolygons()),
      Manifold::LevelSet([](vec3 p) { return 1 - la::length(p); },
                         {vec3(-2), vec3(2)}, 0.5),
      Manifold::Hull({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}),
  };
  for (const Manifold& shape : shapes) {
    EXPECT_EQ(shape.GetPrecision(), kDefaultRelativePrecision);
  }
  ExecutionContext ctx;
  EXPECT_EQ(ctx.LevelSet([](vec3 p) { return 1 - la::length(p); },
                         {vec3(-2), vec3(2)}, 0.5)
                .GetPrecision(),
            kDefaultRelativePrecision);
  // A MeshGL64 operand keeps a Boolean at double precision.
  const Manifold fine(Manifold::Sphere(1).GetMeshGL64());
  EXPECT_EQ((fine - Manifold::Cube()).GetPrecision(), kMinRelativePrecision);
}

TEST(Precision, WithPrecisionAffectsOnlyLaterOps) {
  const Manifold fine =
      Manifold::Sphere(1, {256}) - Manifold::Cube(vec3(1)).Rotate(10, 20, 30);
  const size_t numTri = fine.NumTri();
  const Manifold coarse = fine.WithPrecision(1e-2);
  // The geometry is not recomputed.
  EXPECT_EQ(coarse.NumTri(), numTri);
  EXPECT_EQ(coarse.GetPrecision(), 1e-2);
  EXPECT_DOUBLE_EQ(coarse.GetEpsilon(), 1e-2 * coarse.BoundingBox().Scale());
  EXPECT_EQ(fine.GetPrecision(), kDefaultRelativePrecision);
  EXPECT_EQ(fine.WithPrecision(kMinRelativePrecision).GetPrecision(),
            kMinRelativePrecision);
  EXPECT_EQ(fine.WithPrecision(1e-15).GetPrecision(), kMinRelativePrecision);
  EXPECT_EQ(fine.WithPrecision(1e-12).WithPrecision(0).GetPrecision(),
            kDefaultRelativePrecision);
  EXPECT_EQ(fine.WithPrecision(1e-12).WithPrecision(-1).GetPrecision(),
            kDefaultRelativePrecision);
  EXPECT_EQ(fine.WithPrecision(INFINITY).GetPrecision(),
            kDefaultRelativePrecision);
  // Later operations use it.
  const Manifold cutter = Manifold::Cube(vec3(3), true).Translate({2, 0, 0});
  const Manifold coarseCut = coarse - cutter.WithPrecision(1e-2);
  const Manifold fineCut = fine - cutter;
  EXPECT_EQ(coarseCut.GetPrecision(), 1e-2);
  EXPECT_LT(coarseCut.NumTri(), fineCut.NumTri());
  EXPECT_EQ(fine.NumTri(), numTri);
  EXPECT_EQ((fine - cutter).NumTri(), fineCut.NumTri());
}

TEST(Precision, SmallestKeepsFineDetail) {
  const Manifold sphere = Manifold::Sphere(1, {256});
  const Manifold cube = Manifold::Cube(vec3(1)).Rotate(10, 20, 30);
  const Manifold allFine = sphere - cube;
  const Manifold allCoarse =
      sphere.WithPrecision(1e-2) - cube.WithPrecision(1e-2);
  const Manifold mixed = sphere - cube.WithPrecision(1e-2);
  EXPECT_EQ(allCoarse.GetPrecision(), 1e-2);
  EXPECT_EQ(mixed.GetPrecision(), kDefaultRelativePrecision);
  EXPECT_LT(allCoarse.NumTri(), allFine.NumTri());
  EXPECT_EQ(mixed.NumTri(), allFine.NumTri());
  // An empty operand is ignored.
  EXPECT_EQ((sphere.WithPrecision(1e-2) + Manifold()).GetPrecision(), 1e-2);
  EXPECT_EQ((sphere.WithPrecision(1e-2) - Manifold()).GetPrecision(), 1e-2);
  // Compose and Hull follow the same rule.
  const Manifold far = cube.WithPrecision(1e-2).Translate({10, 0, 0});
  EXPECT_EQ((sphere.WithPrecision(1e-3) + far).GetPrecision(), 1e-3);
  EXPECT_EQ(Manifold::Hull({sphere.WithPrecision(1e-3), far}).GetPrecision(),
            1e-3);
  EXPECT_EQ(Manifold::Hull({Manifold(), far}).GetPrecision(), 1e-2);
  EXPECT_EQ(sphere.WithPrecision(1e-3).MinkowskiSum(far).GetPrecision(), 1e-3);
  // An empty Minkowski result has the smaller precision, in either order.
  const Manifold empty2 = Manifold().WithPrecision(1e-2);
  const Manifold empty3 = Manifold().WithPrecision(1e-3);
  EXPECT_EQ(empty2.MinkowskiSum(empty3).GetPrecision(), 1e-3);
  EXPECT_EQ(empty3.MinkowskiSum(empty2).GetPrecision(), 1e-3);
  EXPECT_EQ(empty2.MinkowskiDifference(cube.WithPrecision(1e-3)).GetPrecision(),
            1e-3);
  EXPECT_EQ(far.MinkowskiDifference(cube.WithPrecision(1e-3).Scale(vec3(0.1)))
                .GetPrecision(),
            1e-3);
  const auto [inside, outside] = cube.WithPrecision(1e-2).Split(
      cube.WithPrecision(1e-3).Translate({0.5, 0.5, 0.5}));
  EXPECT_EQ(inside.GetPrecision(), 1e-3);
  EXPECT_EQ(outside.GetPrecision(), 1e-3);
}

TEST(Precision, UnionOrder) {
  const Manifold a = Manifold::Sphere(1, {128}).WithPrecision(1e-2);
  const Manifold b = Manifold::Cube(vec3(1)).Rotate(10, 20, 30);
  const Manifold c =
      Manifold::Cylinder(2, 0.3, 0.3, {64}).Translate({0.5, 0, -1});
  const Manifold ab = a + b;
  const Manifold ba = b + a;
  EXPECT_EQ(ab.NumTri(), ba.NumTri());
  EXPECT_EQ(ab.GetPrecision(), kDefaultRelativePrecision);
  EXPECT_EQ(ba.GetPrecision(), kDefaultRelativePrecision);
  const Manifold abc = Manifold::BatchBoolean({a, b, c}, OpType::Add);
  const Manifold cba = Manifold::BatchBoolean({c, b, a}, OpType::Add);
  EXPECT_EQ(abc.NumTri(), cba.NumTri());
  EXPECT_EQ(abc.GetPrecision(), kDefaultRelativePrecision);
  EXPECT_EQ(cba.GetPrecision(), kDefaultRelativePrecision);
}

TEST(Precision, SharedSubtree) {
  auto build = []() {
    return Manifold::Sphere(1, {256}).WithPrecision(1e-2) -
           Manifold::Cube(vec3(1)).Rotate(10, 20, 30).WithPrecision(1e-2);
  };
  const size_t alone = build().NumTri();
  EXPECT_EQ(build().GetPrecision(), 1e-2);

  // Evaluated first as part of a finer tree, the subtree keeps its own
  // precision.
  const Manifold shared = build();
  const Manifold parent = shared + Manifold::Cube(vec3(1)).Translate({5, 0, 0});
  const Manifold parent2 = shared.Translate({0, 0, 3}) ^ Manifold::Sphere(5);
  EXPECT_EQ(parent.GetPrecision(), kDefaultRelativePrecision);
  EXPECT_GT(parent2.NumTri(), 0);
  EXPECT_EQ(shared.NumTri(), alone);
  EXPECT_EQ(shared.GetPrecision(), 1e-2);
  EXPECT_EQ(parent.NumTri(),
            (build() + Manifold::Cube(vec3(1)).Translate({5, 0, 0})).NumTri());
}

TEST(Precision, WithPrecisionThenTransform) {
  const Manifold cube = Manifold::Cube(vec3(1));
  const Manifold moved = cube.WithPrecision(1e-3).Translate({10, 0, 0});
  EXPECT_EQ(moved.GetPrecision(), 1e-3);
  EXPECT_DOUBLE_EQ(moved.GetEpsilon(), 1e-3 * 11);
  EXPECT_EQ(moved.Scale(vec3(2)).GetPrecision(), 1e-3);
  EXPECT_EQ(moved.Mirror(vec3(0.0)).GetPrecision(), 1e-3);
  const auto halves = Manifold().WithPrecision(1e-3).SplitByPlane({0, 0, 1}, 0);
  EXPECT_EQ(halves.first.GetPrecision(), 1e-3);
  EXPECT_EQ(halves.second.GetPrecision(), 1e-3);
  EXPECT_EQ(cube.GetPrecision(), kDefaultRelativePrecision);
  EXPECT_DOUBLE_EQ(cube.GetEpsilon(), kDefaultRelativePrecision);
}

TEST(Precision, WithPrecisionOnError) {
  const Manifold invalid(CubeSTL());
  EXPECT_EQ(invalid.Status(), Manifold::Error::NotManifold);
  const Manifold coarse = invalid.WithPrecision(1e-3);
  EXPECT_EQ(coarse.Status(), Manifold::Error::NotManifold);
  EXPECT_EQ(coarse.GetPrecision(), 1e-3);
  EXPECT_EQ(invalid.GetPrecision(), kDefaultRelativePrecision);
}

TEST(Precision, EmptyEpsilon) {
  const Manifold empty;
  const Manifold coarseEmpty = empty.WithPrecision(1e-3);
  EXPECT_EQ(empty.GetEpsilon(), -1);
  EXPECT_EQ(coarseEmpty.GetEpsilon(), -1);
  EXPECT_EQ(coarseEmpty.GetPrecision(), 1e-3);
  EXPECT_EQ(empty.Translate({1, 0, 0}).GetEpsilon(), -1);
  EXPECT_EQ(coarseEmpty.Translate({1, 0, 0}).GetEpsilon(), -1);
  EXPECT_EQ(Manifold::BatchBoolean({empty, coarseEmpty, empty}, OpType::Add)
                .GetEpsilon(),
            -1);
  const Manifold cube = Manifold::Cube();
  EXPECT_EQ((cube - cube).GetEpsilon(), -1);
  EXPECT_EQ((cube ^ cube.Translate({5, 0, 0})).GetEpsilon(), -1);
  EXPECT_EQ(cube.Simplify(1e9).GetEpsilon(), -1);
#ifndef MANIFOLD_NO_IOSTREAM
  for (const Manifold& m :
       {coarseEmpty.Translate({1, 0, 0}), cube.Simplify(1e9)}) {
    std::stringstream obj;
    m.WriteOBJ(obj);
    EXPECT_NE(obj.str().find("# epsilon = -1"), std::string::npos);
  }
#endif
}

TEST(Precision, EmptyBooleanTakesMin) {
  const Manifold fine(Manifold::Cube().GetMeshGL64());
  const Manifold cube = Manifold::Cube();
  const Manifold far = cube.Translate({5, 0, 0});
  const Manifold empty;
  const Manifold coarseEmpty = empty.WithPrecision(1e-2);
  // An empty result has the smaller precision of the two inputs, in either
  // order, including an empty input.
  EXPECT_EQ((fine ^ empty).GetPrecision(), kMinRelativePrecision);
  EXPECT_EQ((empty ^ fine).GetPrecision(), kMinRelativePrecision);
  EXPECT_EQ((empty - fine).GetPrecision(), kMinRelativePrecision);
  EXPECT_EQ((fine ^ far).GetPrecision(), kMinRelativePrecision);
  EXPECT_EQ((far ^ fine).GetPrecision(), kMinRelativePrecision);
  EXPECT_EQ((fine - fine).GetPrecision(), kMinRelativePrecision);
  EXPECT_EQ((coarseEmpty + empty.WithPrecision(1e-3)).GetPrecision(), 1e-3);
  EXPECT_EQ((empty.WithPrecision(1e-3) + coarseEmpty).GetPrecision(), 1e-3);
  EXPECT_EQ(
      Manifold::BatchBoolean(
          {coarseEmpty, empty.WithPrecision(1e-3), coarseEmpty}, OpType::Add)
          .GetPrecision(),
      1e-3);
  EXPECT_EQ((coarseEmpty ^ cube).GetPrecision(), kDefaultRelativePrecision);
  EXPECT_EQ((cube ^ coarseEmpty).GetPrecision(), kDefaultRelativePrecision);
  // A non-empty result ignores an empty input.
  EXPECT_EQ((cube.WithPrecision(1e-2) - empty).GetPrecision(), 1e-2);
  EXPECT_EQ((empty + cube.WithPrecision(1e-2)).GetPrecision(), 1e-2);
  // BatchBoolean gives the same in any order.
  for (const auto& parts : std::vector<std::vector<Manifold>>{
           {fine, cube, far}, {cube, far, fine}, {far, fine, cube}}) {
    const Manifold result = Manifold::BatchBoolean(parts, OpType::Intersect);
    EXPECT_TRUE(result.IsEmpty());
    EXPECT_EQ(result.GetPrecision(), kMinRelativePrecision);
  }
}

TEST(Precision, BooleanTakesMin) {
  const Manifold sphere = Manifold::Sphere(1, {64});
  const Manifold cube = Manifold::Cube(vec3(1));
  EXPECT_EQ((sphere.WithPrecision(1e-3) + cube).GetPrecision(),
            kDefaultRelativePrecision);
  EXPECT_EQ((cube - sphere.WithPrecision(1e-3)).GetPrecision(),
            kDefaultRelativePrecision);
  EXPECT_EQ((sphere.WithPrecision(1e-2).Translate({0.5, 0, 0}) ^
             cube.WithPrecision(1e-3))
                .GetPrecision(),
            1e-3);
}

TEST(Manifold, MergeCollapsedEdgeParity) {
  MeshGL mesh;
  mesh.vertProperties = {0, 0, 0,  //
                         0, 0, 0,  //
                         1, 0, 0,  //
                         0, 0, 0,  //
                         0, 0, 0,  //
                         0, 1, 0};
  mesh.mergeFromVert = {1, 4};
  mesh.mergeToVert = {0, 3};

  // These triangles collapse to (0, 0, 2) and (3, 3, 5). Their opposing
  // non-degenerate edges cancel, leaving one self-edge per triangle. Two
  // copies of each triangle cancel those self-edges as well. All four
  // collapsed verts share a position, so any vert found open merges across the
  // two pairs.
  mesh.triVerts = {0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5};
  MeshGL merged = mesh.Merge();
  EXPECT_EQ(merged.mergeFromVert, (std::vector<uint32_t>{1, 4}));
  EXPECT_EQ(merged.mergeToVert, (std::vector<uint32_t>{0, 3}));

  mesh.triVerts = {0, 1, 2, 3, 4, 5};
  merged = mesh.Merge();
  EXPECT_EQ(merged.mergeFromVert, (std::vector<uint32_t>{1, 3, 4}));
  EXPECT_EQ(merged.mergeToVert, (std::vector<uint32_t>{0, 0, 0}));
}

TEST(Manifold, MergeSingleCollapsedTriangle) {
  const auto check = [](auto mesh) {
    mesh.vertProperties = {0, 0, 0, 0, 0, 0, 1, 0, 0};
    for (const bool premerged : {false, true}) {
      if (premerged) {
        mesh.triVerts = {0, 1, 2};
        mesh.mergeFromVert = {1};
        mesh.mergeToVert = {0};
      } else {
        mesh.triVerts = {0, 0, 2};
      }
      // The opposing edges cancel, leaving one self-edge and one open vertex.
      // There is no other vertex to merge with, so keep the existing mapping.
      const auto merged = mesh.Merge();
      EXPECT_EQ(merged.vertProperties, mesh.vertProperties);
      EXPECT_EQ(merged.triVerts, mesh.triVerts);
      EXPECT_EQ(merged.mergeFromVert, mesh.mergeFromVert);
      EXPECT_EQ(merged.mergeToVert, mesh.mergeToVert);
    }
  };
  check(MeshGL());
  check(MeshGL64());
}

TEST(Manifold, MergeEmpty) {
  MeshGL shape;
  shape.numProp = 7;
  shape.triVerts = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
  shape.vertProperties = {0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  0.5,  -0.43450000882149, 0.0, 0.0, 0.0, 1.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          0.0,  -0.5, 0.434500008821487, 0.0, 0.0, 1.0, 0.0,
                          0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          -0.0, 0.5,  0.434500008821487, 0.0, 0.0, 0.0, 1.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          0.0,  0.5,  -0.43450000882149, 0.0, 0.0, 1.0, 0.0,
                          0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 1.0, 1.0,
                          0.0,  -0.5, 0.434500008821487, 0.0, 0.0, 0.0, 1.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 1.0, 1.0,
                          0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          -0.0, 0.5,  0.434500008821487, 0.0, 0.0, 1.0, 0.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 0.0, 0.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 1.0, 1.0,
                          -0.0, 0.5,  0.434500008821487, 0.0, 0.0, 0.0, 1.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 1.0, 1.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 0.0, 0.0,
                          -0.0, -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 0.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  -0.5, 0.434500008821487, 0.0, 0.0, 0.0, 1.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          -0.0, -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 0.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 0.0, 0.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  0.5,  -0.43450000882149, 0.0, 0.0, 0.0, 1.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 0.0, 0.0,
                          -0.0, -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 0.0};
  Manifold man(shape.Merge());
  EXPECT_EQ(man.Status(), Manifold::Error::NoError);
  man = man.Simplify();
  EXPECT_TRUE(man.IsEmpty());
}

TEST(Manifold, PinchedVert) {
  MeshGL shape;
  shape.numProp = 3;
  shape.vertProperties = {0,        0,  0,   //
                          1,        1,  0,   //
                          1,        -1, 0,   //
                          -0.00001, 0,  0,   //
                          -1,       -1, -0,  //
                          -1,       1,  0,   //
                          0,        0,  2,   //
                          0,        0,  -2};
  shape.triVerts = {0, 2, 6,  //
                    2, 1, 6,  //
                    1, 0, 6,  //
                    4, 3, 6,  //
                    3, 5, 6,  //
                    5, 4, 6,  //
                    2, 0, 4,  //
                    0, 3, 4,  //
                    3, 0, 1,  //
                    3, 1, 5,  //
                    7, 2, 4,  //
                    7, 4, 5,  //
                    7, 5, 1,  //
                    7, 1, 2};
  Manifold touch(shape);
  EXPECT_FALSE(touch.IsEmpty());
  EXPECT_EQ(touch.Status(), Manifold::Error::NoError);
  EXPECT_EQ(touch.Genus(), 0);
}

TEST(Manifold, FaceIDRoundTrip) {
  const Manifold cube = Manifold::Cube();
  EXPECT_GE(cube.OriginalID(), 0);
  MeshGL inGL = cube.GetMeshGL();
  EXPECT_TRUE(inGL.faceID.empty());
  inGL.faceID = {3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5};

  const Manifold cube2(inGL);
  const MeshGL outGL = cube2.GetMeshGL();
  EXPECT_EQ(NumUnique(outGL.faceID), 2);
}

TEST(Manifold, MirrorUnion) {
  auto a = Manifold::Cube({5., 5., 5.}, true);
  auto b = a.Translate({2.5, 2.5, 2.5});
  auto result = a + b + b.Mirror({1, 1, 0});

  if (options.exportModels) WriteTestOBJ("manifold_mirror_union.obj", result);

  auto vol_a = a.Volume();
  EXPECT_FLOAT_EQ(vol_a * 2.75, result.Volume());
  EXPECT_TRUE(a.Mirror(vec3(0.0)).IsEmpty());
}

TEST(Manifold, MirrorUnion2) {
  auto a = Manifold::Cube();
  auto result = Manifold::BatchBoolean({a.Mirror({1, 0, 0})}, OpType::Add);
  EXPECT_TRUE(result.MatchesTriNormals());
}

TEST(Manifold, MirroredNormals) {
  // Issue #1781: Ensure that when a Manifold is mirrored the normals are
  // mirrored along with it, so the resulting mesh vertices do not get
  // duplicated. We check that the MeshGL vertex count remains the same, and
  // that the mirrored normals remain outward-pointing.
  const Manifold s = Manifold::Sphere(1.0, {32}).CalculateNormals(180);
  const MeshGL mesh = s.GetMeshGL();
  const MeshGL mirrored = s.Mirror({1, 0, 0}).GetMeshGL();
  EXPECT_EQ(mesh.NumVert(), mirrored.NumVert());
  EXPECT_TRUE(mirrored.HasNormals(0));
  const auto [good, bad] = CountSphereNormalAlignment(
      mirrored, 1.0, 3, [](vec3 pos) { return la::normalize(pos); });
  EXPECT_EQ(bad, 0);
  EXPECT_EQ(good, static_cast<int>(mirrored.NumVert()));
}

TEST(Manifold, Invalid) {
  auto invalid = Manifold::Error::InvalidConstruction;
  auto circ = CrossSection::Circle(10.);

  EXPECT_EQ(Manifold::Sphere(0).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(0, 5).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(2, -5).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(2, 0).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(2, 0, 0).Status(), invalid);
  EXPECT_EQ(Manifold::Cube(vec3(0.0)).Status(), invalid);
  EXPECT_EQ(Manifold::Cube({-1, 1, 1}).Status(), invalid);
  EXPECT_EQ(Manifold::Extrude(circ.ToPolygons(), 0.).Status(), invalid);
}

// Empty input gives an empty result with no error, however it is extruded or
// revolved, as long as the parameters are valid.
TEST(Manifold, EmptyCrossSectionGivesEmpty) {
  const std::vector<Polygons> empties = {
      {},
      {{}},
      CrossSection::Circle(-2.).ToPolygons(),
      CrossSection::Square(vec2(0.0)).ToPolygons(),
      {{{0, 0}}, {{1, 0}, {2, 0}}},
  };
  // Revolve also drops contours entirely on the negative X side.
  std::vector<Polygons> revolveEmpties = empties;
  revolveEmpties.push_back(
      CrossSection::Square({1, 1}).Translate({-2, 0}).ToPolygons());
  const auto expectEmpty = [](const Manifold& m) {
    EXPECT_EQ(m.Status(), Manifold::Error::NoError);
    EXPECT_TRUE(m.IsEmpty());
  };
  for (size_t i = 0; i < empties.size(); ++i) {
    SCOPED_TRACE(testing::Message() << "empty cross-section " << i);
    expectEmpty(Manifold::Extrude(empties[i], 1));
    expectEmpty(Manifold::Extrude(empties[i], 2, 3, 45, {0, 0}));
    expectEmpty(Manifold::Extrude(empties[i], 2, 0, 0, {2, 0.5}));
  }
  for (size_t i = 0; i < revolveEmpties.size(); ++i) {
    SCOPED_TRACE(testing::Message() << "empty cross-section " << i);
    for (double degrees : {360.0, 90.0, -90.0, 720.0}) {
      for (int segments : {0, 12}) {
        expectEmpty(Manifold::Revolve(revolveEmpties[i], {segments}, degrees));
      }
    }
  }
}

// Invalid parameters are an error whether or not the cross-section is empty.
TEST(Manifold, InvalidParameterWithEmptyCrossSection) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Polygons square = CrossSection::Square({1, 1}).ToPolygons();
  const Polygons offset =
      CrossSection::Square({1, 1}).Translate({1, 0}).ToPolygons();
  const auto expectInvalid = [](const Manifold& m) {
    EXPECT_EQ(m.Status(), Manifold::Error::InvalidConstruction);
    EXPECT_TRUE(m.IsEmpty());
  };
  for (const Polygons& polys : {Polygons(), square}) {
    SCOPED_TRACE(testing::Message() << "contours " << polys.size());
    for (double height : {0.0, -1.0, nan, inf}) {
      expectInvalid(Manifold::Extrude(polys, height));
    }
    expectInvalid(Manifold::Extrude(polys, 1, -1));
    expectInvalid(Manifold::Extrude(polys, 1, 0, nan));
    for (vec2 scaleTop :
         {vec2(-1, 1), vec2(1, -0.5), vec2(nan, 1), vec2(1, inf)}) {
      expectInvalid(Manifold::Extrude(polys, 1, 0, 0, scaleTop));
    }
  }
  for (const Polygons& polys : {Polygons(), offset}) {
    SCOPED_TRACE(testing::Message() << "contours " << polys.size());
    for (double degrees : {0.0, nan, inf, -inf}) {
      expectInvalid(Manifold::Revolve(polys, {}, degrees));
    }
  }
}

TEST(Manifold, NonFiniteCubeIsInvalid) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (double value : {nan, inf, -inf}) {
    for (int axis = 0; axis < 3; ++axis) {
      for (bool center : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << value << ", axis=" << axis << ", center=" << center);
        vec3 size(1);
        size[axis] = value;
        const Manifold cube = Manifold::Cube(size, center);
        EXPECT_EQ(cube.Status(), Manifold::Error::InvalidConstruction);
        EXPECT_TRUE(cube.IsEmpty());
      }
    }
  }
}

TEST(Manifold, NonFiniteSphereIsInvalid) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (double radius : {nan, inf, -inf}) {
    for (int segments : {0, 8}) {
      SCOPED_TRACE(testing::Message() << radius << ", segments=" << segments);
      const Manifold sphere = Manifold::Sphere(radius, {segments});
      EXPECT_EQ(sphere.Status(), Manifold::Error::InvalidConstruction);
      EXPECT_TRUE(sphere.IsEmpty());
    }
  }
}

TEST(Manifold, NonFiniteCylinderIsInvalid) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto expectInvalid = [](const Manifold& cylinder) {
    EXPECT_EQ(cylinder.Status(), Manifold::Error::InvalidConstruction);
    EXPECT_TRUE(cylinder.IsEmpty());
  };
  for (double value : {nan, inf, -inf}) {
    for (bool center : {false, true}) {
      SCOPED_TRACE(testing::Message() << value << ", center=" << center);
      EXPECT_NO_THROW(
          expectInvalid(Manifold::Cylinder(value, 1, 1, {8}, center)));
      EXPECT_NO_THROW(
          expectInvalid(Manifold::Cylinder(1, value, 1, {8}, center)));
      EXPECT_NO_THROW(
          expectInvalid(Manifold::Cylinder(1, 1, value, {8}, center)));
      EXPECT_NO_THROW(
          expectInvalid(Manifold::Cylinder(1, 0, value, {8}, center)));
    }
  }
}

TEST(Manifold, NonFiniteRevolveCoordinatesAreInvalid) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const SimplePolygon square{{1, 0}, {2, 0}, {2, 1}, {1, 1}};
  for (double value : {nan, inf, -inf}) {
    for (int axis : {0, 1}) {
      for (int segments : {0, 16}) {
        for (double degrees : {180.0, -180.0, 360.0}) {
          SCOPED_TRACE(testing::Message()
                       << value << ", axis=" << axis
                       << ", segments=" << segments << ", degrees=" << degrees);
          SimplePolygon invalid = square;
          invalid[1][axis] = value;
          const Manifold result =
              Manifold::Revolve({invalid}, {segments}, degrees);
          EXPECT_EQ(result.Status(), Manifold::Error::InvalidConstruction);
          EXPECT_TRUE(result.IsEmpty());
        }
      }
    }
    // Validate before clipping too: a contour entirely left of the axis
    // must not conceal a non-finite coordinate beside a valid contour.
    SimplePolygon clipped{{-2, 0}, {-1, 0}, {-1, 1}, {-2, 1}};
    clipped[1].y = value;
    EXPECT_EQ(Manifold::Revolve({square, clipped}, {16}).Status(),
              Manifold::Error::InvalidConstruction);
  }
}

TEST(Manifold, RevolveStillIgnoresShortContours) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const SimplePolygon square{{1, 0}, {2, 0}, {2, 1}, {1, 1}};
  for (double degrees : {180.0, -180.0, 360.0}) {
    const Manifold expected = Manifold::Revolve({square}, {16}, degrees);
    const Manifold result = Manifold::Revolve(
        {square, {}, {{nan, inf}}, {{inf, nan}, {-inf, nan}}}, {16}, degrees);
    EXPECT_EQ(result.Status(), Manifold::Error::NoError);
    EXPECT_DOUBLE_EQ(result.Volume(), expected.Volume());
    EXPECT_EQ(result.NumVert(), expected.NumVert());
    EXPECT_EQ(result.NumTri(), expected.NumTri());
  }
}

TEST(Manifold, CylinderFiniteNegativeTopRadiusUsesBottomRadius) {
  const Manifold expected = Manifold::Cylinder(2, 3, 3, {8});
  for (double radiusHigh : {-1.0, -2.0}) {
    const Manifold cylinder = Manifold::Cylinder(2, 3, radiusHigh, {8});
    EXPECT_EQ(cylinder.Status(), Manifold::Error::NoError);
    EXPECT_FALSE(cylinder.IsEmpty());
    EXPECT_DOUBLE_EQ(cylinder.Volume(), expected.Volume());
    EXPECT_EQ(cylinder.NumVert(), expected.NumVert());
    EXPECT_EQ(cylinder.NumTri(), expected.NumTri());
  }
}

TEST(Manifold, CylinderExtremeRadiusRatio) {
  constexpr int segments = 8;
  constexpr double height = 2;
  // The first pair includes a subnormal radius; the second overflows the
  // radius ratio even though both inputs are normal finite doubles. Include
  // an ordinary frustum as a control.
  for (const vec2 radii : {vec2(1e-310, 1), vec2(1e-300, 1e10), vec2(0.5, 1)}) {
    const double expectedVolume =
        segments * std::sin(2 * kPi / segments) * height / 6 *
        (radii.x * radii.x + radii.x * radii.y + radii.y * radii.y);
    for (bool reverse : {false, true}) {
      for (bool center : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << radii.x << ", " << radii.y << ", reverse=" << reverse
                     << ", center=" << center);
        const double radiusLow = reverse ? radii.y : radii.x;
        const double radiusHigh = reverse ? radii.x : radii.y;
        const Manifold cylinder = Manifold::Cylinder(
            height, radiusLow, radiusHigh, {segments}, center);
        ASSERT_EQ(cylinder.Status(), Manifold::Error::NoError);
        ASSERT_FALSE(cylinder.IsEmpty());
        EXPECT_NEAR(cylinder.Volume() / expectedVolume, 1, 1e-12);
        const Box bounds = cylinder.BoundingBox();
        EXPECT_NEAR(bounds.min.x / radii.y, -1, 1e-12);
        EXPECT_NEAR(bounds.max.x / radii.y, 1, 1e-12);
        EXPECT_NEAR(bounds.min.y / radii.y, -1, 1e-12);
        EXPECT_NEAR(bounds.max.y / radii.y, 1, 1e-12);
        EXPECT_DOUBLE_EQ(bounds.min.z, center ? -height / 2 : 0);
        EXPECT_DOUBLE_EQ(bounds.max.z, center ? height / 2 : height);
        // A slice away from the midpoint distinguishes the two orientations.
        const double quarterRadius = 0.75 * radiusLow + 0.25 * radiusHigh;
        const double expectedArea = segments * std::sin(2 * kPi / segments) /
                                    2 * quarterRadius * quarterRadius;
        const CrossSection slice(cylinder.Slice(bounds.min.z + height / 4));
        EXPECT_NEAR(slice.Area() / expectedArea, 1, 1e-10);
        for (double coordinate : cylinder.GetMeshGL64().vertProperties) {
          EXPECT_TRUE(std::isfinite(coordinate));
        }
      }
    }
  }
}

TEST(Manifold, MergeDegenerates) {
  MeshGL cube = Manifold::Cube(vec3(1), true).GetMeshGL();
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
  squash = squash.Merge();
  EXPECT_FALSE(squash.mergeFromVert.empty());
  // Manifold should remove the triangle with two references to the same vert.
  Manifold squashed = Manifold(squash);
  EXPECT_FALSE(squashed.IsEmpty());
  EXPECT_EQ(squashed.Status(), Manifold::Error::NoError);
}

TEST(Manifold, MergeRefine) {
  MeshGL mesh;
  mesh.vertProperties = {
      0.01383194,  -1.88699961, -1.09618223, -0.72694844, -1.30568409,
      -1.09618223, -0.72694844, -1.30568397, -1.09618223, -1.14769018,
      0.00177073,  -1.09618223, -0.72777849, 1.30731535,  -1.09618223,
      -0.72777843, 1.30731547,  -1.09618223, 0.01344040,  1.88713050,
      -1.09618223, 0.75422078,  1.30581498,  -1.09618223, 0.75422555,
      1.30580890,  -1.09618223, 1.14608598,  -0.00000003, -1.09618223,
      0.75343317,  -1.30844986, -1.09618223, 0.00032274,  2.45551205,
      0.70305431,  -0.00045545, 0.49932927,  1.63340366,  -0.00045485,
      0.49933004,  1.63340306,  0.59210259,  2.54137087,  0.46526274,
      -0.00083422, -0.50377685, 1.63340366,  -0.00153223, -2.45697832,
      0.70305431,  0.59018338,  -2.54327321, 0.46526241,  0.59018356,
      -2.54327297, 0.46526247,  -0.00083402, -0.50377715, 1.63340342,
      -0.00153225, -2.45697832, 0.70305425,  -0.00083453, -0.50377727,
      1.63340330,  -0.59329146, -2.54207492, 0.46526280,  -0.00045549,
      0.49932927,  1.63340366,  -0.59137100, 2.54256892,  0.46526301,
      -0.59137243, 2.54256916,  0.46526241,  -1.09916806, 1.80846310,
      0.62867594,  -1.09916782, 1.80846262,  0.62867624,  1.09932983,
      1.80623639,  0.62867647,  1.09933031,  1.80623722,  0.62867594,
      0.59210330,  2.54137087,  0.46526241,  1.09796500,  -1.80969083,
      0.62867600,  -1.10053313, -1.80746412, 0.62867641,  -1.10053349,
      -1.80746484, 0.62867594,  -0.59329236, -2.54207492, 0.46526238,
      -0.00045545, 0.49932924,  1.63340366,  -0.45734894, -0.00169915,
      1.59454787,  -0.00065268, -0.00251516, 1.81522143,  0.45606264,
      -0.00262397, 1.59454787,  0.45606312,  -0.00262326, 1.59454739,
      0.45606264,  -0.00262397, 1.59454787,  -0.00065267, -0.00251517,
      1.81522143,  -0.00083422, -0.50377685, 1.63340366,  1.09796488,
      -1.80969071, 0.62867606,  0.45606279,  -0.00262419, 1.59454775,
      -0.45734891, -0.00169916, 1.59454787,  -0.45734927, -0.00169968,
      1.59454751,  -1.56936228, 1.35052788,  0.52379429,  -0.45734900,
      -0.00169913, 1.59454787,  1.56918728,  1.34735024,  0.52379429,
      1.56811047,  -1.35147583, 0.52380723,  1.56808507,  -1.35131860,
      0.52387440,  -1.57038140, -1.34824181, 0.52379429,  -1.56936240,
      1.35052788,  0.52379423,  -1.44328797, 0.00054518,  0.88277382,
      -1.44327283, 0.00043958,  0.88280576,  1.44195509,  -0.00248179,
      0.88287276,  1.44223559,  -0.00054497, 0.88228679,  1.56918740,
      1.34735012,  0.52379423,  -1.57040441, -1.34818101, 0.52378970,
      -1.57038152, -1.34824169, 0.52379423,  0.00060985,  3.12254715,
      -0.12592883, 0.00060586,  3.12254739,  -0.12593007, -0.21464077,
      3.10757208,  -0.15445986, -0.42353559, 3.05566168,  -0.13569236,
      -0.62559998, 2.98231292,  -0.08878353, 0.62670720,  2.98100901,
      -0.08873826, 0.62661505,  2.98107791,  -0.08880345, 0.42469645,
      3.05480289,  -0.13569044, 0.21584108,  3.10713625,  -0.15445758,
      -0.00174801, -3.12165236, -0.12629783, 0.21322219,  -3.10669637,
      -0.15490441, 0.42281535,  -3.05461216, -0.13599920, 0.62440765,
      -2.98143482, -0.08901373, 0.62447023,  -2.98138809, -0.08896933,
      -0.62786478, -2.98014331, -0.08899876, -0.62786037, -2.98014665,
      -0.08900189, -0.42584255, -3.05390787, -0.13610038, -0.21698712,
      -3.10624146, -0.15494294, -0.00175192, -3.12165260, -0.12629905,
      -0.62560898, 2.98230958,  -0.08879187, -0.78726906, 2.86185288,
      -0.16757384, -0.93646783, 2.72652268,  -0.21237959, -1.07508647,
      2.58084154,  -0.23096853, -1.20526373, 2.42826319,  -0.23021236,
      -1.32975245, 2.27157402,  -0.21704006, -1.45062196, 2.05095077,
      -0.12514453, -1.45065486, 2.05087924,  -0.12510653, 1.45106840,
      2.04791498,  -0.12508209, 1.45102346,  2.04801273,  -0.12513401,
      1.33032060,  2.26888084,  -0.21701908, 1.20595014,  2.42582202,
      -0.23018987, 1.07588828,  2.57866383,  -0.23094597, 0.93737990,
      2.72462535,  -0.21235918, 0.78851759,  2.86004448,  -0.16762893,
      0.78625345,  -2.86083961, -0.16809504, 0.93532181,  -2.72562766,
      -0.21302269, 1.07394040,  -2.57994676, -0.23167852, 1.20411777,
      -2.42736840, -0.23091961, 1.32860637,  -2.27067924, -0.21769993,
      1.44944537,  -2.05011153, -0.12549706, 1.56815219,  -1.35151458,
      0.52370584,  -1.45212758, -2.04719448, -0.12551625, -1.33146667,
      -2.26798582, -0.21772107, -1.20709634, -2.42492723, -0.23094229,
      -1.07703435, -2.57776904, -0.23170128, -0.93852592, -2.72373033,
      -0.21304333, -0.78942937, -2.85936260, -0.16807115, -1.80910003,
      1.01058388,  -0.03365797, -1.84168148, 0.80862963,  -0.10271203,
      -1.86654651, 0.60610521,  -0.15332751, -1.88623869, 0.40378988,
      -0.19160835, -1.90002000, 0.20241861,  -0.21582428, -1.90505385,
      0.00237573,  -0.21922338, -1.90501344, 0.00071962,  -0.21905561,
      -1.90017104, -0.19767725, -0.21568473, -1.88654184, -0.39907673,
      -0.19146930, -1.86700249, -0.60143071, -0.15318951, -1.84229040,
      -0.80400622, -0.10257494, -1.80987811, -1.00592279, -0.03355779,
      1.90388918,  -0.00071938, -0.21917287, 1.89902496,  0.19857220,
      -0.21578738, 1.88539588,  0.39997119,  -0.19157638, 1.86585641,
      0.60232592,  -0.15330335, 1.84114432,  0.80490118,  -0.10269797,
      1.80871546,  1.00692093,  -0.03365800, 1.80794466,  -1.00973761,
      -0.03369782, 1.80795383,  -1.00968957, -0.03371784, 1.84053540,
      -0.80773503, -0.10275956, 1.86540043,  -0.60521013, -0.15336604,
      1.88509274,  -0.40289518, -0.19163992, 1.89881432,  -0.20239551,
      -0.21574667, -1.54677641, 1.84270406,  -0.22365679, -1.63208210,
      1.62775970,  -0.25862685, -1.70706439, 1.40805590,  -0.23586856,
      -1.77522779, 1.18613660,  -0.17659405, -1.80909991, 1.01058459,
      -0.03365869, 1.80871534,  1.00692177,  -0.03365877, 1.77497590,
      1.18254220,  -0.17653592, 1.70699000,  1.40456688,  -0.23577766,
      1.63230419,  1.62443995,  -0.25918254, 1.54714429,  1.83953130,
      -0.22410297, -1.80986142, -1.00602686, -0.03365880, -1.77612185,
      -1.18164730, -0.17739406, -1.70812643, -1.40370321, -0.23700003,
      -1.63330925, -1.62356102, -0.25988522, -1.54816461, -1.83868098,
      -0.22471850, -1.45216942, -2.04711795, -0.12558441, 1.44947588,
      -2.05005574, -0.12554680, 1.54517174,  -1.84302914, -0.22456542,
      1.63157570,  -1.62538552, -0.26028806, 1.70592797,  -1.40712976,
      -0.23690872, 1.77408183,  -1.18524182, -0.17733525, 0.00060586,
      3.12254739,  -0.46959737, -0.21464077, 3.10757208,  -0.45468479,
      0.00060591,  3.12254739,  -0.46959740, 0.21584108,  3.10713625,
      -0.45459676, -0.00175192, -3.12165260, -0.46959740, 0.21322219,
      -3.10669637, -0.45459723, -0.00175193, -3.12165260, -0.46959740,
      -0.21698712, -3.10624146, -0.45467842, -0.42353559, 3.05566168,
      -0.46107247, 0.42469645,  3.05480289,  -0.46099803, 0.42281535,
      -3.05461216, -0.46102279, -0.42584255, -3.05390787, -0.46106711,
      -0.62560898, 2.98230958,  -0.48016420, 0.62440765,  -2.98143482,
      -0.48015898, 0.62661505,  2.98107791,  -0.48015478, -0.62786037,
      -2.98014665, -0.48016420, -0.78726906, 2.86185288,  -0.45703074,
      0.62445766,  -2.98139739, -0.48017445, 0.78625345,  -2.86083961,
      -0.45658660, 0.78851759,  2.86004448,  -0.45661709, 0.62670475,
      2.98101068,  -0.48018256, -0.78942937, -2.85936260, -0.45704728,
      -0.62786043, -2.98014665, -0.48016423, -0.93646783, 2.72652268,
      -0.44622171, 0.93532181,  -2.72562766, -0.44563186, -0.93852592,
      -2.72373033, -0.44624415, 0.93737990,  2.72462535,  -0.44568357,
      -1.07508647, 2.58084154,  -0.44488293, 1.07394040,  -2.57994676,
      -0.44435173, -1.07703435, -2.57776904, -0.44490317, 1.07588828,
      2.57866383,  -0.44439828, -1.20526373, 2.42826319,  -0.45050618,
      1.20411777,  -2.42736840, -0.45019045, -1.20709634, -2.42492723,
      -0.45051819, 1.20595014,  2.42582202,  -0.45021811, -1.32975245,
      2.27157426,  -0.46058667, -1.32975245, 2.27157402,  -0.46058667,
      1.32860637,  -2.27067924, -0.46058667, -1.33146667, -2.26798582,
      -0.46058667, -1.33146656, -2.26798606, -0.46058667, 1.33032060,
      2.26888084,  -0.46058667, 1.33032048,  2.26888084,  -0.46058667,
      -1.45062196, 2.05095077,  -0.45238319, 1.32860649,  -2.27067924,
      -0.46058670, 1.44947588,  -2.05005574, -0.45129201, -1.45216942,
      -2.04711795, -0.45243445, 1.45102346,  2.04801273,  -0.45137730,
      -1.54677641, 1.84270406,  -0.45860177, 1.54517174,  -1.84302914,
      -0.45671692, -1.54816461, -1.83868098, -0.45867091, 1.54714429,
      1.83953130,  -0.45686778, -1.63208210, 1.62775970,  -0.47587365,
      -1.63330925, -1.62356102, -0.47593805, 1.63157570,  -1.62538552,
      -0.47375324, 1.63230419,  1.62443995,  -0.47368667, -1.70706439,
      1.40805590,  -0.50303531, -1.70812643, -1.40370321, -0.50307423,
      1.70592797,  -1.40712976, -0.50065041, 1.70699000,  1.40456688,
      -0.50072700, -1.77522779, 1.18613660,  -0.53636789, -1.77612185,
      -1.18164730, -0.53636789, 1.77308404,  -1.18848991, -0.53344285,
      1.77408183,  -1.18524182, -0.53272378, 1.77497590,  1.18254220,
      -0.53206521, 1.77341759,  1.18763101,  -0.53318721, -1.80909991,
      1.01058459,  -0.51588774, -1.80986142, -1.00602686, -0.51592720,
      -1.77612185, -1.18164718, -0.53636789, 1.80795383,  -1.00968957,
      -0.51267529, 1.80871534,  1.00692177,  -0.51214570, -1.84168148,
      0.80862963,  -0.49827772, -1.84229040, -0.80400622, -0.49834538,
      1.84053540,  -0.80773503, -0.49560285, 1.84114432,  0.80490118,
      -0.49522084, -1.86654651, 0.60610521,  -0.48789516, -1.86700249,
      -0.60143071, -0.48796996, 1.86540043,  -0.60521013, -0.48580948,
      1.86585641,  0.60232592,  -0.48557436, -1.88623869, 0.40378988,
      -0.48232013, -1.88654184, -0.39907673, -0.48238823, 1.88509274,
      -0.40289518, -0.48085603, 1.88539588,  0.39997119,  -0.48076698,
      -1.90002000, 0.20241861,  -0.48219424, -1.90017104, -0.19767725,
      -0.48223990, 1.89881432,  -0.20239551, -0.48138389, 1.89902496,
      0.19857220,  -0.48143867, 1.89892781,  0.20000805,  -0.48143691,
      -1.90505385, 0.00237582,  -0.49013662, -1.90505385, 0.00237573,
      -0.49013659, 1.90388918,  -0.00071975, -0.21917289, 1.90388918,
      -0.00071975, -0.49010351, 1.90388906,  -0.00072469, -0.49010390,
      1.90128529,  0.00611682,  -0.49218670, 1.40738547,  1.02835560,
      -0.88730782, 1.40482545,  -1.03119016, -0.88952076, 1.40470743,
      -1.02958739, -0.88973862, -1.41726339, 1.01772320,  -0.88060957,
      -1.90505385, 0.00237566,  -0.49013662, -1.41691077, -1.01434207,
      -0.88060957, 1.32860637,  -2.27067900, -0.46058673, 1.40398669,
      -1.03246403, -0.88968676, 1.40373409,  1.02831805,  -0.89023006,
      -1.32975245, 2.27157402,  -0.46058670, -1.41726327, 1.01772332,
      -0.88060963, -1.41726327, 1.01772320,  -0.88060963, -1.41691077,
      -1.01434207, -0.88060963, -1.14769018, 0.00177076,  -1.09618223,
      1.14696825,  -0.00000003, -1.09586132, 1.40373087,  1.02831531,
      -0.89023262, 0.40889478,  -2.28727150, -0.89325231, 0.40938482,
      2.28683925,  -0.89325225, -0.62560904, 2.98230958,  -0.48016423,
      -0.39072043, 2.28764248,  -0.89325225, -0.39121491, -2.28645349,
      -0.89325231, 0.40889484,  -2.28727150, -0.89325231, 0.75434840,
      -1.30917358, -1.09577036, 0.40938476,  2.28683901,  -0.89325231,
      -0.39072037, 2.28764224,  -0.89325231, -0.00175192, -3.12165236,
      -0.46959746, 0.40889472,  -2.28727150, -0.89325231, 0.00060591,
      3.12254691,  -0.46959764, -0.39072025, 2.28764224,  -0.89325237,
      0.40938473,  2.28683901,  -0.89325237, 0.01344044,  1.88713050,
      -1.09618223, 0.00032271,  2.45551205,  0.70305431};
  mesh.triVerts = {
      0,   1,   2,   0,   2,   3,   0,   3,   4,   0,   4,   5,   0,   5,   6,
      0,   6,   7,   0,   7,   8,   0,   8,   9,   9,   10,  0,   11,  12,  13,
      13,  14,  11,  15,  16,  17,  15,  17,  18,  18,  19,  15,  20,  15,  21,
      21,  22,  20,  23,  24,  25,  23,  25,  26,  26,  27,  23,  14,  13,  28,
      14,  28,  29,  29,  30,  14,  19,  18,  31,  22,  21,  32,  22,  32,  33,
      33,  34,  22,  35,  27,  36,  36,  37,  35,  12,  37,  38,  12,  38,  39,
      12,  39,  28,  40,  41,  42,  40,  42,  19,  40,  19,  43,  43,  44,  40,
      42,  41,  45,  42,  45,  46,  42,  46,  32,  32,  21,  42,  47,  48,  27,
      27,  26,  47,  28,  39,  49,  49,  29,  28,  43,  31,  50,  43,  50,  51,
      51,  44,  43,  32,  46,  52,  52,  33,  32,  48,  47,  53,  48,  53,  54,
      54,  55,  48,  44,  51,  56,  56,  40,  44,  38,  56,  57,  38,  57,  58,
      38,  58,  49,  49,  39,  38,  45,  55,  59,  45,  59,  60,  45,  60,  52,
      52,  46,  45,  24,  11,  61,  24,  61,  62,  24,  62,  63,  24,  63,  64,
      24,  64,  65,  65,  25,  24,  61,  11,  14,  61,  14,  30,  61,  30,  66,
      61,  66,  67,  61,  67,  68,  68,  69,  61,  17,  20,  70,  17,  70,  71,
      17,  71,  72,  17,  72,  73,  73,  74,  17,  70,  20,  22,  70,  22,  34,
      70,  34,  75,  70,  75,  76,  70,  76,  77,  70,  77,  78,  78,  79,  70,
      25,  65,  80,  25,  80,  81,  25,  81,  82,  25,  82,  83,  25,  83,  84,
      25,  84,  85,  25,  85,  86,  25,  86,  87,  25,  87,  53,  25,  53,  47,
      47,  26,  25,  66,  30,  29,  66,  29,  49,  66,  49,  58,  66,  58,  88,
      66,  88,  89,  66,  89,  90,  66,  90,  91,  66,  91,  92,  66,  92,  93,
      93,  94,  66,  17,  74,  95,  17,  95,  96,  17,  96,  97,  17,  97,  98,
      17,  98,  99,  17,  99,  100, 17,  100, 101, 17,  101, 50,  17,  50,  31,
      52,  60,  102, 52,  102, 103, 52,  103, 104, 52,  104, 105, 52,  105, 106,
      52,  106, 107, 52,  107, 75,  52,  75,  34,  34,  33,  52,  54,  53,  108,
      54,  108, 109, 54,  109, 110, 54,  110, 111, 54,  111, 112, 54,  112, 113,
      113, 114, 54,  54,  114, 115, 54,  115, 116, 54,  116, 117, 54,  117, 118,
      54,  118, 119, 54,  119, 59,  59,  55,  54,  58,  57,  120, 58,  120, 121,
      58,  121, 122, 58,  122, 123, 58,  123, 124, 124, 125, 58,  120, 57,  56,
      120, 56,  51,  120, 51,  50,  120, 50,  101, 120, 101, 126, 120, 126, 127,
      120, 127, 128, 120, 128, 129, 120, 129, 130, 120, 130, 131, 53,  87,  132,
      53,  132, 133, 53,  133, 134, 53,  134, 135, 53,  135, 136, 136, 108, 53,
      58,  125, 137, 58,  137, 138, 58,  138, 139, 58,  139, 140, 58,  140, 141,
      141, 88,  58,  59,  119, 142, 59,  142, 143, 59,  143, 144, 59,  144, 145,
      59,  145, 146, 59,  146, 147, 59,  147, 102, 102, 60,  59,  101, 100, 148,
      101, 148, 149, 101, 149, 150, 101, 150, 151, 101, 151, 152, 152, 126, 101,
      62,  153, 154, 154, 63,  62,  155, 62,  61,  155, 61,  69,  69,  156, 155,
      79,  157, 158, 79,  158, 71,  71,  70,  79,  159, 79,  78,  78,  160, 159,
      63,  154, 161, 161, 64,  63,  69,  68,  162, 162, 156, 69,  71,  158, 163,
      163, 72,  71,  78,  77,  164, 164, 160, 78,  64,  161, 165, 64,  165, 80,
      80,  65,  64,  72,  163, 166, 166, 73,  72,  68,  67,  167, 167, 162, 68,
      77,  76,  168, 168, 164, 77,  80,  165, 169, 169, 81,  80,  73,  166, 170,
      73,  170, 171, 73,  171, 95,  95,  74,  73,  66,  94,  172, 66,  172, 173,
      66,  173, 167, 167, 67,  66,  75,  107, 174, 75,  174, 175, 175, 76,  75,
      81,  169, 176, 176, 82,  81,  95,  171, 177, 177, 96,  95,  107, 106, 178,
      178, 174, 107, 94,  93,  179, 179, 172, 94,  82,  176, 180, 180, 83,  82,
      96,  177, 181, 181, 97,  96,  106, 105, 182, 182, 178, 106, 93,  92,  183,
      183, 179, 93,  83,  180, 184, 184, 84,  83,  97,  181, 185, 185, 98,  97,
      105, 104, 186, 186, 182, 105, 92,  91,  187, 187, 183, 92,  84,  184, 188,
      189, 85,  84,  98,  185, 190, 190, 99,  98,  104, 103, 191, 192, 186, 104,
      91,  90,  193, 194, 187, 91,  85,  189, 195, 195, 86,  85,  99,  196, 197,
      99,  197, 148, 148, 100, 99,  102, 147, 198, 102, 198, 191, 191, 103, 102,
      90,  89,  199, 199, 193, 90,  86,  195, 200, 86,  200, 132, 132, 87,  86,
      148, 197, 201, 201, 149, 148, 147, 146, 202, 202, 198, 147, 88,  141, 203,
      88,  203, 199, 199, 89,  88,  132, 200, 204, 204, 133, 132, 146, 145, 205,
      205, 202, 146, 149, 201, 206, 206, 150, 149, 141, 140, 207, 207, 203, 141,
      133, 204, 208, 208, 134, 133, 145, 144, 209, 209, 205, 145, 150, 206, 210,
      210, 151, 150, 140, 139, 211, 211, 207, 140, 134, 208, 212, 212, 135, 134,
      144, 143, 213, 213, 209, 144, 151, 210, 214, 151, 214, 215, 215, 152, 151,
      139, 138, 216, 139, 216, 217, 217, 211, 139, 135, 212, 218, 218, 136, 135,
      143, 142, 219, 143, 219, 220, 152, 215, 221, 152, 221, 127, 127, 126, 152,
      138, 137, 222, 222, 216, 138, 136, 218, 223, 136, 223, 109, 109, 108, 136,
      119, 118, 224, 119, 224, 219, 219, 142, 119, 127, 221, 225, 225, 128, 127,
      125, 124, 226, 125, 226, 222, 222, 137, 125, 109, 223, 227, 227, 110, 109,
      118, 117, 228, 228, 224, 118, 128, 225, 229, 229, 129, 128, 124, 123, 230,
      230, 226, 124, 110, 227, 231, 231, 111, 110, 117, 116, 232, 232, 228, 117,
      129, 229, 233, 233, 130, 129, 123, 122, 234, 234, 230, 123, 111, 231, 235,
      235, 112, 111, 116, 115, 236, 236, 232, 116, 130, 233, 237, 237, 131, 130,
      122, 121, 238, 122, 238, 239, 239, 234, 122, 112, 235, 240, 240, 113, 112,
      113, 241, 236, 113, 236, 115, 115, 114, 113, 238, 121, 120, 238, 120, 242,
      242, 243, 238, 243, 242, 131, 243, 131, 237, 243, 237, 244, 243, 245, 239,
      239, 238, 243, 239, 245, 246, 239, 246, 217, 239, 217, 216, 239, 216, 222,
      239, 222, 226, 239, 226, 230, 230, 234, 239, 214, 247, 248, 214, 248, 244,
      214, 244, 237, 214, 237, 233, 214, 233, 229, 214, 229, 225, 214, 225, 221,
      221, 215, 214, 212, 249, 240, 212, 240, 235, 212, 235, 231, 212, 231, 227,
      212, 227, 223, 223, 218, 212, 236, 250, 251, 236, 251, 220, 236, 220, 219,
      236, 219, 224, 236, 224, 228, 228, 232, 236, 197, 252, 253, 197, 253, 247,
      197, 247, 214, 197, 214, 210, 197, 210, 206, 206, 201, 197, 217, 246, 254,
      217, 254, 194, 217, 194, 193, 217, 193, 199, 217, 199, 203, 217, 203, 207,
      207, 211, 217, 195, 255, 256, 195, 257, 249, 195, 249, 212, 195, 212, 208,
      195, 208, 204, 204, 200, 195, 220, 258, 192, 220, 191, 198, 220, 198, 202,
      220, 202, 205, 220, 205, 209, 259, 251, 250, 259, 250, 240, 259, 240, 249,
      249, 257, 259, 244, 248, 260, 244, 260, 261, 244, 261, 254, 244, 254, 246,
      244, 246, 245, 171, 170, 262, 171, 262, 252, 171, 190, 185, 171, 185, 181,
      181, 177, 171, 187, 194, 263, 187, 263, 173, 187, 173, 172, 187, 172, 179,
      179, 183, 187, 169, 264, 265, 169, 265, 255, 169, 255, 188, 169, 188, 184,
      169, 184, 180, 180, 176, 169, 192, 266, 175, 192, 175, 174, 192, 174, 178,
      192, 178, 182, 182, 186, 192, 252, 267, 268, 268, 253, 252, 254, 261, 8,
      254, 8,   7,   254, 7,   269, 254, 269, 263, 263, 194, 254, 265, 270, 5,
      265, 4,   256, 256, 255, 265, 192, 258, 2,   192, 2,   1,   1,   266, 192,
      158, 271, 272, 158, 272, 170, 158, 170, 166, 166, 163, 158, 263, 269, 273,
      263, 273, 155, 263, 155, 156, 263, 156, 162, 263, 162, 167, 167, 173, 263,
      271, 160, 164, 271, 164, 168, 168, 266, 271, 153, 273, 274, 153, 274, 270,
      153, 270, 265, 153, 265, 264, 153, 264, 161, 161, 154, 153, 271, 266, 0,
      0,   272, 271, 274, 273, 275, 275, 276, 274, 260, 9,   8,   8,   261, 260,
      9,   260, 248, 9,   248, 247, 9,   247, 253, 9,   253, 268, 268, 10,  9,
      256, 4,   259, 3,   2,   258, 272, 0,   10,  272, 10,  268, 268, 267, 272,
      7,   276, 275, 274, 6,   5,   5,   270, 274, 0,   266, 1,   277, 24,  23};
  Manifold manifold(mesh.Merge(1e-5));
  manifold = manifold.RefineToLength(1.0);
  EXPECT_NEAR(manifold.Volume(), 31.21, 0.01);
}

// Deeply-nested CsgOpNode chain (e.g. repeated `+=` in a loop) must not
// stack-overflow in the leaf-counting pre-pass. Cancel up front so we only
// exercise NumLeaves, not the full boolean evaluation.
//
// 300k depth SIGSEGVs on the recursive NumLeaves (8MB default stack on
// macOS/Linux); runs in ~1s / ~1.2GB peak RSS on the iterative version.
TEST(Manifold, DeepChainDoesNotOverflowNumLeaves) {
  constexpr int kDepth = 300000;
  Manifold m = Manifold::Cube(vec3(1), true);
  for (int i = 0; i < kDepth; ++i) {
    m = m + Manifold::Cube(vec3(1), true).Translate(vec3(i * 2.0, 0, 0));
  }
  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(m.WithContext(ctx).Status(), Manifold::Error::Cancelled);
  // kDepth + 1 leaves in the chain → kDepth booleans to reduce.
  auto& privateCtx = *ctx.impl_;
  EXPECT_EQ(privateCtx.totalBooleans.load(), kDepth);
}

#ifndef MANIFOLD_NO_FILESYSTEM
// DedupeEdges collects every duplicated edge in one pass and then repairs
// them in turn. An earlier repair can resolve a later entry; repairing that
// stale entry relabelled an orbit to a copy of the wrong vertex, moving
// triangle corners and removing solid while reporting NoError. Splitting
// duplicate edges only relabels vertices and adds zero-area triangles, so the
// volume must not change.
//
// The fixture is cut from a 28060-triangle union around its duplicate edges,
// with each cut boundary closed by a fan. Its triangle order sets how the
// 4-way edges pair up on load, which is what reproduces the stale entry.
TEST(Manifold, DedupeEdgesNeverMovesACorner) {
  Manifold m = ReadTestOBJ("dedupe_stale_duplicate.obj");
  EXPECT_EQ(m.Status(), Manifold::Error::NoError);
  // Signed volume of the fixture's triangles, computed exactly.
  EXPECT_NEAR(m.Volume(), 0.00099963253073879, 1e-12);
}
#endif

// A coarse cylinder with three verts pulled to just over epsilon from a
// neighbor, at relative precision 0.1. RemoveDegenerates adds two verts in
// FormLoop and later splits a quad on an edge that ends at a newer vert.
TEST(Manifold, RemoveDegeneratesSplitsAtNewVert) {
  MeshGL64 mesh;
  mesh.vertProperties = {
      -0.47488555173337099,   -0.39698845421345746, -0.34188284539605285,
      -0.38692187123277771,   -0.45770248247545098, -0.37523313166288419,
      -0.55159161949407864,   -0.33743035459550019, -0.2861598174123241,
      -0.61319371198073336,   -0.28201467542938935, -0.21085823301390325,
      -0.65660284304811789,   -0.23352019026423823, -0.11975402807456051,
      -0.67964229760376771,   -0.19437861508156265, -0.017415547847984506,
      -0.29211144423694035,   -0.51652798448472081, -0.3845383536007313,
      -0.19520846144929604,   -0.57051520393661737, -0.36933190852640763,
      -0.10107204305843429,   -0.61695699621682676, -0.33037631109252441,
      -0.67592823673392022,   -0.14577450954880902, 0.10457770625687557,
      -0.66107035128140956,   -0.15143766970761169, 0.20013151388574996,
      -0.62039022534169885,   -0.14979153782315879, 0.30443137732015946,
      -0.5611562747527763,    -0.16169682017498199, 0.39869508857131186,
      -0.39968927624009776,   -0.22312411591338729, 0.53894722501979364,
      -0.48633873713660769,   -0.18655653619383122, 0.478195871546095,
      -0.11383944806575433,   -0.38237862965476316, 0.58380404559015331,
      -0.30555285784923625,   -0.2695659081935966,  0.57790282245367675,
      -0.20864987506159172,   -0.32355312764549332, 0.59310926752800042,
      -0.01442258216192438,   -0.6535245759363828,  -0.26962495761882588,
      0.060394955454244337,   -0.67838429195523209, -0.1901241746440426,
      0.11962890604316684,    -0.69028957430705529, -0.095860463392890283,
      0.16030903198287744,    -0.68864344242260245, 0.0084394000415189824,
      0.18039546189334638,    -0.6735284402282079,  0.11754538498828312,
      0.17888097830523564,    -0.64570249704865157, 0.22598646177525344,
      0.17491014944264449,    -0.61422964253813117, 0.28806942712562233,
      -0.025875767565160995,  -0.44309265791675667, 0.55045375932332208,
      0.050830300195546574,   -0.50265075753471389, 0.49473073133959322,
      0.10738341598184817,    -0.58054108972714324, 0.45367172743023804,
      0.032692828250385608,   0.50285608444653762,  -0.29337167095015726,
      0.006647349609954889,   0.53195277554562836,  -0.23870914798655166,
      -0.0071763231234349689, 0.55543772065523367,  -0.17730605985060605,
      0.16845593269915904,    0.39744340021890068,  -0.39199661013954579,
      0.069654083742378525,   0.46960667694687114,  -0.33855262158920979,
      0.11567772439880306,    0.4338718171760968,   -0.371986438379447,
      -0.0080850132763014004, 0.57213328656296758,  -0.11224141377842371,
      0.003966844669979902,   0.58120228787960426,  -0.046777822810365369,
      0.028374920233806333,   0.58218996701027592,  0.015802095250280326,
      0.06391529058715989,    0.57504679759918209,  0.072360322000971811,
      0.47238639906472574,    0.2578911451199381,   -0.22437300917754949,
      0.49679447462855209,    0.25887882425060982,  -0.16179309111690396,
      0.28348397856924801,    0.32975576734220086,  -0.38845587625765987,
      0.22534218889666144,    0.36214809901333883,  -0.3975797433022541,
      0.33996582960376509,    0.3018906919740752,   -0.36508251779732992,
      0.39195550614167096,    0.27995014414234154,  -0.32863170571311084,
      0.43684602871137224,    0.26503431453103199,  -0.28093123592824087,
      0.50884633257483347,    0.2679478255672465,   -0.096329500148845462,
      0.50793764242196704,    0.28464339147498041,  -0.031264854076663282,
      0.49411396968857718,    0.30812833658458572,  0.030138234059282398,
      0.46806849104814646,    0.33722502768367646,  0.084800757022888085,
      0.10880581315686108,    0.56013096798787254,  0.12006079178584164,
      0.16079548969476698,    0.53819042015613894,  0.15651160387006077,
      0.2754191304018706,     0.47793301311687525,  0.18900882937498489,
      0.21727734072928392,    0.51032534478801328,  0.17988496233039072,
      0.33230538659937303,    0.4426377119113134,   0.18342569621227664,
      0.4311072355561536,     0.37047443518334289,  0.12998170766194059,
      0.38508359489972904,    0.40620929495411728,  0.16341552445217786,
  };
  mesh.triVerts = {
      3,  20, 4,  4,  21, 5,  5,  22, 9,  1,  8,  0,  0,  18, 2,  6,  7,  1,
      2,  19, 3,  1,  7,  8,  0,  8,  18, 2,  18, 19, 3,  19, 20, 4,  20, 21,
      5,  21, 22, 33, 0,  2,  31, 1,  0,  41, 6,  1,  9,  23, 10, 9,  22, 23,
      10, 23, 24, 11, 24, 27, 12, 27, 26, 14, 26, 25, 10, 24, 11, 11, 27, 12,
      12, 26, 14, 14, 25, 13, 13, 15, 16, 16, 15, 17, 13, 25, 15, 28, 3,  4,
      29, 4,  5,  30, 5,  9,  34, 9,  10, 32, 2,  3,  28, 32, 3,  32, 33, 2,
      29, 28, 4,  30, 29, 5,  34, 30, 9,  35, 34, 10, 36, 35, 11, 35, 10, 11,
      36, 11, 12, 37, 12, 14, 49, 14, 13, 50, 13, 16, 52, 16, 17, 37, 36, 12,
      49, 37, 14, 40, 7,  6,  42, 8,  7,  43, 18, 8,  44, 19, 18, 39, 21, 20,
      38, 20, 19, 38, 39, 20, 45, 22, 21, 51, 17, 15, 54, 26, 27, 55, 25, 26,
      53, 15, 25, 47, 24, 23, 46, 23, 22, 48, 27, 24, 31, 41, 1,  33, 31, 0,
      41, 40, 6,  40, 42, 7,  42, 43, 8,  31, 33, 42, 33, 32, 43, 32, 28, 44,
      28, 29, 38, 29, 30, 39, 30, 34, 45, 34, 35, 46, 35, 36, 47, 36, 37, 48,
      43, 44, 18, 44, 38, 19, 39, 45, 21, 45, 46, 22, 46, 47, 23, 31, 42, 40,
      33, 43, 42, 41, 31, 40, 32, 44, 43, 28, 38, 44, 29, 39, 38, 30, 45, 39,
      34, 46, 45, 35, 47, 46, 36, 48, 47, 37, 54, 48, 52, 50, 16, 51, 52, 17,
      53, 51, 15, 50, 49, 13, 37, 49, 54, 54, 55, 26, 48, 54, 27, 55, 53, 25,
      47, 48, 24, 49, 50, 55, 50, 52, 53, 52, 51, 53, 49, 55, 54, 50, 53, 55,
  };
  Manifold result;
  EXPECT_NO_THROW(result =
                      Manifold(mesh).WithPrecision(0.1).RemoveDegenerates());
  EXPECT_EQ(result.Status(), Manifold::Error::NoError);
  EXPECT_EQ(result.NumDegenerateTris(), 0);
}
