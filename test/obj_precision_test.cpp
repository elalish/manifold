// Copyright 2026 The Manifold Authors.
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

#ifndef MANIFOLD_NO_IOSTREAM

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "manifold/manifold.h"

using namespace manifold;

namespace {
uint64_t DoubleBits(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void ExpectExactVertices(const MeshGL64& original) {
  std::stringstream stream;
  ASSERT_TRUE(WriteOBJ(stream, original));
  const MeshGL64 restored = ReadOBJ(stream);
  ASSERT_EQ(restored.vertProperties.size(), original.vertProperties.size());
  for (size_t i = 0; i < original.vertProperties.size(); ++i) {
    SCOPED_TRACE(i);
    ASSERT_EQ(DoubleBits(restored.vertProperties[i]),
              DoubleBits(original.vertProperties[i]));
  }
}
}  // namespace

TEST(OBJPrecision, FiniteVerticesRoundTripExactly) {
  MeshGL64 mesh;
  mesh.vertProperties = {0.0,
                         -0.0,
                         1.0,
                         -1.0,
                         0.1,
                         -0.1,
                         std::nextafter(0.1, 1.0),
                         std::nextafter(1.0, 2.0),
                         std::nextafter(1.0, 0.0),
                         1e-10,
                         -1e-10,
                         std::nextafter(1e-10, 0.0),
                         1e-20,
                         -1e-20,
                         1e-100,
                         -1e100,
                         std::numeric_limits<double>::min(),
                         -std::numeric_limits<double>::min(),
                         std::numeric_limits<double>::denorm_min(),
                         -std::numeric_limits<double>::denorm_min(),
                         1e300,
                         -1e300,
                         std::numeric_limits<double>::max(),
                         -std::numeric_limits<double>::max()};
  ExpectExactVertices(mesh);
}

TEST(OBJPrecision, RandomFiniteVerticesRoundTripExactly) {
  MeshGL64 mesh;
  std::mt19937_64 random(0x4f424a);
  while (mesh.vertProperties.size() < 3000) {
    const uint64_t bits = random();
    if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000)) {
      continue;
    }
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    mesh.vertProperties.push_back(value);
  }
  ExpectExactVertices(mesh);
}

TEST(OBJPrecision, ReadScientificCoordinates) {
  std::istringstream stream("v 1e-20 -1E+20 2.2250738585072014e-308\n");
  const MeshGL64 mesh = ReadOBJ(stream);
  ASSERT_EQ(mesh.vertProperties.size(), 3);
  EXPECT_EQ(mesh.vertProperties[0], 1e-20);
  EXPECT_EQ(mesh.vertProperties[1], -1e20);
  EXPECT_EQ(mesh.vertProperties[2], std::numeric_limits<double>::min());
}

TEST(OBJPrecision, ReadHexadecimalCoordinates) {
  std::istringstream stream(
      "v 0x1.8p+0 -0x1.6p+1 0X1.CP-2\n"
      "v 0x0.0000000000001p-1022 0x1.fffffffffffffp+1023 -0x0p+0\n");
  const MeshGL64 mesh = ReadOBJ(stream);
  ASSERT_EQ(mesh.vertProperties.size(), 6);
  EXPECT_EQ(mesh.vertProperties[0], 1.5);
  EXPECT_EQ(mesh.vertProperties[1], -2.75);
  EXPECT_EQ(mesh.vertProperties[2], 0.4375);
  EXPECT_EQ(mesh.vertProperties[3], std::numeric_limits<double>::denorm_min());
  EXPECT_EQ(mesh.vertProperties[4], std::numeric_limits<double>::max());
  EXPECT_EQ(DoubleBits(mesh.vertProperties[5]), DoubleBits(-0.0));
}

TEST(OBJPrecision, ThinSolidRetainsVolume) {
  MeshGL64 mesh = Manifold::Cube().GetMeshGL64();
  for (size_t i = 0; i < mesh.vertProperties.size(); i += 3) {
    mesh.vertProperties[i] *= 1e-10;
    mesh.vertProperties[i + 1] *= 1e-10;
    mesh.vertProperties[i + 2] *= 1e-20;
  }
  const Manifold original(mesh);
  ASSERT_EQ(original.Status(), Manifold::Error::NoError);
  ASSERT_GT(original.Volume(), 0.0);
  ASSERT_GT(original.BoundingBox().Size().z, 1e-21);
  std::stringstream stream;
  ASSERT_TRUE(original.WriteOBJ(stream));
  const Manifold restored = Manifold::ReadOBJ(stream);
  EXPECT_EQ(restored.Status(), Manifold::Error::NoError);
  EXPECT_EQ(restored.NumTri(), original.NumTri());
  EXPECT_EQ(restored.BoundingBox().Size().z, original.BoundingBox().Size().z);
  EXPECT_NEAR(restored.Volume(), original.Volume(), original.Volume() * 1e-12);
}

TEST(OBJPrecision, OrdinarySolidRoundTrip) {
  const Manifold original =
      Manifold::Cube({2, 3, 4}).Translate({0.25, -0.5, 1.5});
  std::stringstream stream;
  ASSERT_TRUE(original.WriteOBJ(stream));
  const Manifold restored = Manifold::ReadOBJ(stream);
  EXPECT_EQ(restored.Status(), Manifold::Error::NoError);
  EXPECT_EQ(restored.NumTri(), original.NumTri());
  EXPECT_DOUBLE_EQ(restored.Volume(), original.Volume());
  EXPECT_EQ(restored.BoundingBox().min, original.BoundingBox().min);
  EXPECT_EQ(restored.BoundingBox().max, original.BoundingBox().max);
}

TEST(OBJPrecision, RetainOutputFormattingState) {
  for (const auto floatfield :
       {std::ios_base::fmtflags(0), std::ios_base::fixed,
        std::ios_base::scientific, std::ios_base::floatfield}) {
    std::ostringstream stream;
    stream.setf(floatfield, std::ios_base::floatfield);
    stream.precision(3);
    ASSERT_TRUE(Manifold::Cube().WriteOBJ(stream));
    const bool hex =
        stream.str().find("# float_format = hexfloat") != std::string::npos;
    EXPECT_EQ(stream.flags() & std::ios_base::floatfield,
              hex ? floatfield : std::ios_base::fixed);
    EXPECT_EQ(stream.precision(), 19);
  }
}

#endif
