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

#include "manifold/math.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "gtest/gtest.h"

namespace {
constexpr double kRelativeTolerance =
    4 * std::numeric_limits<double>::epsilon();
}  // namespace

TEST(Math, AsinSmallAngles) {
  for (double magnitude :
       {std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::min(), 1e-300, 1e-20, 1e-12, 1e-8, 1e-4,
        0.1, 0.25, std::nextafter(0.5, 0.0)}) {
    for (double sign : {-1.0, 1.0}) {
      const double x = sign * magnitude;
      SCOPED_TRACE(x);
      const double expected = std::asin(x);
      EXPECT_NEAR(manifold::math::asin(x), expected,
                  kRelativeTolerance * std::abs(expected));
    }
  }
}

TEST(Math, AsinSignedZero) {
  EXPECT_EQ(manifold::math::asin(0.0), 0.0);
  EXPECT_FALSE(std::signbit(manifold::math::asin(0.0)));
  EXPECT_EQ(manifold::math::asin(-0.0), 0.0);
  EXPECT_TRUE(std::signbit(manifold::math::asin(-0.0)));
}

TEST(Math, AsinDomain) {
  EXPECT_EQ(manifold::math::asin(1.0), std::asin(1.0));
  EXPECT_EQ(manifold::math::asin(-1.0), std::asin(-1.0));
  for (double x : {std::nextafter(1.0, 2.0), std::nextafter(-1.0, -2.0),
                   std::numeric_limits<double>::infinity(),
                   -std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_TRUE(std::isnan(manifold::math::asin(x)));
  }
}

TEST(Math, AsinAcrossDomain) {
  double largestRelativeError = 0.0;
  for (int i = -8192; i <= 8192; ++i) {
    if (i == 0) continue;
    const double x = i / 8192.0;
    const double expected = std::asin(x);
    largestRelativeError =
        std::max(largestRelativeError,
                 std::abs((manifold::math::asin(x) - expected) / expected));
  }
  EXPECT_LE(largestRelativeError, kRelativeTolerance);
  for (double x : {std::nextafter(0.5, 0.0), 0.5, std::nextafter(0.5, 1.0),
                   std::nextafter(1.0, 0.0)}) {
    for (double sign : {-1.0, 1.0}) {
      const double expected = std::asin(sign * x);
      EXPECT_NEAR(manifold::math::asin(sign * x), expected,
                  kRelativeTolerance * std::abs(expected));
    }
  }
}

TEST(Math, AcosAcrossDomain) {
  double largestRelativeError = 0.0;
  for (int i = -8192; i < 8192; ++i) {
    const double x = i / 8192.0;
    const double expected = std::acos(x);
    largestRelativeError =
        std::max(largestRelativeError,
                 std::abs((manifold::math::acos(x) - expected) / expected));
  }
  EXPECT_LE(largestRelativeError, kRelativeTolerance);
  EXPECT_EQ(manifold::math::acos(1.0), 0.0);
  EXPECT_TRUE(std::isnan(manifold::math::acos(2.0)));
}
