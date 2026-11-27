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

#include <limits>

#include "gtest/gtest.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"

using namespace manifold;

TEST(Mirror, ManifoldNormalMagnitude) {
  const Manifold input = Manifold::Cube({1, 2, 3}).Translate({2, 3, 4});
  for (const vec3 normal : {vec3(1, 0, 0), vec3(1, 2, 3), vec3(-3, 2, -1)}) {
    const Box expected = input.Mirror(normal).BoundingBox();
    for (const double scale :
         {std::numeric_limits<double>::denorm_min(),
          std::numeric_limits<double>::min(), 1e-200, 1e-160, 1.0, 1e160, 1e200,
          std::numeric_limits<double>::max() / 4}) {
      SCOPED_TRACE(scale);
      const Manifold mirrored = input.Mirror(normal * scale);
      ASSERT_EQ(mirrored.Status(), Manifold::Error::NoError);
      EXPECT_FALSE(mirrored.IsEmpty());
      EXPECT_NEAR(mirrored.Volume(), 6, 1e-12);
      EXPECT_NEAR(mirrored.SurfaceArea(), 22, 1e-12);
      const Box bounds = mirrored.BoundingBox();
      for (const int axis : {0, 1, 2}) {
        EXPECT_NEAR(bounds.min[axis], expected.min[axis], 1e-12);
        EXPECT_NEAR(bounds.max[axis], expected.max[axis], 1e-12);
      }
    }
  }
}

TEST(Mirror, CrossSectionNormalMagnitude) {
  const CrossSection input = CrossSection::Square({1, 2}).Translate({3, 4});
  for (const vec2 normal : {vec2(1, 0), vec2(1, 2), vec2(-2, 1)}) {
    const Rect expected = input.Mirror(normal).Bounds();
    for (const double scale :
         {std::numeric_limits<double>::denorm_min(),
          std::numeric_limits<double>::min(), 1e-200, 1e-160, 1.0, 1e160, 1e200,
          std::numeric_limits<double>::max() / 4}) {
      SCOPED_TRACE(scale);
      const CrossSection mirrored = input.Mirror(normal * scale);
      EXPECT_FALSE(mirrored.IsEmpty());
      EXPECT_NEAR(mirrored.Area(), 2, 1e-12);
      const Rect bounds = mirrored.Bounds();
      for (const int axis : {0, 1}) {
        EXPECT_NEAR(bounds.min[axis], expected.min[axis], 1e-12);
        EXPECT_NEAR(bounds.max[axis], expected.max[axis], 1e-12);
      }
    }
  }
}

TEST(Mirror, NormalSignAndRepeatedReflection) {
  const Manifold box = Manifold::Cube({1, 2, 3}).Translate({2, 3, 4});
  const CrossSection section = CrossSection::Square({1, 2}).Translate({3, 4});
  for (const double scale : {1e-200, 1.0, 1e200}) {
    const vec3 normal3 = vec3(1, -2, 3) * scale;
    const Box positive3 = box.Mirror(normal3).BoundingBox();
    const Box negative3 = box.Mirror(-normal3).BoundingBox();
    const Manifold twice3 = box.Mirror(normal3).Mirror(normal3);
    EXPECT_NEAR(twice3.Volume(), 6, 1e-12);
    for (const int axis : {0, 1, 2}) {
      EXPECT_DOUBLE_EQ(positive3.min[axis], negative3.min[axis]);
      EXPECT_DOUBLE_EQ(positive3.max[axis], negative3.max[axis]);
      EXPECT_NEAR(twice3.BoundingBox().min[axis], box.BoundingBox().min[axis],
                  1e-12);
      EXPECT_NEAR(twice3.BoundingBox().max[axis], box.BoundingBox().max[axis],
                  1e-12);
    }
    const vec2 normal2 = vec2(-2, 1) * scale;
    const Rect positive2 = section.Mirror(normal2).Bounds();
    const Rect negative2 = section.Mirror(-normal2).Bounds();
    const CrossSection twice2 = section.Mirror(normal2).Mirror(normal2);
    EXPECT_NEAR(twice2.Area(), 2, 1e-12);
    for (const int axis : {0, 1}) {
      EXPECT_DOUBLE_EQ(positive2.min[axis], negative2.min[axis]);
      EXPECT_DOUBLE_EQ(positive2.max[axis], negative2.max[axis]);
      EXPECT_NEAR(twice2.Bounds().min[axis], section.Bounds().min[axis], 1e-12);
      EXPECT_NEAR(twice2.Bounds().max[axis], section.Bounds().max[axis], 1e-12);
    }
  }
}

TEST(Mirror, ZeroNormal) {
  const Manifold box = Manifold::Cube().WithPrecision(1e-4);
  for (const double zero : {0.0, -0.0}) {
    const Manifold empty = box.Mirror(vec3(zero));
    EXPECT_EQ(empty.Status(), Manifold::Error::NoError);
    EXPECT_TRUE(empty.IsEmpty());
    EXPECT_DOUBLE_EQ(empty.GetPrecision(), box.GetPrecision());
    EXPECT_TRUE(CrossSection::Square(vec2(1)).Mirror(vec2(zero)).IsEmpty());
  }
}

TEST(Mirror, NonFiniteNormal) {
  const Manifold box = Manifold::Cube();
  const CrossSection section = CrossSection::Square({1, 2}).Translate({3, 4});
  for (const double bad : {std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    for (const int axis : {0, 1, 2}) {
      vec3 normal(0.0);
      normal[axis] = bad;
      EXPECT_EQ(box.Mirror(normal).Status(), Manifold::Error::NonFiniteVertex);
    }
    for (const int axis : {0, 1}) {
      vec2 normal(0.0);
      normal[axis] = bad;
      const CrossSection mirrored = section.Mirror(normal);
      EXPECT_DOUBLE_EQ(mirrored.Area(), section.Area());
      EXPECT_EQ(mirrored.Bounds().min, section.Bounds().min);
      EXPECT_EQ(mirrored.Bounds().max, section.Bounds().max);
    }
  }
}

TEST(Mirror, InputError) {
  MeshGL64 mesh = Manifold::Cube().GetMeshGL64();
  mesh.vertProperties[0] = std::numeric_limits<double>::quiet_NaN();
  const Manifold invalid(mesh);
  ASSERT_EQ(invalid.Status(), Manifold::Error::NonFiniteVertex);
  for (const double scale : {0.0, 1e-200, 1.0, 1e200}) {
    EXPECT_EQ(invalid.Mirror({scale, 0, 0}).Status(),
              Manifold::Error::NonFiniteVertex);
  }
}

TEST(Mirror, DeferredEvaluation) {
  const Manifold cube = Manifold::Cube();
  for (const double scale : {1e-200, 1.0, 1e200}) {
    const Manifold pending =
        (cube + cube.Translate({0.5, 0, 0})).Mirror({scale, 0, 0});
    ExecutionContext ctx;
    ctx.Cancel();
    EXPECT_EQ(pending.WithContext(ctx).Status(), Manifold::Error::Cancelled);

    const Manifold evaluated =
        (cube + cube.Translate({0.5, 0, 0})).Mirror({scale, 0, 0});
    ASSERT_EQ(evaluated.Status(), Manifold::Error::NoError);
    EXPECT_DOUBLE_EQ(evaluated.Volume(), 1.5);
    EXPECT_EQ(evaluated.BoundingBox(), Box({-1.5, 0, 0}, {0, 1, 1}));
  }
}
