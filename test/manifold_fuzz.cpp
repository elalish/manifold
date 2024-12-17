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

#include <atomic>
#include <future>

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"
#include "manifold/manifold.h"

using namespace fuzztest;

enum class TransformType { Translate, Rotate, Scale };
struct Transform {
  TransformType ty;
  std::array<double, 3> vector;
};
struct CubeOp {
  std::vector<Transform> transforms;
  bool isUnion;
};

// larger numbers may cause precision issues, prefer to test them later
auto GoodNumbers = OneOf(InRange(0.1, 10.0), InRange(-10.0, -0.1));
auto Vec3Domain = ArrayOf<3>(GoodNumbers);
auto TransformDomain = StructOf<Transform>(
    ElementOf({TransformType::Translate, TransformType::Rotate,
               TransformType::Scale}),
    Vec3Domain);
auto CsgDomain =
    VectorOf(StructOf<CubeOp>(VectorOf(TransformDomain).WithMaxSize(20),
                              ElementOf({false, true})))
        .WithMaxSize(100);

void SimpleCube(const std::vector<CubeOp> &inputs) {
  manifold::ManifoldParams().intermediateChecks = true;
  manifold::ManifoldParams().processOverlaps = false;
  manifold::Manifold result;
  for (const auto &input : inputs) {
    auto cube = manifold::Manifold::Cube();
    for (const auto &transform : input.transforms) {
      switch (transform.ty) {
        case TransformType::Translate:
          cube = cube.Translate({std::get<0>(transform.vector),
                                 std::get<1>(transform.vector),
                                 std::get<2>(transform.vector)});
          break;
        case TransformType::Rotate:
          cube = cube.Rotate(std::get<0>(transform.vector),
                             std::get<1>(transform.vector),
                             std::get<2>(transform.vector));
          break;
        case TransformType::Scale:
          cube = cube.Scale({std::get<0>(transform.vector),
                             std::get<1>(transform.vector),
                             std::get<2>(transform.vector)});
          break;
      }
    }

    std::atomic<pid_t> tid;
    std::atomic<bool> faulted(true);
    auto asyncFuture = std::async(
        std::launch::async, [&result, &faulted, &tid, &cube, &input]() {
          tid.store(gettid());
          if (input.isUnion) {
            result += cube;
          } else {
            result -= cube;
          }
          EXPECT_EQ(result.Status(), manifold::Manifold::Error::NoError);
          faulted.store(false);
        });
    if (asyncFuture.wait_for(std::chrono::milliseconds(10000)) ==
        std::future_status::timeout) {
      printf("timeout after %dms...\n", 10000);
      pthread_cancel(tid.load());
    }

    EXPECT_FALSE(faulted.load());
    if (faulted.load()) break;
  }
}

FUZZ_TEST(ManifoldFuzz, SimpleCube).WithDomains(CsgDomain);

TEST(ManifoldFuzz, SimpleCubeRegression) {
  SimpleCube({{{{static_cast<TransformType>(1),
                 {-0.10000000000000001, 0.10000000000000001, -1.}}},
               1},
              {{}, 1},
              {{{static_cast<TransformType>(1),
                 {-0.10000000000000001, -0.10000000000066571, -1.}}},
               0}});
}
