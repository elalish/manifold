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

enum class Transform { Translate, Rotate, Scale };

// larger numbers may cause precision issues, prefer to test them later
auto GoodNumbers = OneOf(InRange(0.1, 10.0), InRange(-10.0, -0.1));
auto Vec3Domain = ArrayOf<3>(GoodNumbers);
auto TransformDomain = PairOf(
    ElementOf({Transform::Translate, Transform::Rotate, Transform::Scale}),
    Vec3Domain);
auto CsgDomain = VectorOf(PairOf(VectorOf(TransformDomain).WithMaxSize(20),
                                 ElementOf({0, 1})))
                     .WithMaxSize(100);

void SimpleCube(
    const std::vector<std::pair<
        std::vector<std::pair<Transform, std::array<double, 3>>>, int>>
        &inputs) {
  manifold::ManifoldParams().intermediateChecks = true;
  manifold::ManifoldParams().processOverlaps = false;
  manifold::Manifold result;
  for (const auto &input : inputs) {
    auto cube = manifold::Manifold::Cube();
    for (const auto &transform : input.first) {
      switch (transform.first) {
        case Transform::Translate:
          cube = cube.Translate({std::get<0>(transform.second),
                                 std::get<1>(transform.second),
                                 std::get<2>(transform.second)});
          break;
        case Transform::Rotate:
          cube = cube.Rotate(std::get<0>(transform.second),
                             std::get<1>(transform.second),
                             std::get<2>(transform.second));
          break;
        case Transform::Scale:
          cube = cube.Scale({std::get<0>(transform.second),
                             std::get<1>(transform.second),
                             std::get<2>(transform.second)});
          break;
      }
    }

    std::atomic<pid_t> tid;
    std::atomic<bool> faulted(true);
    auto asyncFuture = std::async(
        std::launch::async, [&result, &faulted, &tid, &cube, &input]() {
          tid.store(gettid());
          if (input.second) {
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
  SimpleCube({{{{static_cast<Transform>(1),
                 {-0.10000000000000001, 0.10000000000000001, -1.}}},
               1},
              {{}, 1},
              {{{static_cast<Transform>(1),
                 {-0.10000000000000001, -0.10000000000066571, -1.}}},
               0}});
}
