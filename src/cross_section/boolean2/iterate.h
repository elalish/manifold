// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "driver.h"

namespace manifold {
namespace boolean2 {

using FingerprintData =
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int>>;

FingerprintData FingerprintAt(const OverlapResult& r, double quantum);
FingerprintData Fingerprint(const OverlapResult& r, double eps);

enum class IterStatus {
  Converged,
  Cycled,
  MaxedOut,
};

OverlapResult IterateToFixedPoint(const std::vector<vec2>& vIn,
                                  const std::vector<EdgeM>& eIn, double eps,
                                  int maxIter = 2, int* outIters = nullptr,
                                  IterStatus* outStatus = nullptr,
                                  WindRule pred = WindRule::Add);

}  // namespace boolean2
}  // namespace manifold
