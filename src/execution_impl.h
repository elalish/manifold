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

#pragma once
#include <atomic>

#include "manifold/common.h"

namespace manifold {

/** @ingroup Private
 *
 * Pimpl for ExecutionContext. Holds the atomic state observed/mutated by
 * the CSG evaluation machinery. Internal code accesses this via
 * ctx.impl_->field directly.
 */
struct ExecutionContext::Impl {
  std::atomic<int> totalBooleans{0};
  std::atomic<int> doneBooleans{0};
  std::atomic<bool> cancel{false};
};

}  // namespace manifold
