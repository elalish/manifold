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
#include <type_traits>
#include <utility>

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

/** @ingroup Private
 *
 * Wrap a functor so each invocation returns early if the
 * ExecutionContext has been cancelled. `for_each` / `transform` / etc.
 * will still call the wrapped functor for every remaining iteration,
 * but each no-ops — TBB's scheduler doesn't let us early-terminate,
 * so we drain cheaply. `ctx` may be nullptr, in which case the cancel
 * branch is always cold.
 *
 * Only appropriate for functors whose "no-op this iteration" behavior
 * is safe — `for_each` over a range that writes independent output
 * slots. Not safe for `transform` / scan-style primitives where
 * skipping an iteration would silently corrupt the result. Non-void
 * functors fail to compile (the `return;` on the cancel path is
 * ill-formed for a non-void return type), enforcing this at build
 * time.
 */
template <typename F>
struct Cancellable {
  ExecutionContext::Impl* ctx;
  F f;
  template <typename... Args>
  auto operator()(Args&&... args) -> decltype(f(std::forward<Args>(args)...)) {
    if (ctx && ctx->cancel.load(std::memory_order_relaxed)) return;
    f(std::forward<Args>(args)...);
  }
};

template <typename F>
Cancellable<std::decay_t<F>> cancellable(ExecutionContext::Impl* ctx, F&& f) {
  return {ctx, std::forward<F>(f)};
}

}  // namespace manifold
