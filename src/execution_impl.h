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

inline bool IsCancelled(ExecutionContext::Impl* ctx);

template <typename F>
struct Cancellable;

/** @ingroup Private
 *
 * Pimpl for ExecutionContext. `cancel` is private; use `IsCancelled(ctx)`
 * to read it -- this is the canonical reader, enforced by the type system.
 * `totalBooleans` and `doneBooleans` are public for progress reporting and
 * test introspection.
 */
struct ExecutionContext::Impl {
 public:
  std::atomic<int> totalBooleans{0};
  std::atomic<int> doneBooleans{0};

 private:
  std::atomic<bool> cancel{false};

  friend bool IsCancelled(Impl*);
  template <typename F>
  friend struct Cancellable;
  friend class manifold::ExecutionContext;
};

/** @ingroup Private
 *
 * Canonical reader for the cancel flag. The only path to observe cancel
 * state from internal code -- `Impl::cancel` is private and friended only
 * to this function, the `Cancellable<F>` wrapper, and `ExecutionContext`
 * itself. New code that wants to check cancel must call this.
 *
 * Returns false if `ctx` is nullptr (no-cancellation calls), otherwise
 * loads `cancel` with `memory_order_relaxed` (cancel is advisory; we
 * don't need synchronization with other operations).
 */
inline bool IsCancelled(ExecutionContext::Impl* ctx) {
  return ctx && ctx->cancel.load(std::memory_order_relaxed);
}

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
 * skipping an iteration would silently corrupt the result. The
 * trailing return type plus bare `return;` on the cancel path make
 * any non-void functor ill-formed at instantiation, enforcing this
 * at build time.
 */
template <typename F>
struct Cancellable {
  ExecutionContext::Impl* ctx;
  F f;
  template <typename... Args>
  auto operator()(Args&&... args) -> decltype(f(std::forward<Args>(args)...)) {
    if (IsCancelled(ctx)) return;
    f(std::forward<Args>(args)...);
  }
};

template <typename F>
Cancellable<std::decay_t<F>> cancellable(ExecutionContext::Impl* ctx, F&& f) {
  return {ctx, std::forward<F>(f)};
}

}  // namespace manifold
