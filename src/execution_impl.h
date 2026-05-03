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

inline bool IsCancelled(ExecutionContext::Impl* ctx);

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
  friend class manifold::ExecutionContext;
};

/** @ingroup Private
 *
 * Canonical reader for the cancel flag. The only path to observe cancel
 * state from internal code -- `Impl::cancel` is private and friended only
 * to this function and `ExecutionContext` (for its public `Cancel`/`Cancelled`
 * members). The ctx-aware overloads in `parallel.h` go through here.
 *
 * Returns false if `ctx` is nullptr (no-cancellation calls), otherwise
 * loads `cancel` with `memory_order_relaxed` (cancel is advisory; we
 * don't need synchronization with other operations).
 */
inline bool IsCancelled(ExecutionContext::Impl* ctx) {
  return ctx && ctx->cancel.load(std::memory_order_relaxed);
}

}  // namespace manifold
