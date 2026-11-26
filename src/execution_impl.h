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
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
#include <chrono>
#endif

#include "manifold/common.h"

namespace manifold {

inline bool IsCancelled(ExecutionContext::Impl* ctx);

/** @ingroup Private
 *
 * Number of progress phases per `Boolean3::Result`. Must equal the count of
 * `phase()` sites in `boolean_result.cpp`'s `Boolean3::Result` (asserted on
 * the happy-path return). Bump in lockstep when adding/removing a site.
 */
constexpr int kPhasesPerBoolean = 11;

/** @ingroup Private
 *
 * Heavy phases in the `Manifold::Impl::Impl(MeshGLP, ...)` ingest;
 * credit is published only after a phase completes successfully. Bump
 * in lockstep with the `ADVANCE_PHASE_OR_RETURN(ctx)` sites in the
 * ctor body.
 */
constexpr int kPhasesPerFromMesh = 6;

/** @ingroup Private
 *
 * Heavy phases in `Manifold::Impl::CreateLevelSet(...)`: the four grid
 * loops (voxel SDF sampling, NearSurface, ComputeVerts, BuildTris)
 * followed by one lumped finalize phase (CreateHalfedges through
 * InitializeOriginal). The NearSurface hash-table resize loop may
 * re-run NearSurface, but counts as a single phase regardless. Bump in
 * lockstep with `ADVANCE_PHASE_OR_RETURN(ctx)` sites in CreateLevelSet.
 */
constexpr int kPhasesPerLevelSet = 5;

/** @ingroup Private
 *
 * Pimpl for ExecutionContext. `cancel` is private; use `IsCancelled(ctx)`
 * to read it -- this is the canonical reader, enforced by the type system.
 * `totalBooleans`, `doneBooleans`, `totalPhases`, and `donePhases` are
 * public for progress reporting and test introspection.
 *
 * `Progress() = donePhases / totalPhases`. `totalPhases` is the canonical
 * progress denominator across op types; for Boolean trees it is set at
 * `GetCsgLeafNode` to `totalBooleans * kPhasesPerBoolean`. The static
 * factories (`FromMeshGL`, `LevelSet`) set it to their own `kPhasesPer*`
 * and advance `donePhases` with `ADVANCE_PHASE_OR_RETURN`.
 *
 * `totalBooleans`/`doneBooleans` are kept as introspection counters for
 * tests and tools; they are not used in `Progress()`.
 */
struct ExecutionContext::Impl {
 public:
  std::atomic<int> totalBooleans{0};
  std::atomic<int> doneBooleans{0};
  std::atomic<int> totalPhases{0};
  std::atomic<int> donePhases{0};

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

#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
// Per-op local phase-timing state. Each op keeps this on its own stack rather
// than on the shared ctx, so concurrent ops (booleans in a CSG tree, or calls
// sharing one ctx) don't clobber each other's baseline. `uid` tags the printed
// lines so the interleaved output of parallel ops can be split apart.
// RecordPhase prints the span since the previous boundary when verbose >= 2;
// it is defined in execution_impl.cpp so <iostream> stays out of this
// widely-included header.
struct LocalPhaseTiming {
  std::chrono::high_resolution_clock::time_point last;
  int uid;
};
LocalPhaseTiming BeginLocalPhaseTiming();
void RecordPhase(LocalPhaseTiming& timing, int phase, const char* file,
                 int line);
#endif

}  // namespace manifold

/** @ingroup Private
 *
 * Phase-boundary checkpoint for `Manifold::Impl` methods (relies on
 * `this->MakeEmpty` and on a `MANIFOLD_BEGIN_PHASE_TIMING()` at the top of the
 * method). Call count must equal the method's `kPhasesPer*`.
 */
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
#define MANIFOLD_BEGIN_PHASE_TIMING() \
  manifold::LocalPhaseTiming phaseTiming = manifold::BeginLocalPhaseTiming()
#define MANIFOLD_RECORD_PHASE(ctx)                                         \
  manifold::RecordPhase(phaseTiming,                                       \
                        (ctx)->donePhases.load(std::memory_order_relaxed), \
                        __FILE__, __LINE__)
#else
#define MANIFOLD_BEGIN_PHASE_TIMING() ((void)0)
#define MANIFOLD_RECORD_PHASE(ctx) ((void)0)
#endif

#define ADVANCE_PHASE_OR_RETURN(ctx)                             \
  do {                                                           \
    if (manifold::IsCancelled(ctx)) {                            \
      MakeEmpty(manifold::Manifold::Error::Cancelled);           \
      return;                                                    \
    }                                                            \
    if (ctx) {                                                   \
      (ctx)->donePhases.fetch_add(1, std::memory_order_relaxed); \
      MANIFOLD_RECORD_PHASE(ctx);                                \
    }                                                            \
  } while (0)
