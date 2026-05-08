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

// Tests for the public `ExecutionContext` API (progress reporting,
// cancellation semantics, sticky-cancel behavior, ctx reuse, counter
// reset invariants) plus mechanism tests for the ctx-aware `for_each`
// overloads in parallel.h and the parallel mergeSort stability in
// sort.h.

#include <atomic>
#include <chrono>
#include <thread>

#include "../src/execution_impl.h"
#include "../src/parallel.h"
#include "../src/vec.h"
#include "manifold/manifold.h"
#include "test.h"

using namespace manifold;

// A CSG tree with N leaves reduces to 1 result in N-1 combinations.
TEST(Manifold, ExecutionContextProgress) {
  // Build a tree with 5 leaves: a union of 5 cubes.
  std::vector<Manifold> items;
  for (int i = 0; i < 5; i++) {
    items.push_back(Manifold::Cube(vec3(1), true).Translate(vec3(i * 2, 0, 0)));
  }
  Manifold u = Manifold::BatchBoolean(items, OpType::Add);

  ExecutionContext ctx;
  EXPECT_EQ(u.Status(ctx), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 4);  // 5 leaves - 1
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 4);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Forcing evaluation first, then observing with a fresh context, should
// report no pending work: the Manifold is already a leaf.
TEST(Manifold, ExecutionContextAlreadyEvaluated) {
  Manifold u = Manifold::Cube(vec3(1), true) + Manifold::Cube(vec3(1), true);
  // Force evaluation.
  EXPECT_EQ(u.Status(), Manifold::Error::NoError);

  ExecutionContext ctx;
  EXPECT_EQ(u.Status(ctx), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 0);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 0);
}

// Setting cancel before calling Status(ctx) should return Cancelled
// without doing any work.
TEST(Manifold, ExecutionContextCancelBeforeEval) {
  std::vector<Manifold> items;
  for (int i = 0; i < 10; i++) {
    items.push_back(Manifold::Sphere(1.0, 32).Translate(vec3(i * 0.5, 0, 0)));
  }
  Manifold u = Manifold::BatchBoolean(items, OpType::Add);

  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(u.Status(ctx), Manifold::Error::Cancelled);
}

// Cancel from another thread mid-evaluation. Expensive CSG tree so there's
// time to set the flag before the full eval finishes. Verify Error::Cancelled
// is returned and doneBooleans < totalBooleans.
//
// MANIFOLD_PAR guard: emscripten/WASM builds without pthreads can't
// construct std::thread and abort at runtime. Skip on non-parallel builds
// — cancellation mid-evaluation isn't meaningful there anyway since the
// eval is synchronous.
#if MANIFOLD_PAR == 1
TEST(Manifold, ExecutionContextCancelConcurrent) {
  // Build a large enough tree that evaluation takes measurable time.
  std::vector<Manifold> items;
  for (int i = 0; i < 50; i++) {
    items.push_back(Manifold::Sphere(1.0, 64).Translate(vec3(i * 0.3, 0, 0)));
  }
  Manifold u = Manifold::BatchBoolean(items, OpType::Add);

  ExecutionContext ctx;
  std::atomic<Manifold::Error> result{Manifold::Error::NoError};
  std::thread evalThread([&] { result.store(u.Status(ctx)); });

  // Yield briefly so evaluation starts, then request cancel.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ctx.Cancel();
  evalThread.join();

  // Cancel may have fired between ops or after the whole eval finished
  // (depending on timing). Either Cancelled (expected) or NoError (raced
  // past us) is acceptable.
  EXPECT_TRUE(result.load() == Manifold::Error::Cancelled ||
              result.load() == Manifold::Error::NoError);
  if (result.load() == Manifold::Error::Cancelled) {
    EXPECT_LT(ctx.impl_->doneBooleans.load(), ctx.impl_->totalBooleans.load());
    // Sub-Boolean granularity: PhaseBalance skips its top-up on cancel, so
    // donePhases must reflect partial work (strictly less than the full
    // budget). This pins the "partial publication is intentional" invariant.
    EXPECT_LT(ctx.Progress(), 1.0);
  }
}
#endif  // MANIFOLD_PAR == 1

// Cancel mid-evaluation of a *single* Boolean3 call (as opposed to the outer
// CSG tree loop). Before per-phase cancel checks inside Boolean3, this would
// only return Cancelled if the flag fired before the boolean started; a large
// single boolean would run to completion. With phase-boundary checks, cancel
// requested mid-way is honored.
#if MANIFOLD_PAR == 1
TEST(Manifold, ExecutionContextCancelMidBoolean) {
  // One big boolean — N-1 = 1 op, so outer-loop cancel only has a single
  // checkpoint. The work itself (two large spheres overlapping) takes long
  // enough that cancel fires reliably while Boolean3 is running.
  Manifold a = Manifold::Sphere(1.0, 256);
  Manifold b = Manifold::Sphere(1.0, 256).Translate(vec3(0.5, 0, 0));
  Manifold u = a + b;

  ExecutionContext ctx;
  std::atomic<Manifold::Error> result{Manifold::Error::NoError};
  std::thread evalThread([&] { result.store(u.Status(ctx)); });

  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ctx.Cancel();
  evalThread.join();

  // Either outcome is acceptable. NoError: eval raced past the 1ms sleep
  // before cancel fired. Cancelled: caught by an inner Boolean3 phase
  // check or by the outer CsgOpNode::ToLeafNode check after a fast
  // SimpleBoolean — we can't cheaply distinguish those paths.
  EXPECT_TRUE(result.load() == Manifold::Error::Cancelled ||
              result.load() == Manifold::Error::NoError);
}
#endif  // MANIFOLD_PAR == 1

// Status() and Status(ctx) should return identical results when no cancel.
TEST(Manifold, ExecutionContextMatchesPlainStatus) {
  Manifold u = (Manifold::Cube(vec3(1), true) + Manifold::Sphere(1.0, 32)) -
               Manifold::Tetrahedron();

  // Build two equivalent manifolds (CsgOpNodes) to avoid caching effects.
  auto makeTree = [] {
    return (Manifold::Cube(vec3(1), true) + Manifold::Sphere(1.0, 32)) -
           Manifold::Tetrahedron();
  };
  Manifold a = makeTree();
  Manifold b = makeTree();

  Manifold::Error aStatus = a.Status();
  ExecutionContext ctx;
  Manifold::Error bStatus = b.Status(ctx);
  EXPECT_EQ(aStatus, bStatus);
}

// N-1 invariant holds for Subtract trees too.
TEST(Manifold, ExecutionContextProgressSubtract) {
  // Tree shape: (cube + sphere) - tet - octahedron
  Manifold u = (Manifold::Cube(vec3(1), true) + Manifold::Sphere(1.0, 16)) -
               Manifold::Tetrahedron() - Manifold::Cube(vec3(0.5), true);

  ExecutionContext ctx;
  EXPECT_EQ(u.Status(ctx), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 3);  // 4 leaves - 1
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 3);
}

// Reusing the same context for two sequential evaluations should reset
// doneBooleans so Progress is always in [0, 1].
TEST(Manifold, ExecutionContextReuse) {
  auto makeTree = [](int n) {
    std::vector<Manifold> items;
    for (int i = 0; i < n; i++) {
      items.push_back(
          Manifold::Cube(vec3(1), true).Translate(vec3(i * 2, 0, 0)));
    }
    return Manifold::BatchBoolean(items, OpType::Add);
  };
  Manifold a = makeTree(5);
  Manifold b = makeTree(3);

  ExecutionContext ctx;
  EXPECT_EQ(a.Status(ctx), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 4);

  // Reuse ctx for b. doneBooleans should reset to 0 at start.
  EXPECT_EQ(b.Status(ctx), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 2);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 2);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Cancel is permanent: once fired, Status on that Manifold stays Cancelled
// even when called without a context.
TEST(Manifold, ExecutionContextCancelPermanent) {
  std::vector<Manifold> items;
  for (int i = 0; i < 30; i++) {
    items.push_back(Manifold::Sphere(1.0, 48).Translate(vec3(i * 0.3, 0, 0)));
  }
  Manifold u = Manifold::BatchBoolean(items, OpType::Add);

  ExecutionContext ctx;
  ctx.Cancel();  // cancel before any work
  EXPECT_EQ(u.Status(ctx), Manifold::Error::Cancelled);

  // Plain Status() on the same Manifold should still report Cancelled —
  // the cached leaf is the Cancelled one.
  EXPECT_EQ(u.Status(), Manifold::Error::Cancelled);

  // Re-querying with the same context (still cancelled) is still Cancelled.
  EXPECT_EQ(u.Status(ctx), Manifold::Error::Cancelled);
}

// Once a context is cancelled it stays cancelled for any future evaluation
// through it — not just the Manifold in flight at the time of Cancel().
TEST(Manifold, ExecutionContextCancelStickyAcrossManifolds) {
  ExecutionContext ctx;
  ctx.Cancel();

  // Fresh Manifold that needs evaluation (CsgOpNode, not a leaf) — the
  // cancelled ctx should short-circuit it to Cancelled.
  Manifold u = Manifold::Cube() + Manifold::Sphere(0.5);
  EXPECT_EQ(u.Status(ctx), Manifold::Error::Cancelled);

  // Another unrelated Manifold, same ctx — also Cancelled.
  Manifold v =
      Manifold::Tetrahedron() + Manifold::Sphere(0.3).Translate(vec3(2, 0, 0));
  EXPECT_EQ(v.Status(ctx), Manifold::Error::Cancelled);
}

// A cancelled ctx does NOT contaminate a fresh context: constructing a new
// ExecutionContext gives an evaluation that can complete normally.
TEST(Manifold, ExecutionContextFreshContextEscapesCancel) {
  ExecutionContext cancelledCtx;
  cancelledCtx.Cancel();
  Manifold dead = Manifold::Cube() + Manifold::Sphere(0.5);
  EXPECT_EQ(dead.Status(cancelledCtx), Manifold::Error::Cancelled);

  ExecutionContext fresh;
  Manifold live = Manifold::Cube() + Manifold::Sphere(0.5);
  EXPECT_EQ(live.Status(fresh), Manifold::Error::NoError);
  EXPECT_FALSE(fresh.Cancelled());
}

// A cancelled ctx applied to a Manifold that is already a leaf (no
// evaluation needed) returns the Manifold's real status — cancel only
// applies when there is actual evaluation work to short-circuit. This
// matches the docstring phrasing "every subsequent evaluation".
TEST(Manifold, ExecutionContextCancelSkippedOnLeaf) {
  Manifold cube = Manifold::Cube();  // CsgLeafNode from the start
  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(cube.Status(ctx), Manifold::Error::NoError);
}

// Progress is observable from another thread while evaluation runs.
// MANIFOLD_PAR guard: see note on ExecutionContextCancelConcurrent above.
#if MANIFOLD_PAR == 1
TEST(Manifold, ExecutionContextConcurrentProgress) {
  std::vector<Manifold> items;
  for (int i = 0; i < 30; i++) {
    items.push_back(Manifold::Sphere(1.0, 48).Translate(vec3(i * 0.5, 0, 0)));
  }
  Manifold u = Manifold::BatchBoolean(items, OpType::Add);

  ExecutionContext ctx;
  std::atomic<bool> sawProgress{false};
  std::thread observer([&] {
    for (int i = 0; i < 1000 && !sawProgress.load(); i++) {
      int done = ctx.impl_->doneBooleans.load();
      int total = ctx.impl_->totalBooleans.load();
      if (total > 0 && done > 0 && done <= total) {
        sawProgress.store(true);
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  EXPECT_EQ(u.Status(ctx), Manifold::Error::NoError);
  observer.join();
  // Final state invariants.
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 29);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 29);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
  // We expect to have seen a mid-evaluation snapshot, though this is
  // timing-dependent. Not strictly required but usually the case.
  // (Not asserting sawProgress to avoid CI flakes.)
}

// Trivial Booleans (empty input, no intersection) early-return from
// Boolean3::Result before any phase() site executes. The PhaseBalance scope
// guard must still credit a full kPhasesPerBoolean phases on those returns
// so Progress() reaches 1.0 after a sequence of no-op evaluations.
TEST(Manifold, ExecutionContextProgressReachesOneOnTrivialBooleans) {
  ExecutionContext ctx;
  // Boolean of empty + cube: Boolean3::Result takes the IsEmpty fast-path.
  Manifold result = Manifold() + Manifold::Cube();
  EXPECT_EQ(result.Status(ctx), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 1);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 1);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), kPhasesPerBoolean);
  EXPECT_EQ(ctx.impl_->donePhases.load(), kPhasesPerBoolean);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Sub-Boolean granularity: within a single Boolean3 op, Progress() should
// observably advance through phases (donePhases > 0 while doneBooleans == 0).
// MANIFOLD_PAR guard: same as above, polling thread requires std::thread.
TEST(Manifold, ExecutionContextSubBooleanProgress) {
  // A single nontrivial Boolean with a real intersection: NumLeaves == 2 ->
  // totalBooleans == 1, so doneBooleans only flips from 0 to 1 at the very
  // end. Any observed Progress() > 0 mid-eval comes from sub-Boolean phase
  // advancement. Two overlapping spheres ensure the Boolean takes the full
  // path (no degenerate-input early return that would short-circuit phases).
  Manifold a = Manifold::Sphere(1.0, 256);
  Manifold b = Manifold::Sphere(1.0, 256).Translate(vec3(0.5, 0, 0));
  Manifold result = a + b;

  ExecutionContext ctx;
  std::atomic<bool> sawSubBoolean{false};
  std::thread observer([&] {
    for (int i = 0; i < 10000 && !sawSubBoolean.load(); i++) {
      const int donePhases = ctx.impl_->donePhases.load();
      const int doneBooleans = ctx.impl_->doneBooleans.load();
      if (donePhases > 0 && doneBooleans == 0) {
        sawSubBoolean.store(true);
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  });

  EXPECT_EQ(result.Status(ctx), Manifold::Error::NoError);
  observer.join();
  // Final state: exactly kPhasesPerBoolean phases per completed Boolean.
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 1);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 1);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), kPhasesPerBoolean);
  EXPECT_EQ(ctx.impl_->donePhases.load(), kPhasesPerBoolean);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
  // sawSubBoolean is timing-dependent; not asserted to avoid CI flakes.
}
#endif  // MANIFOLD_PAR == 1

// Calling Status(ctx) on an already-evaluated Manifold should reset counters,
// not leave stale values from a previous evaluation on a different Manifold.
TEST(Manifold, ExecutionContextNoStaleState) {
  Manifold complex = Manifold::BatchBoolean(
      {Manifold::Cube(vec3(1), true),
       Manifold::Cube(vec3(1), true).Translate(vec3(2, 0, 0)),
       Manifold::Cube(vec3(1), true).Translate(vec3(4, 0, 0))},
      OpType::Add);
  Manifold leaf = Manifold::Cube(vec3(1), true);

  ExecutionContext ctx;
  EXPECT_EQ(complex.Status(ctx), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 2);

  // Now evaluate a leaf Manifold with the same ctx. Counters should
  // reflect that there's no work to do (0/0), not the stale 2/2.
  EXPECT_EQ(leaf.Status(ctx), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 0);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 0);
}

// ExecutionContext copies share state (pimpl semantics): one thread
// evaluates, another thread holds a copy and observes the same progress.
TEST(Manifold, ExecutionContextCopyShareState) {
  ExecutionContext ctx1;
  ExecutionContext ctx2 = ctx1;  // copy shares impl

  // Cancel on one copy should be observable through the other.
  EXPECT_FALSE(ctx2.Cancelled());
  ctx1.Cancel();
  EXPECT_TRUE(ctx1.Cancelled());
  EXPECT_TRUE(ctx2.Cancelled());

  // Counters also shared.
  EXPECT_EQ(ctx1.impl_->totalBooleans.load(), 0);
  ctx2.impl_->totalBooleans.store(42);
  EXPECT_EQ(ctx1.impl_->totalBooleans.load(), 42);
}

// Move construction and move assignment preserve shared state.
TEST(Manifold, ExecutionContextMoveSemantics) {
  ExecutionContext ctx1;
  ctx1.Cancel();
  ExecutionContext ctx2 = std::move(ctx1);  // move-construct
  EXPECT_TRUE(ctx2.Cancelled());

  ExecutionContext ctx3;
  EXPECT_FALSE(ctx3.Cancelled());
  ctx3 = std::move(ctx2);  // move-assign
  EXPECT_TRUE(ctx3.Cancelled());
}

// A Manifold that's already a leaf (no CSG tree) should report no work.
TEST(Manifold, ExecutionContextNoWorkNeeded) {
  Manifold cube = Manifold::Cube(vec3(1), true);
  ExecutionContext ctx;
  EXPECT_EQ(cube.Status(ctx), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 0);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 0);
  // No work scheduled = trivially complete = 1.0.
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// doneBooleans reaches totalBooleans exactly for many tree shapes — the N-1
// invariant shouldn't depend on op type or tree shape.
TEST(Manifold, ExecutionContextProgressInvariant) {
  // Intersect
  {
    std::vector<Manifold> items{
        Manifold::Cube(vec3(2), true),
        Manifold::Cube(vec3(2), true).Translate(vec3(0.5, 0, 0)),
        Manifold::Cube(vec3(2), true).Translate(vec3(0, 0.5, 0))};
    Manifold u = Manifold::BatchBoolean(items, OpType::Intersect);
    ExecutionContext ctx;
    EXPECT_EQ(u.Status(ctx), Manifold::Error::NoError);
    EXPECT_EQ(ctx.impl_->totalBooleans.load(), 2);
    EXPECT_EQ(ctx.impl_->doneBooleans.load(), 2);
    EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
  }
  // Mixed Add/Subtract
  {
    Manifold u = (Manifold::Cube(vec3(1), true) +
                  Manifold::Cube(vec3(1), true).Translate(vec3(2, 0, 0))) -
                 Manifold::Tetrahedron();
    ExecutionContext ctx;
    EXPECT_EQ(u.Status(ctx), Manifold::Error::NoError);
    EXPECT_EQ(ctx.impl_->totalBooleans.load(), 2);  // 3 leaves - 1
    EXPECT_EQ(ctx.impl_->doneBooleans.load(), 2);
    EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
  }
  // Disjoint union (triggers Compose path)
  {
    std::vector<Manifold> items;
    for (int i = 0; i < 6; i++) {
      items.push_back(
          Manifold::Cube(vec3(0.5), true).Translate(vec3(i * 2, 0, 0)));
    }
    Manifold u = Manifold::BatchBoolean(items, OpType::Add);
    ExecutionContext ctx;
    EXPECT_EQ(u.Status(ctx), Manifold::Error::NoError);
    EXPECT_EQ(ctx.impl_->totalBooleans.load(), 5);
    EXPECT_EQ(ctx.impl_->doneBooleans.load(), 5);
    // Progress == 1.0 here is the load-bearing assertion: it cross-checks
    // that Compose's `donePhases += (set.size() - 1) * kPhasesPerBoolean`
    // mirror was wired in. Without it, Progress would land below 1.0 and
    // only the Compose-path multiplier would catch the bug.
    EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
  }
}

// Refine(ctx) on an already-evaluated leaf: no boolean work, so the only
// phase advance comes from Refine itself. donePhases = totalPhases = 1
// at completion. Cube has 12 tris; Refine(2) splits each into 4 = 48.
TEST(Manifold, ExecutionContextRefineProgress) {
  Manifold cube = Manifold::Cube(vec3(1), true);
  ASSERT_EQ(cube.NumTri(), 12u);
  ExecutionContext ctx;
  Manifold refined = cube.Refine(2, ctx);
  EXPECT_EQ(refined.Status(), Manifold::Error::NoError);
  EXPECT_EQ(refined.NumTri(), 48u);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), 1);
  EXPECT_EQ(ctx.impl_->donePhases.load(), 1);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Pre-cancelled ctx: Refine(ctx) returns Cancelled without doing the work.
TEST(Manifold, ExecutionContextRefineCancelled) {
  Manifold cube = Manifold::Cube(vec3(1), true);
  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(cube.Refine(2, ctx).Status(), Manifold::Error::Cancelled);
  EXPECT_EQ(cube.RefineToLength(0.1, ctx).Status(), Manifold::Error::Cancelled);
  EXPECT_EQ(cube.RefineToTolerance(0.01, ctx).Status(),
            Manifold::Error::Cancelled);
}

// Refine on an unevaluated CSG tree: totalPhases reserves the full eventual
// budget (boolean phases + 1 for Refine) up front via GetCsgLeafNode's
// extraPhases parameter. This guarantees Progress() monotonically rises and
// never dips mid-call, even between the boolean tree finishing and Refine
// starting. Pre-cancel exposes the post-reset totalPhases value before any
// donePhases work happens.
TEST(Manifold, ExecutionContextRefineReservesPhasesUpFront) {
  Manifold u = Manifold::Cube() + Manifold::Sphere(0.5);  // 1 boolean
  ExecutionContext ctx;
  ctx.Cancel();
  Manifold refined = u.Refine(2, ctx);
  EXPECT_EQ(refined.Status(), Manifold::Error::Cancelled);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), kPhasesPerBoolean + 1);
}

// Direct test of the ctx-aware `for_each` overload: cancel-already-set
// must skip the entire range; cancel-unset must run every element. This
// is the mechanism test; the integration test that cancel works
// end-to-end on a real boolean is ExecutionContextCancelMidBoolean.
TEST(Manifold, ForEachCtxSkipsWhenCancelled) {
  ExecutionContext ctx;
  ctx.Cancel();
  std::atomic<int> calls{0};
  Vec<int> range(100);
  sequence(range.begin(), range.end());
  for_each(ExecutionPolicy::Seq, range.begin(), range.end(), ctx.impl_.get(),
           [&calls](int) { calls.fetch_add(1, std::memory_order_relaxed); });
  EXPECT_EQ(calls.load(), 0);
}

// Nullptr ctx: must not crash and must invoke the functor for every element.
TEST(Manifold, ForEachCtxNullCtxPassesThrough) {
  std::atomic<int> calls{0};
  Vec<int> range(10);
  sequence(range.begin(), range.end());
  for_each(ExecutionPolicy::Seq, range.begin(), range.end(),
           static_cast<ExecutionContext::Impl*>(nullptr),
           [&calls](int) { calls.fetch_add(1, std::memory_order_relaxed); });
  EXPECT_EQ(calls.load(), 10);
}

// Cancel set mid-iteration must be observed within bounded latency
// (kSeqCancelChunk = 1024 in parallel.h). Functor cancels after 100
// calls; total calls must be < N + 1024 (= 100 + at most one full chunk).
TEST(Manifold, ForEachCtxCancelMidSeqLoop) {
  ExecutionContext ctx;
  std::atomic<int> calls{0};
  constexpr int N = 10000;
  Vec<int> range(N);
  sequence(range.begin(), range.end());
  for_each(ExecutionPolicy::Seq, range.begin(), range.end(), ctx.impl_.get(),
           [&calls, &ctx](int) {
             const int n = calls.fetch_add(1, std::memory_order_relaxed) + 1;
             if (n == 100) ctx.Cancel();
           });
  EXPECT_GE(calls.load(), 100);
  EXPECT_LT(calls.load(), 100 + 1024);
}

// Sweeps equal-key bucket sizes that span the parallel mergeSort's
// partition boundaries (n > kSeqThreshold under MANIFOLD_PAR=ON). Tags
// are sequential and unique; keys are scrambled by a multiplicative hash
// so the input isn't already sorted by key. std::stable_sort is the
// oracle.
TEST(Manifold, ParallelStableSortStability) {
  constexpr size_t kN = 50000;  // > kSeqThreshold to hit parallel
  constexpr size_t kBucketSizes[] = {kN, kN / 2, kN / 4, kN / 8, 1000, 100};
  for (size_t bucket : kBucketSizes) {
    struct Item {
      size_t key;
      int tag;
    };
    std::vector<Item> items;
    items.reserve(kN);
    for (size_t i = 0; i < kN; ++i) {
      items.push_back({(i * 2654435761u % kN) / bucket, static_cast<int>(i)});
    }
    auto expected = items;
    auto cmp = [](const Item& a, const Item& b) { return a.key > b.key; };
    std::stable_sort(expected.begin(), expected.end(), cmp);
    manifold::stable_sort(items.begin(), items.end(), cmp);
    for (size_t i = 0; i < items.size(); ++i) {
      ASSERT_EQ(items[i].tag, expected[i].tag)
          << "bucket=" << bucket << " i=" << i;
    }
  }
}
