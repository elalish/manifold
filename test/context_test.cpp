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
#include <type_traits>

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
  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::NoError);
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
  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::NoError);
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
  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::Cancelled);
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
  std::thread evalThread([&] { result.store(u.WithContext(ctx).Status()); });

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
  std::thread evalThread([&] { result.store(u.WithContext(ctx).Status()); });

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
  Manifold::Error bStatus = b.WithContext(ctx).Status();
  EXPECT_EQ(aStatus, bStatus);
}

// N-1 invariant holds for Subtract trees too.
TEST(Manifold, ExecutionContextProgressSubtract) {
  // Tree shape: (cube + sphere) - tet - octahedron
  Manifold u = (Manifold::Cube(vec3(1), true) + Manifold::Sphere(1.0, 16)) -
               Manifold::Tetrahedron() - Manifold::Cube(vec3(0.5), true);

  ExecutionContext ctx;
  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::NoError);
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
  EXPECT_EQ(a.WithContext(ctx).Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 4);

  // Reuse ctx for b. doneBooleans should reset to 0 at start.
  EXPECT_EQ(b.WithContext(ctx).Status(), Manifold::Error::NoError);
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
  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::Cancelled);

  // Plain Status() on the same Manifold should still report Cancelled —
  // the cached leaf is the Cancelled one.
  EXPECT_EQ(u.Status(), Manifold::Error::Cancelled);

  // Re-querying with the same context (still cancelled) is still Cancelled.
  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::Cancelled);
}

// Once a context is cancelled it stays cancelled for any future evaluation
// through it — not just the Manifold in flight at the time of Cancel().
TEST(Manifold, ExecutionContextCancelStickyAcrossManifolds) {
  ExecutionContext ctx;
  ctx.Cancel();

  // Fresh Manifold that needs evaluation (CsgOpNode, not a leaf) — the
  // cancelled ctx should short-circuit it to Cancelled.
  Manifold u = Manifold::Cube() + Manifold::Sphere(0.5);
  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::Cancelled);

  // Another unrelated Manifold, same ctx — also Cancelled.
  Manifold v =
      Manifold::Tetrahedron() + Manifold::Sphere(0.3).Translate(vec3(2, 0, 0));
  EXPECT_EQ(v.WithContext(ctx).Status(), Manifold::Error::Cancelled);
}

// A cancelled ctx does NOT contaminate a fresh context: constructing a new
// ExecutionContext gives an evaluation that can complete normally.
TEST(Manifold, ExecutionContextFreshContextEscapesCancel) {
  ExecutionContext cancelledCtx;
  cancelledCtx.Cancel();
  Manifold dead = Manifold::Cube() + Manifold::Sphere(0.5);
  EXPECT_EQ(dead.WithContext(cancelledCtx).Status(),
            Manifold::Error::Cancelled);

  ExecutionContext fresh;
  Manifold live = Manifold::Cube() + Manifold::Sphere(0.5);
  EXPECT_EQ(live.WithContext(fresh).Status(), Manifold::Error::NoError);
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
  EXPECT_EQ(cube.WithContext(ctx).Status(), Manifold::Error::NoError);
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

  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::NoError);
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
  EXPECT_EQ(result.WithContext(ctx).Status(), Manifold::Error::NoError);
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

  EXPECT_EQ(result.WithContext(ctx).Status(), Manifold::Error::NoError);
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
  EXPECT_EQ(complex.WithContext(ctx).Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 2);

  // Now evaluate a leaf Manifold with the same ctx. Counters should
  // reflect that there's no work to do (0/0), not the stale 2/2.
  EXPECT_EQ(leaf.WithContext(ctx).Status(), Manifold::Error::NoError);
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
  EXPECT_EQ(cube.WithContext(ctx).Status(), Manifold::Error::NoError);
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
    EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::NoError);
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
    EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::NoError);
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
    EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::NoError);
    EXPECT_EQ(ctx.impl_->totalBooleans.load(), 5);
    EXPECT_EQ(ctx.impl_->doneBooleans.load(), 5);
    // Progress == 1.0 here is the load-bearing assertion: it cross-checks
    // that Compose's `donePhases += (set.size() - 1) * kPhasesPerBoolean`
    // mirror was wired in. Without it, Progress would land below 1.0 and
    // only the Compose-path multiplier would catch the bug.
    EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
  }
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

// ---- Data-attached ExecutionContext (WithContext) ------------------------
//
// Behavioral tests for the attach-ctx model: WithContext() attaches a ctx that
// is consumed only by the next eager op on the result (Status, Refine family).
// Deferred ops (Boolean, transforms, batch ops) ignore any attached ctx and
// produce a result with no attached ctx. Raw copy / assignment preserves the
// attachment (it's the same logical Manifold).
//
// "No attached ctx" is asserted behaviorally: pre-set a sentinel value on
// the ctx counters, then trigger Status() on the result. If the result has
// no attached ctx, GetCsgLeafNode(nullptr) skips the counter-reset path,
// leaving the sentinel intact. If a ctx is attached, the reset overwrites it.

// Raw copy preserves the attachment: ctx-attached on the source means
// Status on the copy observes the same ctx.
TEST(Manifold, ManifoldContextRawCopyPreservesAttachment) {
  ExecutionContext ctx;
  Manifold u = Manifold::Cube(vec3(1), true) +
               Manifold::Cube(vec3(1), true).Translate(vec3(2, 0, 0));
  Manifold attached = u.WithContext(ctx);
  Manifold copy = attached;  // copy ctor

  EXPECT_EQ(copy.Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 1);  // 2 leaves - 1
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 1);
}

// Assignment also preserves the attachment.
TEST(Manifold, ManifoldContextAssignmentPreservesAttachment) {
  ExecutionContext ctx;
  Manifold u = Manifold::Cube(vec3(1), true) +
               Manifold::Cube(vec3(1), true).Translate(vec3(2, 0, 0));
  Manifold attached = u.WithContext(ctx);
  Manifold target;
  target = attached;  // copy assignment

  EXPECT_EQ(target.Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 1);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 1);
}

// Deferred ops on a ctx-attached Manifold drop the attachment from their
// result. Verified behaviorally: a pre-cancelled ctx attached to the input
// does not cancel the result's Status (the deferred op's result has no
// ctx, so Status routes around it).
TEST(Manifold, ManifoldContextDeferredOpsDropAttachment) {
  ExecutionContext ctx;
  ctx.Cancel();
  Manifold a = Manifold::Cube(vec3(1), true).WithContext(ctx);
  Manifold b = Manifold::Cube(vec3(1), true).Translate(vec3(0.5, 0, 0));

  // None of these results carry ctx, so Status evaluates normally despite
  // the cancelled ctx attached to the input.
  EXPECT_EQ(a.Translate(vec3(1, 0, 0)).Status(), Manifold::Error::NoError);
  EXPECT_EQ(a.Scale(vec3(2)).Status(), Manifold::Error::NoError);
  EXPECT_EQ(a.Rotate(0, 0, 90).Status(), Manifold::Error::NoError);
  EXPECT_EQ(a.Boolean(b, OpType::Add).Status(), Manifold::Error::NoError);
  EXPECT_EQ((a + b).Status(), Manifold::Error::NoError);
}

// Static factories (BatchBoolean, vector-of-Manifold Hull) are deferred and
// drop any ctx attached to their inputs.
TEST(Manifold, ManifoldContextStaticFactoriesDropAttachment) {
  ExecutionContext ctx;
  ctx.Cancel();
  std::vector<Manifold> items = {
      Manifold::Cube(vec3(1), true).WithContext(ctx),
      Manifold::Cube(vec3(1), true).Translate(vec3(2, 0, 0)).WithContext(ctx)};

  // Both factories drop the input ctx; results evaluate to NoError despite
  // the cancelled ctx on every input.
  EXPECT_EQ(Manifold::BatchBoolean(items, OpType::Add).Status(),
            Manifold::Error::NoError);
  EXPECT_EQ(Manifold::Hull(items).Status(), Manifold::Error::NoError);
}

// no-arg Status() on a ctx-attached Manifold observes the attached ctx:
// counters update exactly as if Status had been passed an explicit ctx.
TEST(Manifold, ManifoldContextStatusObservesAttached) {
  std::vector<Manifold> items;
  for (int i = 0; i < 4; i++) {
    items.push_back(Manifold::Cube(vec3(1), true).Translate(vec3(i * 2, 0, 0)));
  }
  Manifold u = Manifold::BatchBoolean(items, OpType::Add);

  ExecutionContext ctx;
  Manifold attached = u.WithContext(ctx);
  EXPECT_EQ(attached.Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 3);  // 4 leaves - 1
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 3);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Idiom for observing a deferred CSG tree: build the tree, attach ctx to
// the root, call Status. Cancel via the same ctx aborts mid-evaluation.
TEST(Manifold, ManifoldContextDeferredTreeRootObserves) {
  std::vector<Manifold> items;
  for (int i = 0; i < 8; i++) {
    items.push_back(Manifold::Cube(vec3(1), true).Translate(vec3(i * 2, 0, 0)));
  }
  Manifold tree = Manifold::BatchBoolean(items, OpType::Add);

  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(tree.WithContext(ctx).Status(), Manifold::Error::Cancelled);
}

// Eager ops (Refine family) read this->ctx_ during the in-call work. A
// pre-cancelled attached ctx aborts Refine via the cancel check between
// Subdivide and the post-pass; the result is a Cancelled-status leaf.
TEST(Manifold, ManifoldContextCancelMidRefine) {
  ExecutionContext ctx;
  ctx.Cancel();
  Manifold cube = Manifold::Cube(vec3(1), true).WithContext(ctx);
  Manifold refined = cube.Refine(4);
  EXPECT_EQ(refined.Status(), Manifold::Error::Cancelled);
}

// Eager Refine reads ctx from `*this`; the produced result has no ctx
// attached. Verified behaviorally: a fresh ctx pre-loaded with a sentinel
// counter value is left untouched by Status() on the Refine result, because
// no ctx is attached to dispatch through.
TEST(Manifold, ManifoldContextRefineDropsAttachmentFromResult) {
  ExecutionContext ctx;
  Manifold cube = Manifold::Cube(vec3(1), true).WithContext(ctx);
  Manifold refined = cube.Refine(2);
  ASSERT_EQ(refined.Status(), Manifold::Error::NoError);

  // Sentinel counter on a fresh ctx: result.Status() must not touch it,
  // proving the Refine result has no attached ctx.
  ExecutionContext other;
  constexpr int kSentinel = 99;
  other.impl_->totalBooleans.store(kSentinel);
  EXPECT_EQ(refined.Status(), Manifold::Error::NoError);
  EXPECT_EQ(other.impl_->totalBooleans.load(), kSentinel);
}

// Eager Hull observes attached ctx -- a pre-cancelled ctx aborts the
// post-pass that runs after QuickHull's buildMesh, returning a Cancelled
// result rather than a finished hull.
TEST(Manifold, ManifoldContextCancelMidHull) {
  ExecutionContext ctx;
  ctx.Cancel();
  // Non-trivial input so the post-buildMesh path is reached.
  Manifold sphere = Manifold::Sphere(1.0, 64).WithContext(ctx);
  Manifold hull = sphere.Hull();
  EXPECT_EQ(hull.Status(), Manifold::Error::Cancelled);
}

// Eager Minkowski observes attached ctx -- a pre-cancelled ctx aborts
// between the hull/Boolean phases. Cancel granularity is "one batch" of
// per-face hulls.
TEST(Manifold, ManifoldContextCancelMidMinkowski) {
  ExecutionContext ctx;
  ctx.Cancel();
  // Two convex inputs hit the fast path; that's enough to exercise the
  // up-front cancel check.
  Manifold a = Manifold::Cube(vec3(1), true).WithContext(ctx);
  Manifold b = Manifold::Cube(vec3(0.5), true);
  Manifold sum = a.MinkowskiSum(b);
  EXPECT_EQ(sum.Status(), Manifold::Error::Cancelled);
}

// Concurrent cancel mid-Hull: spawn a thread that fires cancel a short
// delay after the main thread launches a Hull on a non-trivial input.
// Cancel can fire at either the post-buildMesh checkpoint or inside
// SortGeometry's per-phase checks; either way the result is Cancelled.
//
// MANIFOLD_PAR guard: same as the other concurrent tests -- requires
// std::thread.
#if MANIFOLD_PAR == 1
TEST(Manifold, ManifoldContextCancelConcurrentHull) {
  // High-resolution sphere so QuickHull + SortGeometry have measurable
  // work to do.
  Manifold sphere = Manifold::Sphere(1.0, 256);

  ExecutionContext ctx;
  std::atomic<Manifold::Error> result{Manifold::Error::NoError};
  std::thread evalThread(
      [&] { result.store(sphere.WithContext(ctx).Hull().Status()); });
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ctx.Cancel();
  evalThread.join();

  // Either NoError (eval raced past the 1ms sleep) or Cancelled
  // (caught at one of the Hull cancel checkpoints).
  EXPECT_TRUE(result.load() == Manifold::Error::Cancelled ||
              result.load() == Manifold::Error::NoError);
}
#endif  // MANIFOLD_PAR == 1

// Concurrent cancel mid-Minkowski: spawn a thread that fires cancel a
// short delay after the main thread launches a non-convex Minkowski
// (the slow path). The internal BatchBoolean calls observe ctx (via
// the tree.GetCsgLeafNode(ctx) trigger in Impl::Minkowski), so cancel
// fires inside Boolean3's per-phase checks rather than waiting for the
// whole batch.
//
// MANIFOLD_PAR guard: same as the other concurrent tests -- requires
// std::thread.
#if MANIFOLD_PAR == 1
TEST(Manifold, ManifoldContextCancelConcurrentMinkowski) {
  // Two non-convex inputs to force the slow path. The tet-difference here
  // produces near-degenerate geometry that fails the MANIFOLD_DEBUG
  // triangulation CCW check on macOS without processOverlaps -- matches
  // the workaround in Boolean.NonConvexNonConvexMinkowski{Sum,Difference}.
  ManifoldParamGuard guard;
  ManifoldParams().processOverlaps = true;

  Manifold tet = Manifold::Tetrahedron();
  Manifold nonConvex = tet - tet.Rotate(0, 0, 90).Translate(vec3(1));
  Manifold other = nonConvex.Scale(vec3(0.5));

  ExecutionContext ctx;
  std::atomic<Manifold::Error> result{Manifold::Error::NoError};
  std::thread evalThread([&] {
    result.store(nonConvex.WithContext(ctx).MinkowskiSum(other).Status());
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ctx.Cancel();
  evalThread.join();

  // Either NoError (eval raced past the 1ms sleep) or Cancelled
  // (caught mid-Boolean inside Minkowski) is acceptable.
  EXPECT_TRUE(result.load() == Manifold::Error::Cancelled ||
              result.load() == Manifold::Error::NoError);
}
#endif  // MANIFOLD_PAR == 1

// Cancellation requested on the attached ctx after Status() begins (or
// before, in the deferred-tree-root idiom) aborts evaluation.
TEST(Manifold, ManifoldContextCancelMidEval) {
  std::vector<Manifold> items;
  for (int i = 0; i < 8; i++) {
    items.push_back(Manifold::Cube(vec3(1), true).Translate(vec3(i * 2, 0, 0)));
  }
  Manifold u = Manifold::BatchBoolean(items, OpType::Add);

  ExecutionContext ctx;
  ctx.Cancel();
  Manifold attached = u.WithContext(ctx);
  EXPECT_EQ(attached.Status(), Manifold::Error::Cancelled);
}

// Trips `numProp < 3` validation but has enough verts/tris to clear the
// empty/under-4 short-circuits first. Coupled to ingest validation
// order: if MissingPositionProperties moved past NotManifold these
// triVerts wouldn't form a closed tet.
static MeshGL MalformedMissingPositionProperties() {
  MeshGL bad;
  bad.numProp = 2;
  bad.vertProperties = {0, 0, 1, 0, 0, 1, 1, 1};        // 4 verts * 2 props
  bad.triVerts = {0, 1, 2, 1, 2, 3, 2, 3, 0, 3, 0, 1};  // 4 tris
  return bad;
}

// Traits describing each ctx-aware static factory: the ctx call, the plain
// (non-ctx) equivalent for the parity test, and the expected phase count.
//
// TODO: this contract is `(ExecutionContext&, const MeshGL&) -> Manifold`
// only. When a 3rd ctx static factory lands with a non-MeshGL signature
// (LevelSet takes `(sdf, bounds, edgeLength)`, Hull takes a vector, etc.),
// either add a parallel typed suite for that shape or refactor to
// fixture-based traits where each `SmallValid`/`Empty`/`Malformed` is a
// trait method returning a Manifold. Until then the duplication cost of a
// per-factory cluster is lower than the generalization cost.
struct FromMeshGLTraits {
  static constexpr int kPhases = kPhasesPerFromMesh;
  static constexpr const char* kName = "FromMeshGL";
  static Manifold ViaCtx(ExecutionContext& ctx, const MeshGL& mesh) {
    return ctx.FromMeshGL(mesh);
  }
  static Manifold ViaPlain(const MeshGL& mesh) { return Manifold(mesh); }
};

struct SmoothTraits {
  static constexpr int kPhases = kPhasesPerSmooth;
  static constexpr const char* kName = "Smooth";
  static Manifold ViaCtx(ExecutionContext& ctx, const MeshGL& mesh) {
    return ctx.Smooth(mesh);
  }
  static Manifold ViaPlain(const MeshGL& mesh) {
    return Manifold::Smooth(mesh);
  }
};

// Compile-time contract check: each traits struct must expose `kPhases`
// (int), `kName` (const char*), and the two static factory methods.
template <typename T>
inline constexpr bool kIsStaticFactoryTraits =
    std::is_same_v<decltype(T::kPhases), const int> &&
    std::is_convertible_v<decltype(T::kName), const char*> &&
    std::is_invocable_r_v<Manifold, decltype(&T::ViaCtx), ExecutionContext&,
                          const MeshGL&> &&
    std::is_invocable_r_v<Manifold, decltype(&T::ViaPlain), const MeshGL&>;
static_assert(kIsStaticFactoryTraits<FromMeshGLTraits>);
static_assert(kIsStaticFactoryTraits<SmoothTraits>);

template <typename Factory>
class ExecutionContextStaticFactory : public ::testing::Test {};

class FactoryNameGenerator {
 public:
  template <typename T>
  static std::string GetName(int) {
    return T::kName;
  }
};

using StaticFactories = ::testing::Types<FromMeshGLTraits, SmoothTraits>;
TYPED_TEST_SUITE(ExecutionContextStaticFactory, StaticFactories,
                 FactoryNameGenerator);

TYPED_TEST(ExecutionContextStaticFactory, HappyPath) {
  ExecutionContext ctx;
  Manifold m = TypeParam::ViaCtx(ctx, TetGL());
  EXPECT_FALSE(m.IsEmpty());
  EXPECT_EQ(m.Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), TypeParam::kPhases);
  EXPECT_EQ(ctx.impl_->donePhases.load(), TypeParam::kPhases);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Equivalent geometry to the plain (non-ctx) call.
TYPED_TEST(ExecutionContextStaticFactory, MatchesPlain) {
  MeshGL mesh = Manifold::Sphere(1.0, 32).GetMeshGL();
  ExecutionContext ctx;
  Manifold viaCtx = TypeParam::ViaCtx(ctx, mesh);
  Manifold viaPlain = TypeParam::ViaPlain(mesh);
  EXPECT_EQ(viaCtx.Status(), viaPlain.Status());
  EXPECT_EQ(viaCtx.NumTri(), viaPlain.NumTri());
  EXPECT_EQ(viaCtx.NumVert(), viaPlain.NumVert());
}

// Pre-cancel beats the heavy path on well-formed input.
TYPED_TEST(ExecutionContextStaticFactory, CancelBeforeIngest) {
  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(TypeParam::ViaCtx(ctx, TetGL()).Status(),
            Manifold::Error::Cancelled);
}

// Pre-cancel beats the empty-input fast path (would otherwise be NoError).
TYPED_TEST(ExecutionContextStaticFactory,
           CancelBeforeEmptyInputWinsOverNoError) {
  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(TypeParam::ViaCtx(ctx, MeshGL{}).Status(),
            Manifold::Error::Cancelled);
}

// Pre-cancel beats validation errors.
TYPED_TEST(ExecutionContextStaticFactory,
           CancelBeforeMalformedWinsOverValidation) {
  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(
      TypeParam::ViaCtx(ctx, MalformedMissingPositionProperties()).Status(),
      Manifold::Error::Cancelled);
}

// Validation errors surface unchanged when no cancel is in flight; counters
// stay at 0/kPhases since no phase ran.
TYPED_TEST(ExecutionContextStaticFactory, ValidationErrorWithoutCancel) {
  ExecutionContext ctx;
  Manifold m = TypeParam::ViaCtx(ctx, MalformedMissingPositionProperties());
  EXPECT_EQ(m.Status(), Manifold::Error::MissingPositionProperties);
  EXPECT_EQ(ctx.impl_->donePhases.load(), 0);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), TypeParam::kPhases);
}

// Concurrent cancel mid-flight. Adaptive backoff guarantees we observe at
// least one Cancelled outcome; each Cancelled hit pins donePhases <
// totalPhases.
#if MANIFOLD_PAR == 1
TYPED_TEST(ExecutionContextStaticFactory, CancelConcurrent) {
  MeshGL mesh = Manifold::Sphere(1.0, 512).GetMeshGL();  // ~524k tris
  int cancelledHits = 0;
  auto sleep = std::chrono::microseconds(100);
  for (int attempt = 0; attempt < 12 && cancelledHits == 0; ++attempt) {
    ExecutionContext ctx;
    std::atomic<Manifold::Error> result{Manifold::Error::NoError};
    std::thread evalThread(
        [&] { result.store(TypeParam::ViaCtx(ctx, mesh).Status()); });
    std::this_thread::sleep_for(sleep);
    ctx.Cancel();
    evalThread.join();
    EXPECT_TRUE(result.load() == Manifold::Error::Cancelled ||
                result.load() == Manifold::Error::NoError);
    if (result.load() == Manifold::Error::Cancelled) {
      ++cancelledHits;
      EXPECT_LT(ctx.impl_->donePhases.load(), ctx.impl_->totalPhases.load());
      EXPECT_LT(ctx.Progress(), 1.0);
    }
    sleep *= 2;  // 100us, 200us, 400us, ... 204800us at attempt 11
  }
  EXPECT_GT(cancelledHits, 0);
}
#endif  // MANIFOLD_PAR == 1

// Factory then a Boolean on the same ctx: counters reset between.
TYPED_TEST(ExecutionContextStaticFactory, ReuseForBoolean) {
  ExecutionContext ctx;
  ASSERT_EQ(TypeParam::ViaCtx(ctx, TetGL()).Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), TypeParam::kPhases);

  Manifold u = Manifold::Cube() + Manifold::Sphere(0.5);
  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 1);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), kPhasesPerBoolean);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Two factory calls on the same ctx: counters reset between.
TYPED_TEST(ExecutionContextStaticFactory, ReuseForSecondCall) {
  ExecutionContext ctx;
  ASSERT_EQ(TypeParam::ViaCtx(ctx, TetGL()).Status(), Manifold::Error::NoError);
  ASSERT_EQ(ctx.impl_->donePhases.load(), TypeParam::kPhases);
  Manifold second = TypeParam::ViaCtx(ctx, TetGL());
  EXPECT_EQ(second.Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), TypeParam::kPhases);
  EXPECT_EQ(ctx.impl_->donePhases.load(), TypeParam::kPhases);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Sticky cancel survives across factory calls.
TYPED_TEST(ExecutionContextStaticFactory, StickyCancelAcrossCalls) {
  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(TypeParam::ViaCtx(ctx, TetGL()).Status(),
            Manifold::Error::Cancelled);
  EXPECT_EQ(TypeParam::ViaCtx(ctx, TetGL()).Status(),
            Manifold::Error::Cancelled);
}

// Smooth-only extras (outside the typed suite because no other factory
// takes a `sharpenedEdges` argument). Cancels mid-flight with a non-empty
// `sharpenedEdges` and asserts `NumTri() == 0` on Cancelled to lock in
// that the faceID-restoration loop in `MakeSmoothImpl` is a natural no-op
// after `MakeEmpty`.
#if MANIFOLD_PAR == 1
TEST(ExecutionContextSmoothExtras, CancelWithSharpenedEdges) {
  MeshGL mesh = Manifold::Sphere(1.0, 512).GetMeshGL();
  std::vector<Smoothness> sharpenedEdges = {{0, 0.0}, {3, 0.5}, {6, 0.0}};
  int cancelledHits = 0;
  auto sleep = std::chrono::microseconds(100);
  for (int attempt = 0; attempt < 12 && cancelledHits == 0; ++attempt) {
    ExecutionContext ctx;
    std::atomic<Manifold::Error> result{Manifold::Error::NoError};
    std::atomic<size_t> numTri{0};
    std::thread evalThread([&] {
      Manifold m = ctx.Smooth(mesh, sharpenedEdges);
      result.store(m.Status());
      numTri.store(m.NumTri());
    });
    std::this_thread::sleep_for(sleep);
    ctx.Cancel();
    evalThread.join();
    if (result.load() == Manifold::Error::Cancelled) {
      ++cancelledHits;
      EXPECT_EQ(numTri.load(), 0u);
    }
    sleep *= 2;
  }
  EXPECT_GT(cancelledHits, 0);
}
#endif  // MANIFOLD_PAR == 1
