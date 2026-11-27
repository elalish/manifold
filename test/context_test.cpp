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
#include <functional>
#include <thread>

#include "../src/execution_impl.h"
#include "../src/parallel.h"
#include "../src/vec.h"
#include "gtest/gtest.h"
#include "manifold/manifold.h"
#include "test.h"

using namespace manifold;

#if MANIFOLD_PAR == 1
// Runs `evalFn` on a worker thread, sleeps `delay`, fires `ctx.Cancel()`,
// then joins. An exception escaping a std::thread's entry function calls
// std::terminate() and aborts the whole process instead of failing just
// the one test, so this captures it with std::exception_ptr and rethrows
// on the caller's (test) thread where gtest can catch and report it.
Manifold::Error RunConcurrentEvalAndCancel(
    ExecutionContext& ctx, std::chrono::microseconds delay,
    const std::function<Manifold::Error()>& evalFn) {
  std::atomic<Manifold::Error> result{Manifold::Error::NoError};
  std::exception_ptr eptr;
  std::thread evalThread([&] {
    try {
      result.store(evalFn());
    } catch (...) {
      eptr = std::current_exception();
    }
  });
  std::this_thread::sleep_for(delay);
  ctx.Cancel();
  evalThread.join();
  if (eptr) std::rethrow_exception(eptr);
  return result.load();
}
#endif  // MANIFOLD_PAR == 1

// A CSG tree with N leaves reduces to 1 result in N-1 combinations.
TEST(Context, ExecutionContextProgress) {
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

TEST(Context, ManifoldMirrorRemainsDeferred) {
  const Manifold cube = Manifold::Cube();
  const Manifold mirrored =
      (cube + cube.Translate({0.5, 0, 0})).Mirror({1, 0, 0});

  ExecutionContext ctx;
  EXPECT_EQ(mirrored.WithContext(ctx).Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 1);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 1);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
  EXPECT_DOUBLE_EQ(mirrored.Volume(), 1.5);
  EXPECT_EQ(mirrored.BoundingBox(), Box({-1.5, 0, 0}, {0, 1, 1}));
}

TEST(Context, ManifoldMirrorCanBeCancelledBeforeEvaluation) {
  const Manifold cube = Manifold::Cube();
  const Manifold mirrored =
      (cube + cube.Translate({0.5, 0, 0})).Mirror({1, 0, 0});

  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(mirrored.WithContext(ctx).Status(), Manifold::Error::Cancelled);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 1);
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 0);
}

// Forcing evaluation first, then observing with a fresh context, should
// report no pending work: the Manifold is already a leaf.
TEST(Context, ExecutionContextAlreadyEvaluated) {
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
TEST(Context, ExecutionContextCancelBeforeEval) {
  std::vector<Manifold> items;
  for (int i = 0; i < 10; i++) {
    items.push_back(Manifold::Sphere(1.0, {32}).Translate(vec3(i * 0.5, 0, 0)));
  }
  Manifold u = Manifold::BatchBoolean(items, OpType::Add);

  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::Cancelled);
  // WithPrecision evaluates under the attached context.
  EXPECT_EQ(Manifold::BatchBoolean(items, OpType::Add)
                .WithContext(ctx)
                .WithPrecision(1e-3)
                .Status(),
            Manifold::Error::Cancelled);
  // So does GetPrecision.
  const Manifold lazy = Manifold::BatchBoolean(items, OpType::Add);
  lazy.WithContext(ctx).GetPrecision();
  EXPECT_EQ(lazy.Status(), Manifold::Error::Cancelled);
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
TEST(Context, ExecutionContextCancelConcurrent) {
  // Build a large enough tree that evaluation takes measurable time.
  std::vector<Manifold> items;
  for (int i = 0; i < 50; i++) {
    items.push_back(Manifold::Sphere(1.0, {64}).Translate(vec3(i * 0.3, 0, 0)));
  }
  Manifold u = Manifold::BatchBoolean(items, OpType::Add);

  ExecutionContext ctx;
  // Yield briefly so evaluation starts, then request cancel.
  Manifold::Error result =
      RunConcurrentEvalAndCancel(ctx, std::chrono::milliseconds(1),
                                 [&] { return u.WithContext(ctx).Status(); });

  // Cancel may have fired between ops or after the whole eval finished
  // (depending on timing). Either Cancelled (expected) or NoError (raced
  // past us) is acceptable.
  EXPECT_TRUE(result == Manifold::Error::Cancelled ||
              result == Manifold::Error::NoError);
  if (result == Manifold::Error::Cancelled) {
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
TEST(Context, ExecutionContextCancelMidBoolean) {
  // One big boolean — N-1 = 1 op, so outer-loop cancel only has a single
  // checkpoint. The work itself (two large spheres overlapping) takes long
  // enough that cancel fires reliably while Boolean3 is running.
  Manifold a = Manifold::Sphere(1.0, {256});
  Manifold b = Manifold::Sphere(1.0, {256}).Translate(vec3(0.5, 0, 0));
  Manifold u = a + b;

  ExecutionContext ctx;
  Manifold::Error result =
      RunConcurrentEvalAndCancel(ctx, std::chrono::milliseconds(1),
                                 [&] { return u.WithContext(ctx).Status(); });

  // Either outcome is acceptable. NoError: eval raced past the 1ms sleep
  // before cancel fired. Cancelled: caught by an inner Boolean3 phase
  // check or by the outer CsgOpNode::ToLeafNode check after a fast
  // SimpleBoolean — we can't cheaply distinguish those paths.
  EXPECT_TRUE(result == Manifold::Error::Cancelled ||
              result == Manifold::Error::NoError);
}
#endif  // MANIFOLD_PAR == 1

// Status() and Status(ctx) should return identical results when no cancel.
TEST(Context, ExecutionContextMatchesPlainStatus) {
  Manifold u = (Manifold::Cube(vec3(1), true) + Manifold::Sphere(1.0, {32})) -
               Manifold::Tetrahedron();

  // Build two equivalent manifolds (CsgOpNodes) to avoid caching effects.
  auto makeTree = [] {
    return (Manifold::Cube(vec3(1), true) + Manifold::Sphere(1.0, {32})) -
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
TEST(Context, ExecutionContextProgressSubtract) {
  // Tree shape: (cube + sphere) - tet - octahedron
  Manifold u = (Manifold::Cube(vec3(1), true) + Manifold::Sphere(1.0, {16})) -
               Manifold::Tetrahedron() - Manifold::Cube(vec3(0.5), true);

  ExecutionContext ctx;
  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 3);  // 4 leaves - 1
  EXPECT_EQ(ctx.impl_->doneBooleans.load(), 3);
}

// Reusing the same context for two sequential evaluations should reset
// doneBooleans so Progress is always in [0, 1].
TEST(Context, ExecutionContextReuse) {
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
TEST(Context, ExecutionContextCancelPermanent) {
  std::vector<Manifold> items;
  for (int i = 0; i < 30; i++) {
    items.push_back(Manifold::Sphere(1.0, {48}).Translate(vec3(i * 0.3, 0, 0)));
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
TEST(Context, ExecutionContextCancelStickyAcrossManifolds) {
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
TEST(Context, ExecutionContextFreshContextEscapesCancel) {
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
TEST(Context, ExecutionContextCancelSkippedOnLeaf) {
  Manifold cube = Manifold::Cube();  // CsgLeafNode from the start
  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(cube.WithContext(ctx).Status(), Manifold::Error::NoError);
}

// Progress is observable from another thread while evaluation runs.
// MANIFOLD_PAR guard: see note on ExecutionContextCancelConcurrent above.
#if MANIFOLD_PAR == 1
TEST(Context, ExecutionContextConcurrentProgress) {
  std::vector<Manifold> items;
  for (int i = 0; i < 30; i++) {
    items.push_back(Manifold::Sphere(1.0, {48}).Translate(vec3(i * 0.5, 0, 0)));
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

// Threads evaluate trees that share an unevaluated subtree, starting together.
// Thread 0 evaluates without a context, which collapses the shared nodes,
// thread 1 evaluates the shared root under a cancelled context, which marks it
// Cancelled, thread 2 evaluates the same root under its own context, and thread
// 3 counts the shared nodes under its own context.
// Any thread may see Cancelled from thread 1. This test is meant to be run
// under ThreadSanitizer.
TEST(Context, ExecutionContextSharedSubtreeConcurrent) {
  constexpr int kThreads = 4;
  for (int round = 0; round < 50; ++round) {
    Manifold shared = Manifold::Cube(vec3(2), true);
    const Manifold hole = Manifold::Cylinder(3, 0.3, -1, {12}, true);
    for (int i = 0; i < 4; ++i) {
      const Manifold h = hole.Translate(vec3(0.4 * (i - 1.5), 0, 0));
      shared = (shared - h) + h.Rotate(90, 0, 0).Translate(vec3(0, 0, 3));
    }

    std::atomic<int> ready{0};
    std::vector<Manifold::Error> status(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        Manifold m =
            t == 1 || t == 2 ? shared : shared.Translate(vec3(t, 0, 0));
        ExecutionContext ctx;
        if (t == 1) ctx.Cancel();
        ready++;
        while (ready.load() < kThreads) {
        }
        status[t] = t == 0 ? m.Status() : m.WithContext(ctx).Status();
      });
    }
    for (auto& thread : threads) thread.join();
    for (int t = 0; t < kThreads; ++t) {
      EXPECT_TRUE(status[t] == Manifold::Error::NoError ||
                  status[t] == Manifold::Error::Cancelled);
    }
  }
}

// Trivial Booleans (empty input, no intersection) early-return from
// Boolean3::Result before any phase() site executes. The PhaseBalance scope
// guard must still credit a full kPhasesPerBoolean phases on those returns
// so Progress() reaches 1.0 after a sequence of no-op evaluations.
TEST(Context, ExecutionContextProgressReachesOneOnTrivialBooleans) {
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
TEST(Context, ExecutionContextSubBooleanProgress) {
  // A single nontrivial Boolean with a real intersection: NumLeaves == 2 ->
  // totalBooleans == 1, so doneBooleans only flips from 0 to 1 at the very
  // end. Any observed Progress() > 0 mid-eval comes from sub-Boolean phase
  // advancement. Two overlapping spheres ensure the Boolean takes the full
  // path (no degenerate-input early return that would short-circuit phases).
  Manifold a = Manifold::Sphere(1.0, {256});
  Manifold b = Manifold::Sphere(1.0, {256}).Translate(vec3(0.5, 0, 0));
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

// A call with no Booleans left to evaluate leaves the ctx as its last
// evaluation left it, whether that evaluation finished or was cancelled.
TEST(Context, ExecutionContextQueryLeavesProgress) {
  using Query = std::function<void(const Manifold&)>;
  const std::vector<std::pair<const char*, Query>> queries = {
      {"Status", [](const Manifold& m) { m.Status(); }},
      {"GetPrecision", [](const Manifold& m) { m.GetPrecision(); }},
      {"WithPrecision", [](const Manifold& m) { m.WithPrecision(1e-3); }},
  };
  const auto counters = [](const ExecutionContext& ctx) {
    return std::vector{
        ctx.impl_->totalBooleans.load(), ctx.impl_->doneBooleans.load(),
        ctx.impl_->totalPhases.load(), ctx.impl_->donePhases.load()};
  };
  for (bool cancel : {false, true}) {
    for (const auto& [name, query] : queries) {
      SCOPED_TRACE(std::string(name) + (cancel ? " cancelled" : ""));
      std::vector<Manifold> items;
      for (int i = 0; i < 3; i++) {
        items.push_back(Manifold::Cube(vec3(1)).Translate(vec3(i * 0.5)));
      }
      const Manifold tree = Manifold::BatchBoolean(items, OpType::Add);
      const Manifold evaluated =
          Manifold::Cube() + Manifold::Cube().Translate(vec3(0.5));
      evaluated.Status();

      ExecutionContext ctx;
      if (cancel) ctx.Cancel();
      // Evaluating this copy leaves tree holding the op node with its result
      // cached.
      const Manifold attached = tree.WithContext(ctx);
      attached.Status();
      const std::vector<int> before = counters(ctx);
      ASSERT_EQ(before[0], 2);
      EXPECT_EQ(ctx.Progress() < 1.0, cancel);

      std::vector<Manifold> noWork = {attached, tree, evaluated, Manifold(),
                                      Manifold::Cube()};
      // After a finished evaluation, a Translate of tree is an unevaluated op
      // node over the single result. A cancelled one leaves the Booleans.
      if (!cancel) noWork.push_back(tree.Translate(vec3(1)));
      for (const Manifold& m : noWork) {
        query(m.WithContext(ctx));
        EXPECT_EQ(counters(ctx), before);
        EXPECT_EQ(ctx.Cancelled(), cancel);
      }
    }
  }
}

// ExecutionContext copies share state (pimpl semantics): one thread
// evaluates, another thread holds a copy and observes the same progress.
TEST(Context, ExecutionContextCopyShareState) {
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
TEST(Context, ExecutionContextMoveSemantics) {
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
TEST(Context, ExecutionContextNoWorkNeeded) {
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
TEST(Context, ExecutionContextProgressInvariant) {
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
TEST(Context, ForEachCtxSkipsWhenCancelled) {
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
TEST(Context, ForEachCtxNullCtxPassesThrough) {
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
TEST(Context, ForEachCtxCancelMidSeqLoop) {
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
TEST(Context, ParallelStableSortStability) {
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
// "No attached ctx" is asserted behaviorally: cancel the ctx, then run an
// eager op on the result. A result with no error status reports Cancelled
// only if it carries the ctx.

// Raw copy preserves the attachment: ctx-attached on the source means
// Status on the copy observes the same ctx.
TEST(Context, ManifoldContextRawCopyPreservesAttachment) {
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
TEST(Context, ManifoldContextAssignmentPreservesAttachment) {
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
TEST(Context, ManifoldContextDeferredOpsDropAttachment) {
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
TEST(Context, ManifoldContextStaticFactoriesDropAttachment) {
  ExecutionContext ctx;
  ctx.Cancel();
  std::vector<Manifold> items = {
      (Manifold::Cube(vec3(1), true) + Manifold::Cube(vec3(1), true))
          .WithContext(ctx),
      Manifold::Cube(vec3(1), true).Translate(vec3(2, 0, 0)).WithContext(ctx)};

  // Hull runs first, while the first input is still lazy.
  EXPECT_EQ(Manifold::Hull(items).Status(), Manifold::Error::NoError);
  EXPECT_EQ(Manifold::BatchBoolean(items, OpType::Add).Status(),
            Manifold::Error::NoError);
}

static Manifold FreshHullOperand() {
  return Manifold::Cube(vec3(1), true) +
         Manifold::Cube(vec3(1), true).Translate(vec3(2, 0, 0));
}

TEST(Context, ManifoldContextStaticHullLeavesInputProgress) {
  for (bool lazy : {false, true}) {
    SCOPED_TRACE(lazy);
    ExecutionContext ctx;
    ctx.impl_->totalBooleans.store(7);
    ctx.impl_->doneBooleans.store(3);
    ctx.impl_->totalPhases.store(37);
    ctx.impl_->donePhases.store(11);
    const Manifold source =
        lazy ? FreshHullOperand() : Manifold::Cube(vec3(1), true);
    const Manifold hull =
        Manifold::Hull(std::vector<Manifold>{source.WithContext(ctx)});
    EXPECT_EQ(hull.Status(), Manifold::Error::NoError);
    EXPECT_NEAR(hull.Volume(), lazy ? 3.0 : 1.0, 1e-10);
    EXPECT_EQ(ctx.impl_->totalBooleans.load(), 7);
    EXPECT_EQ(ctx.impl_->doneBooleans.load(), 3);
    EXPECT_EQ(ctx.impl_->totalPhases.load(), 37);
    EXPECT_EQ(ctx.impl_->donePhases.load(), 11);
    EXPECT_DOUBLE_EQ(ctx.Progress(), 11.0 / 37.0);
  }
}

TEST(Context, ManifoldContextStaticHullPreservesCachedErrors) {
  ExecutionContext ctx;
  ctx.Cancel();
  const Manifold errored = FreshHullOperand().WithContext(ctx);
  ASSERT_EQ(errored.Status(), Manifold::Error::Cancelled);
  // Once materialized, cancellation is an actual operand error, not merely
  // an attached context the static factory must ignore.
  const Manifold hull = Manifold::Hull(std::vector<Manifold>{errored});
  EXPECT_EQ(hull.Status(), Manifold::Error::Cancelled);
  EXPECT_TRUE(hull.IsEmpty());
}

// no-arg Status() on a ctx-attached Manifold observes the attached ctx:
// counters update exactly as if Status had been passed an explicit ctx.
TEST(Context, ManifoldContextStatusObservesAttached) {
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
TEST(Context, ManifoldContextDeferredTreeRootObserves) {
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
TEST(Context, ManifoldContextCancelMidRefine) {
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
TEST(Context, ManifoldContextRefineDropsAttachmentFromResult) {
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

// True if `m` still carries `ctx`: Minkowski on it observes the ctx's cancel,
// even when `m` is empty. Cancels `ctx`. A Manifold with an error status
// never uses its ctx, so for one the probe reads its own status instead.
static bool CarriesContext(const Manifold& m, ExecutionContext& ctx) {
  ctx.Cancel();
  return m.MinkowskiSum(Manifold()).Status() == Manifold::Error::Cancelled;
}

// Every op that returns a new Manifold drops the attachment, whatever its
// inputs, including compound assignment, which assigns a derived result.
// Each case attaches a ctx, runs the op, and checks that no result still
// observes it. The attached Manifold is used as either operand.
TEST(Context, ManifoldContextEveryOpDropsAttachment) {
  using Op = std::function<std::vector<Manifold>(const Manifold&)>;
  const Manifold other = Manifold::Cube(vec3(1), true).Translate({5, 0, 0});
  const Manifold empty;
  const auto pair = [](std::pair<Manifold, Manifold> p) {
    return std::vector{p.first, p.second};
  };
  const std::vector<std::pair<const char*, Op>> ops = {
      {"Decompose", [](const Manifold& m) { return m.Decompose(); }},
      {"AsOriginal",
       [](const Manifold& m) { return std::vector{m.AsOriginal()}; }},
      {"Translate",
       [](const Manifold& m) { return std::vector{m.Translate(vec3(0.0))}; }},
      {"Scale",
       [](const Manifold& m) { return std::vector{m.Scale(vec3(1))}; }},
      {"Rotate", [](const Manifold& m) { return std::vector{m.Rotate(0)}; }},
      {"Mirror",
       [](const Manifold& m) { return std::vector{m.Mirror({1, 0, 0})}; }},
      {"MirrorZero",
       [](const Manifold& m) { return std::vector{m.Mirror(vec3(0.0))}; }},
      {"Transform",
       [](const Manifold& m) {
         return std::vector{m.Transform(la::identity)};
       }},
      {"Warp",
       [](const Manifold& m) { return std::vector{m.Warp([](vec3&) {})}; }},
      {"WarpBatch",
       [](const Manifold& m) {
         return std::vector{m.WarpBatch([](VecView<vec3>) {})};
       }},
      {"Simplify", [](const Manifold& m) { return std::vector{m.Simplify()}; }},
      {"RemoveDegenerates",
       [](const Manifold& m) { return std::vector{m.RemoveDegenerates()}; }},
      {"Boolean",
       [&](const Manifold& m) {
         return std::vector{m.Boolean(other, OpType::Add)};
       }},
      {"BooleanOther",
       [&](const Manifold& m) {
         return std::vector{other.Boolean(m, OpType::Add)};
       }},
      {"AddEmpty", [&](const Manifold& m) { return std::vector{m + empty}; }},
      {"EmptyAdd", [&](const Manifold& m) { return std::vector{empty + m}; }},
      {"SubtractEmpty",
       [&](const Manifold& m) { return std::vector{m - empty}; }},
      {"IntersectEmpty",
       [&](const Manifold& m) { return std::vector{m ^ empty}; }},
      {"AddAssign",
       [&](const Manifold& m) {
         Manifold r = m;
         r += other;
         return std::vector{r};
       }},
      {"SubtractAssign",
       [&](const Manifold& m) {
         Manifold r = m;
         r -= other;
         return std::vector{r};
       }},
      {"IntersectAssign",
       [&](const Manifold& m) {
         Manifold r = m;
         r ^= other;
         return std::vector{r};
       }},
      {"BatchBooleanOne",
       [](const Manifold& m) {
         return std::vector{Manifold::BatchBoolean({m}, OpType::Add)};
       }},
      {"BatchBooleanSubtractOne",
       [](const Manifold& m) {
         return std::vector{Manifold::BatchBoolean({m}, OpType::Subtract)};
       }},
      {"BatchBooleanEmpty",
       [&](const Manifold& m) {
         return std::vector{Manifold::BatchBoolean({m, empty}, OpType::Add)};
       }},
      {"Split", [&](const Manifold& m) { return pair(m.Split(other)); }},
      {"SplitEmpty", [&](const Manifold& m) { return pair(m.Split(empty)); }},
      {"SplitByPlane",
       [&](const Manifold& m) { return pair(m.SplitByPlane({1, 0, 0}, 0)); }},
      {"SplitByPlaneAway",
       [&](const Manifold& m) { return pair(m.SplitByPlane({1, 0, 0}, 100)); }},
      {"TrimByPlaneAway",
       [](const Manifold& m) {
         return std::vector{m.TrimByPlane({1, 0, 0}, -100)};
       }},
      {"MinkowskiSumOther",
       [&](const Manifold& m) { return std::vector{other.MinkowskiSum(m)}; }},
      {"MinkowskiDifferenceOther",
       [&](const Manifold& m) {
         return std::vector{other.MinkowskiDifference(m)};
       }},
      {"SetProperties",
       [](const Manifold& m) {
         return std::vector{m.SetProperties(0, nullptr)};
       }},
      {"CalculateCurvature",
       [](const Manifold& m) {
         return std::vector{m.CalculateCurvature(-1, -1)};
       }},
      {"CalculateNormals",
       [](const Manifold& m) { return std::vector{m.CalculateNormals()}; }},
      {"Refine", [](const Manifold& m) { return std::vector{m.Refine(1)}; }},
      {"RefineToLength",
       [](const Manifold& m) { return std::vector{m.RefineToLength(100)}; }},
      {"RefineToTolerance",
       [](const Manifold& m) { return std::vector{m.RefineToTolerance(100)}; }},
      {"SmoothByNormals",
       [](const Manifold& m) { return std::vector{m.SmoothByNormals()}; }},
      {"Hull", [](const Manifold& m) { return std::vector{m.Hull()}; }},
      {"WithPrecision",
       [](const Manifold& m) { return std::vector{m.WithPrecision(1e-3)}; }},
      {"HullVector",
       [](const Manifold& m) {
         return std::vector{Manifold::Hull(std::vector{m})};
       }},
      {"MinkowskiSum",
       [&](const Manifold& m) { return std::vector{m.MinkowskiSum(other)}; }},
      {"MinkowskiDifference",
       [&](const Manifold& m) {
         return std::vector{m.MinkowskiDifference(other)};
       }},
  };

  // An empty Manifold, an evaluated leaf, a lazy tree that is one piece, and
  // a lazy tree of two disjoint pieces.
  const auto makeSource = [](int kind) {
    const Manifold cube = Manifold::Cube(vec3(1), true);
    if (kind == 0) return Manifold();
    if (kind == 1) return cube;
    return cube + cube.Translate({kind == 2 ? 0.5 : 2.0, 0, 0});
  };
  for (int kind : {0, 1, 2, 3}) {
    for (const auto& [name, op] : ops) {
      SCOPED_TRACE(std::string(name) + " source " + std::to_string(kind));
      ExecutionContext ctx;
      for (const Manifold& result : op(makeSource(kind).WithContext(ctx))) {
        EXPECT_FALSE(CarriesContext(result, ctx));
      }
      // The probe can detect an attachment.
      EXPECT_TRUE(CarriesContext(makeSource(kind).WithContext(ctx), ctx));
    }
  }
}

// Eager Hull observes attached ctx -- a pre-cancelled ctx aborts the
// post-pass that runs after QuickHull's buildMesh, returning a Cancelled
// result rather than a finished hull.
TEST(Context, ManifoldContextCancelMidHull) {
  ExecutionContext ctx;
  ctx.Cancel();
  // Non-trivial input so the post-buildMesh path is reached.
  Manifold sphere = Manifold::Sphere(1.0, {64}).WithContext(ctx);
  Manifold hull = sphere.Hull();
  EXPECT_EQ(hull.Status(), Manifold::Error::Cancelled);
}

// Eager Minkowski observes attached ctx -- a pre-cancelled ctx aborts
// between the hull/Boolean phases. Cancel granularity is "one batch" of
// per-face hulls.
TEST(Context, ManifoldContextCancelMidMinkowski) {
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
TEST(Context, ManifoldContextCancelConcurrentHull) {
  // High-resolution sphere so QuickHull + SortGeometry have measurable
  // work to do.
  Manifold sphere = Manifold::Sphere(1.0, {256});

  ExecutionContext ctx;
  Manifold::Error result = RunConcurrentEvalAndCancel(
      ctx, std::chrono::milliseconds(1),
      [&] { return sphere.WithContext(ctx).Hull().Status(); });

  // Either NoError (eval raced past the 1ms sleep) or Cancelled
  // (caught at one of the Hull cancel checkpoints).
  EXPECT_TRUE(result == Manifold::Error::Cancelled ||
              result == Manifold::Error::NoError);
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
TEST(Context, ManifoldContextCancelConcurrentMinkowski) {
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
  Manifold::Error result = RunConcurrentEvalAndCancel(
      ctx, std::chrono::milliseconds(1),
      [&] { return nonConvex.WithContext(ctx).MinkowskiSum(other).Status(); });

  // Either NoError (eval raced past the 1ms sleep) or Cancelled
  // (caught mid-Boolean inside Minkowski) is acceptable.
  EXPECT_TRUE(result == Manifold::Error::Cancelled ||
              result == Manifold::Error::NoError);
}
#endif  // MANIFOLD_PAR == 1

// Cancellation requested on the attached ctx after Status() begins (or
// before, in the deferred-tree-root idiom) aborts evaluation.
TEST(Context, ManifoldContextCancelMidEval) {
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

// FromMeshGL: ctx-aware analog of the `Manifold(MeshGL)` ctor.
TEST(Context, MeshGLHappyPath) {
  ExecutionContext ctx;
  Manifold tet = ctx.FromMeshGL(TetGL());
  EXPECT_FALSE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), kPhasesPerFromMesh);
  EXPECT_EQ(ctx.impl_->donePhases.load(), kPhasesPerFromMesh);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Equivalent geometry to the plain ctor.
TEST(Context, MeshGLMatchesCtor) {
  MeshGL mesh = Manifold::Sphere(1.0, {32}).GetMeshGL();
  ExecutionContext ctx;
  Manifold viaCtx = ctx.FromMeshGL(mesh);
  Manifold viaCtor(mesh);
  EXPECT_EQ(viaCtx.Status(), viaCtor.Status());
  EXPECT_EQ(viaCtx.NumTri(), viaCtor.NumTri());
  EXPECT_EQ(viaCtx.NumVert(), viaCtor.NumVert());
}

// Pre-cancel beats the heavy path on well-formed input.
TEST(Context, MeshGLCancelBeforeIngest) {
  ExecutionContext ctx;
  ctx.Cancel();
  Manifold m = ctx.FromMeshGL(TetGL());
  EXPECT_EQ(m.Status(), Manifold::Error::Cancelled);
}

// Empty input is finished work: no error and full progress.
TEST(Context, EmptyInputFactoriesComplete) {
  ExecutionContext ctx;
  const auto expectComplete = [&ctx](const Manifold& m) {
    EXPECT_EQ(m.Status(), Manifold::Error::NoError);
    EXPECT_TRUE(m.IsEmpty());
    EXPECT_EQ(ctx.Progress(), 1.0);
  };
  expectComplete(ctx.FromMeshGL(MeshGL{}));
  expectComplete(ctx.FromMeshGL(MeshGL64{}));
  expectComplete(
      ctx.LevelSet([](vec3 p) { return 1 - la::length(p); }, Box(), 0.5));
}

// Pre-cancel beats the empty-input fast path (would otherwise be NoError).
TEST(Context, MeshGLCancelBeforeEmptyInputWinsOverNoError) {
  ExecutionContext ctx;
  ctx.Cancel();
  Manifold m = ctx.FromMeshGL(MeshGL{});
  EXPECT_EQ(m.Status(), Manifold::Error::Cancelled);
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

// Pre-cancel beats validation errors.
TEST(Context, MeshGLCancelBeforeMalformedWinsOverValidation) {
  ExecutionContext ctx;
  ctx.Cancel();
  Manifold m = ctx.FromMeshGL(MalformedMissingPositionProperties());
  EXPECT_EQ(m.Status(), Manifold::Error::Cancelled);
}

// Validation errors surface unchanged when no cancel is in flight;
// counters stay at 0/kPhasesPerFromMesh since no phase ran.
TEST(Context, MeshGLValidationErrorWithoutCancel) {
  ExecutionContext ctx;
  Manifold m = ctx.FromMeshGL(MalformedMissingPositionProperties());
  EXPECT_EQ(m.Status(), Manifold::Error::MissingPositionProperties);
  EXPECT_EQ(ctx.impl_->donePhases.load(), 0);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), kPhasesPerFromMesh);
}

// Concurrent cancel mid-ingest. Adaptive backoff guarantees we observe
// at least one Cancelled outcome (otherwise the cancel path is dead);
// each Cancelled hit also pins donePhases < totalPhases.
#if MANIFOLD_PAR == 1
TEST(Context, MeshGLCancelConcurrent) {
  MeshGL mesh = Manifold::Sphere(1.0, {512}).GetMeshGL();  // ~524k tris
  int cancelledHits = 0;
  auto sleep = std::chrono::microseconds(100);
  for (int attempt = 0; attempt < 12 && cancelledHits == 0; ++attempt) {
    ExecutionContext ctx;
    Manifold::Error result = RunConcurrentEvalAndCancel(
        ctx, sleep, [&] { return ctx.FromMeshGL(mesh).Status(); });
    EXPECT_TRUE(result == Manifold::Error::Cancelled ||
                result == Manifold::Error::NoError);
    if (result == Manifold::Error::Cancelled) {
      ++cancelledHits;
      EXPECT_LT(ctx.impl_->donePhases.load(), ctx.impl_->totalPhases.load());
      EXPECT_LT(ctx.Progress(), 1.0);
    }
    sleep *= 2;  // 100us, 200us, 400us, ... 204800us at attempt 11
  }
  EXPECT_GT(cancelledHits, 0);
}
#endif  // MANIFOLD_PAR == 1

// Two threads ingesting through one shared ctx. Progress is undefined in this
// case, but the calls must not race. Meaningful under TSan, which builds with
// MANIFOLD_TIMING so the phase timing is compiled in.
#if MANIFOLD_PAR == 1
TEST(Context, MeshGLSharedContextConcurrent) {
  MeshGL mesh = Manifold::Sphere(1.0, {64}).GetMeshGL();
  ExecutionContext ctx;
  auto ingest = [&] {
    for (int i = 0; i < 20; ++i) {
      EXPECT_EQ(ctx.FromMeshGL(mesh).Status(), Manifold::Error::NoError);
    }
  };
  std::thread a(ingest), b(ingest);
  a.join();
  b.join();
}
#endif  // MANIFOLD_PAR == 1

// FromMeshGL then a Boolean on the same ctx: counters reset between.
TEST(Context, MeshGLReuseForBoolean) {
  ExecutionContext ctx;
  Manifold tet = ctx.FromMeshGL(TetGL());
  ASSERT_EQ(tet.Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), kPhasesPerFromMesh);

  Manifold u = Manifold::Cube() + Manifold::Sphere(0.5);
  EXPECT_EQ(u.WithContext(ctx).Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalBooleans.load(), 1);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), kPhasesPerBoolean);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Two FromMeshGL calls on the same ctx: counters reset between.
TEST(Context, MeshGLReuseForSecondFromMeshGL) {
  ExecutionContext ctx;
  ASSERT_EQ(ctx.FromMeshGL(TetGL()).Status(), Manifold::Error::NoError);
  ASSERT_EQ(ctx.impl_->donePhases.load(), kPhasesPerFromMesh);
  Manifold second = ctx.FromMeshGL(TetGL());
  EXPECT_EQ(second.Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), kPhasesPerFromMesh);
  EXPECT_EQ(ctx.impl_->donePhases.load(), kPhasesPerFromMesh);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Sticky cancel survives across FromMeshGL calls.
TEST(Context, MeshGLStickyCancelAcrossCalls) {
  ExecutionContext ctx;
  ctx.Cancel();
  EXPECT_EQ(ctx.FromMeshGL(TetGL()).Status(), Manifold::Error::Cancelled);
  EXPECT_EQ(ctx.FromMeshGL(TetGL()).Status(), Manifold::Error::Cancelled);
}

// LevelSet: ctx-aware analog of `Manifold::LevelSet`. The shared
// counter/reuse/validation machinery is the same as FromMeshGL and is covered
// by that cluster above, so here we only test LevelSet's unique paths: the
// happy path, parity with the plain factory, the entry-time cancel gate in
// CreateLevelSet, and concurrent cancel through the dominant voxel-sampling
// phase.
TEST(Context, LevelSetHappyPath) {
  // Sphere SDF: positive inside, negative outside.
  auto sphere = [](vec3 p) { return 1.0 - la::length(p); };
  ExecutionContext ctx;
  Manifold m = ctx.LevelSet(sphere, {vec3(-1.1), vec3(1.1)}, 0.1);
  EXPECT_FALSE(m.IsEmpty());
  EXPECT_EQ(m.Status(), Manifold::Error::NoError);
  EXPECT_EQ(ctx.impl_->totalPhases.load(), kPhasesPerLevelSet);
  EXPECT_EQ(ctx.impl_->donePhases.load(), kPhasesPerLevelSet);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

TEST(Context, LevelSetCoarseGrid) {
  ExecutionContext ctx;
  bool finite = true;
  const Manifold box = ctx.LevelSet(
      [&](vec3 p) {
        finite &= la::all(la::isfinite(p));
        return 1.0;
      },
      {vec3(-1.0), vec3(1.0)}, 3.0, 0.0, -1.0, false);
  EXPECT_TRUE(finite);
  EXPECT_EQ(box.Status(), Manifold::Error::NoError);
  EXPECT_NEAR(box.Volume(), 8.0, 1e-10);
  EXPECT_DOUBLE_EQ(ctx.Progress(), 1.0);
}

// Equivalent geometry to the plain factory. (Named MatchesFactory, not
// MatchesCtor - LevelSet has no constructor form.)
TEST(Context, LevelSetMatchesFactory) {
  auto sphere = [](vec3 p) { return 1.0 - la::length(p); };
  const Box bounds = {vec3(-1.1), vec3(1.1)};
  const double edgeLength = 0.1;
  ExecutionContext ctx;
  Manifold viaCtx = ctx.LevelSet(sphere, bounds, edgeLength);
  Manifold viaFactory = Manifold::LevelSet(sphere, bounds, edgeLength);
  EXPECT_EQ(viaCtx.Status(), viaFactory.Status());
  EXPECT_EQ(viaCtx.NumTri(), viaFactory.NumTri());
  EXPECT_EQ(viaCtx.NumVert(), viaFactory.NumVert());
}

// Pre-cancel hits CreateLevelSet's entry gate and returns before allocating or
// sampling the voxel grid - LevelSet's analog of Smooth's CancelBeforeIngest.
TEST(Context, LevelSetCancelBeforeSampling) {
  auto sphere = [](vec3 p) { return 1.0 - la::length(p); };
  ExecutionContext ctx;
  ctx.Cancel();
  Manifold m = ctx.LevelSet(sphere, {vec3(-1.1), vec3(1.1)}, 0.1);
  EXPECT_EQ(m.Status(), Manifold::Error::Cancelled);
}

// Concurrent cancel mid-sampling. A deliberately slow sdf (a few us per
// evaluation) over a largish grid keeps the voxel-sampling phase running long
// enough that cancel reliably lands before the first phase boundary. Adaptive
// backoff guarantees at least one Cancelled outcome; each pins donePhases <
// totalPhases.
#if MANIFOLD_PAR == 1
TEST(Context, LevelSetCancelConcurrent) {
  auto slowSphere = [](vec3 p) {
    std::this_thread::sleep_for(std::chrono::microseconds(5));
    return 1.0 - la::length(p);
  };
  int cancelledHits = 0;
  auto sleep = std::chrono::microseconds(100);
  for (int attempt = 0; attempt < 12 && cancelledHits == 0; ++attempt) {
    ExecutionContext ctx;
    Manifold::Error result = RunConcurrentEvalAndCancel(ctx, sleep, [&] {
      return ctx.LevelSet(slowSphere, {vec3(-1), vec3(1)}, 0.05).Status();
    });
    EXPECT_TRUE(result == Manifold::Error::Cancelled ||
                result == Manifold::Error::NoError);
    if (result == Manifold::Error::Cancelled) {
      ++cancelledHits;
      EXPECT_LT(ctx.impl_->donePhases.load(), ctx.impl_->totalPhases.load());
      EXPECT_LT(ctx.Progress(), 1.0);
    }
    sleep *= 2;
  }
  EXPECT_GT(cancelledHits, 0);
}
#endif  // MANIFOLD_PAR == 1
