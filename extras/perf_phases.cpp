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
//
// Prints per-phase Boolean3 wall-clock timings on representative workloads.
// Requires MANIFOLD_TIMING=ON (or MANIFOLD_DEBUG=ON) so Timer::Print fires.
// Used for cancel-latency analysis and for identifying dominant phases.
//
// With --progress[=SEC], runs each op with a live ExecutionContext,
// prints progress every SEC seconds (default 5), and installs a SIGINT
// handler so ^C cancels the running op and reports cancel latency.
// First ^C requests cancel; second ^C force-exits. Useful as an
// interactive demo of ExecutionContext cancellation.

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#include "manifold/manifold.h"
#include "samples.h"

using namespace manifold;

namespace {

// Global state for the SIGINT handler (only used in interactive mode).
std::atomic<int> g_interrupts{0};

constexpr char kCancelMsg[] =
    "\n^C received, cancelling... (press ^C again to force exit)\n";
constexpr char kForceMsg[] = "\n^C x2, forcing exit.\n";

void SigintHandler(int) {
  int prior = g_interrupts.fetch_add(1, std::memory_order_relaxed);
  if (prior == 0) {
    // write(2) is async-signal-safe; std::cerr is not.
    (void)write(STDERR_FILENO, kCancelMsg, sizeof(kCancelMsg) - 1);
  } else {
    (void)write(STDERR_FILENO, kForceMsg, sizeof(kForceMsg) - 1);
    _exit(130);  // 128 + SIGINT
  }
}

// Polls g_interrupts at 100ms granularity so cancel fires promptly even
// when display_interval is large. Prints progress every display_interval
// seconds. Records t_cancel (the moment Cancel() is called) via atomic,
// so the main thread can compute cancel latency after Status() returns.
void ProgressLoop(ExecutionContext& ctx, double display_interval,
                  std::atomic<bool>& running,
                  std::chrono::steady_clock::time_point t_start,
                  std::atomic<int64_t>& t_cancel_ns) {
  using clock = std::chrono::steady_clock;
  const auto poll = std::chrono::milliseconds(100);
  auto next_display =
      t_start + std::chrono::duration_cast<clock::duration>(
                    std::chrono::duration<double>(display_interval));
  bool cancel_sent = false;

  while (running.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(poll);

    if (!cancel_sent && g_interrupts.load(std::memory_order_relaxed) > 0) {
      auto now = clock::now();
      t_cancel_ns.store(
          std::chrono::duration_cast<std::chrono::nanoseconds>(now - t_start)
              .count(),
          std::memory_order_relaxed);
      ctx.Cancel();
      cancel_sent = true;
    }

    auto now = clock::now();
    if (now >= next_display) {
      double elapsed = std::chrono::duration<double>(now - t_start).count();
      std::cerr << "[progress] elapsed " << std::fixed << std::setprecision(1)
                << elapsed << "s, " << static_cast<int>(ctx.Progress() * 100)
                << "%" << std::endl;
      next_display = now + std::chrono::duration_cast<clock::duration>(
                               std::chrono::duration<double>(display_interval));
    }
  }
}

// Returns true if the user cancelled (caller should stop iterating).
bool BenchOne(const std::string& workload, const char* op_name,
              Manifold (*apply)(const Manifold&, const Manifold&),
              const Manifold& a, const Manifold& b, double progress_interval) {
  std::cout << "=== " << workload << ": " << op_name
            << ", nTri(a) = " << a.NumTri() << " ===" << std::endl;

  auto t_start = std::chrono::steady_clock::now();
  Manifold result = apply(a, b);

  if (progress_interval <= 0) {
    // Non-interactive: no ctx, preserves existing perfPhases behavior.
    result.NumTri();
    auto t_end = std::chrono::steady_clock::now();
    std::cout << "total = "
              << std::chrono::duration<double>(t_end - t_start).count()
              << " sec\n"
              << std::endl;
    return false;
  }

  // Interactive: create ctx, spawn progress thread, call Status(ctx).
  ExecutionContext ctx;
  std::atomic<bool> running{true};
  std::atomic<int64_t> t_cancel_ns{-1};

  std::thread progress(ProgressLoop, std::ref(ctx), progress_interval,
                       std::ref(running), t_start, std::ref(t_cancel_ns));

  Manifold::Error status = result.Status(ctx);
  auto t_end = std::chrono::steady_clock::now();

  running.store(false, std::memory_order_relaxed);
  progress.join();

  std::cout << "total = "
            << std::chrono::duration<double>(t_end - t_start).count()
            << " sec, status = " << static_cast<int>(status) << std::endl;

  if (status == Manifold::Error::Cancelled) {
    int64_t cancel_ns = t_cancel_ns.load(std::memory_order_relaxed);
    if (cancel_ns >= 0) {
      double cancel_latency_ms =
          (std::chrono::duration<double>(t_end - t_start).count() -
           cancel_ns * 1e-9) *
          1000.0;
      std::cout << "cancel latency = " << std::fixed << std::setprecision(2)
                << cancel_latency_ms << " ms (Cancel() -> Status returned)"
                << std::endl;
    }
    std::cout << std::endl;
    return true;
  }
  std::cout << std::endl;
  return false;
}

// Runs one boolean op of each kind (Add, Subtract, Intersect) against the
// same input pair. Returns true if the user cancelled.
bool BenchAll(const std::string& workload, const Manifold& a, const Manifold& b,
              double progress_interval) {
  struct Op {
    const char* name;
    Manifold (*apply)(const Manifold&, const Manifold&);
  };
  const Op ops[] = {
      {"add", [](const Manifold& x, const Manifold& y) { return x + y; }},
      {"subtract", [](const Manifold& x, const Manifold& y) { return x - y; }},
      {"intersect", [](const Manifold& x, const Manifold& y) { return x ^ y; }},
  };
  for (const auto& op : ops) {
    if (BenchOne(workload, op.name, op.apply, a, b, progress_interval)) {
      return true;
    }
  }
  return false;
}

void PrintHelp(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--progress[=SECONDS]] [-h|--help]\n"
            << "\n"
            << "  --progress[=SECONDS]  Interactive demo mode. Runs each op\n"
            << "                        with an ExecutionContext, prints\n"
            << "                        progress every SECONDS (default 5),\n"
            << "                        and installs a SIGINT handler.\n"
            << "                        First ^C cancels the running op and\n"
            << "                        reports cancel latency. Second ^C\n"
            << "                        forces exit.\n"
            << "  -h, --help            Show this help.\n";
}

}  // namespace

int main(int argc, char** argv) {
  double progress_interval = 0;  // 0 = non-interactive

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      PrintHelp(argv[0]);
      return 0;
    } else if (arg == "--progress") {
      progress_interval = 5.0;
    } else if (arg.rfind("--progress=", 0) == 0) {
      try {
        progress_interval = std::stod(arg.substr(std::strlen("--progress=")));
      } catch (...) {
        std::cerr << "Invalid --progress value: " << arg << std::endl;
        return 2;
      }
      if (progress_interval <= 0) {
        std::cerr << "--progress requires a positive number of seconds."
                  << std::endl;
        return 2;
      }
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      PrintHelp(argv[0]);
      return 2;
    }
  }

  ManifoldParams().verbose = 2;

  if (progress_interval > 0) {
    std::signal(SIGINT, SigintHandler);
    std::cerr << "Interactive mode: progress every " << progress_interval
              << "s. Press ^C to cancel the running op (^C again to force "
                 "exit)."
              << std::endl;
  }

  // Sphere at escalating scales — uniform geometry, single boolean per op
  // so Timer output is readable.
  for (int i = 0; i < 8; ++i) {
    const int segments = (8 << i) * 4;
    Manifold sphere = Manifold::Sphere(1, segments);
    Manifold sphere2 = sphere.Translate(vec3(0.5));
    if (BenchAll("Sphere(" + std::to_string(segments) + ")", sphere, sphere2,
                 progress_interval)) {
      return 0;
    }
  }

  // Triangulator-heavy: Menger sponge. Forcing sponge.NumTri() first
  // separates the sponge's internal CSG construction from the measured
  // ops, so `total` reflects just each boolean.
  for (int level : {3, 4}) {
    Manifold sponge = MengerSponge(level);
    Manifold sponge2 = sponge.Translate(vec3(0.3));
    sponge.NumTri();  // force construction before timing the ops
    if (BenchAll("MengerSponge(" + std::to_string(level) + ")", sponge, sponge2,
                 progress_interval)) {
      return 0;
    }
  }
}
