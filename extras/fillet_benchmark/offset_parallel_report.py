#!/usr/bin/env python3
"""Report before/after scaling and deterministic output for offset filleting."""
import csv
import sys
from pathlib import Path


def read(path):
    rows = list(csv.DictReader(Path(path).open()))
    assert all(r["exact_match"] == "1" for r in rows)
    return {(r["case"], int(r["threads"])): r for r in rows}


def main():
    before, after = (read(p) for p in sys.argv[1:3])
    assert before.keys() == after.keys()
    assert all(before[k]["output_hash"] == after[k]["output_hash"] for k in before)
    print("# Boolean2 offset parallelism\n")
    print("Two round offsets for fillet-style opening/closing, 32 segments per "
          "circle. Release with MANIFOLD_PAR=ON, 1/2/4/8 TBB threads, affinity "
          "to eight physical P cores (0,2,4,6,8,10,12,14). Five timed calls "
          "per large case after warmup; table values are medians.\n")
    print("Before: existing parallel cleanup, sequential contour generation. "
          "After: additionally parallel contour generation above 1,024 input "
          "vertices when multiple contours exist. Small inputs and builds "
          "with MANIFOLD_PAR=OFF retain the sequential generation path.\n")
    print("**The added contour-parallel production change was not retained:** "
          "it produced no meaningful end-to-end speedup in these measurements. "
          "The tested source and determinism test are preserved in "
          "`contour_parallel.patch`. The existing MANIFOLD_PAR backend is the "
          "recommended way to obtain the measured threading benefit.\n")
    print("| Case | Input contours | Before 1 thread, ms | Before 8 threads, ms | After 1 thread, ms | After 8 threads, ms | After 1→8 speedup | Added change speedup at 8 threads |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    cases = [c for c, t in after if t == 1]
    for c in cases:
        b1, b8, a1, a8 = (float(d[c,t]["median_us"])/1000
                           for d,t in ((before,1),(before,8),(after,1),(after,8)))
        print(f"| {c} | {after[c,1]['input_contours']} | {b1:.3f} | {b8:.3f} | "
              f"{a1:.3f} | {a8:.3f} | {a1/a8:.3f}× | {b8/a8:.3f}× |")
    print("\nSpeedup above 1 means faster. The before/after columns distinguish "
          "existing cleanup parallelism from the added contour loop. Separate "
          "runs remain subject to frequency and scheduling variation; small "
          "changes should not be interpreted as decisive speedups.\n")
    print("## Thread scaling after the change\n")
    print("| Case | 1 thread, ms | 2 threads, ms | 4 threads, ms | 8 threads, ms |")
    print("|---|---:|---:|---:|---:|")
    for c in cases:
        print(f"| {c} | " + " | ".join(
            f"{float(after[c,t]['median_us'])/1000:.3f}" for t in (1,2,4,8)) + " |")
    print("\n## Phase profile, 8 threads\n")
    print("Profiling is a separate warmup invocation with phase clocks enabled, "
          "not the median timed call. Phase percentages use cleanup time, "
          "excluding contour generation and final polygon tracing.\n")
    print("| Case | Cleanup, ms | Sweep, ms | Sweep / cleanup |")
    print("|---|---:|---:|---:|")
    for c in cases:
        r = after[c,8]
        total, sweep = float(r["cleanup_us"]), float(r["sweep_us"])
        print(f"| {c} | {total/1000:.3f} | {sweep/1000:.3f} | {100*sweep/total:.1f}% |")
    print("\nEvery warmup and timed result was compared coordinate-for-coordinate "
          "and contour-for-contour with the same case's one-thread result. "
          "Before/after output hashes also match for every case/thread pair. "
          "This confirms deterministic scheduling for the tested inputs; "
          "existing geometry differences from Clipper2 are not repaired.\n")
    if len(sys.argv) > 3:
        verified = read(sys.argv[3])
        print(f"The full verification-only run checked {len(verified)} outputs "
              f"across {len(verified)//4} cases and 1/2/4/8 threads. "
              "Its single profiled invocations are not performance samples.")


if __name__ == "__main__":
    main()
