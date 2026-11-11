# fillet2d output and runtime benchmark

The primary comparison uses **Clipper2 as ground truth** to compare current
`CrossSection::Fillet` with Boolean2's current `CrossSection::Offset` construction.
The offset methods use:

```text
offset(offset(input, -radius, Round), +radius, Round)
```

For positive radii this is opening; for negative radii it is closing. Clipper2
has no direct fillet call. Neither approximation is assumed to reproduce the
other implementation. Round joins are needed on both passes to approximate the rolling ball
at concave features. No candidate output is intersected with, unioned with, or
otherwise repaired using the reference.

## Run

Use a local Clipper2 checkout; the benchmark does not download dependencies or
change the main project's build configuration.

```sh
cmake -S extras/fillet_benchmark -B build/fillet-benchmark \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLIPPER2_SOURCE="$PWD/build-bench/_deps/clipper2-src/CPP/Clipper2Lib"
cmake --build build/fillet-benchmark --target fillet_benchmark -j 4
mkdir -p extras/fillet_benchmark/results
taskset -c 0 build/fillet-benchmark/fillet_benchmark "$PWD" \
  extras/fillet_benchmark/results/clipper_gt.csv --large --clipper-gt
python3 extras/fillet_benchmark/clipper_gt_report.py \
  extras/fillet_benchmark/results/clipper_gt.csv \
  > extras/fillet_benchmark/results/clipper_gt.md
```

Omit `--large` for the smaller suite. An optional argument after the flags
filters case names by substring. Select a permitted CPU for `taskset` on your
machine (or omit it on systems without `taskset`).

Without `--clipper-gt`, fillet2d is the reference. Use `report.py` for that mode.
The earlier fillet-reference run is retained as `results/current.csv` and
`results/current.md`.

## Method

- Release, sequential backend, debug/assert output disabled. All methods run in
  the same process. There is one untimed warmup per method/case, then 5–21
  samples, stopping after 20 ms once at least five samples have completed.
  Method timing order rotates per case. CSV reports median and minimum time.
- Timings include output polygon conversion, fillet decomposition, both offset
  passes and cleanup, and Clipper integer conversion. Input normalization through
  `CrossSection` and all validation are outside the timed region. Results are
  consumed to prevent optimization. These are public API costs, not isolated
  internal kernel timings.
- All methods receive the same normalized input contours. Original coordinate
  scale/translation is preserved for Manifold. Clipper input uses a local origin
  and approximately 1e9 integer units per input span; its arc tolerance is
  `abs(radius) * (1 - cos(pi/32))`. All methods request 32 nominal segments per
  circle. Arc phases and segment rounding can differ; vertex identity is not an
  expected invariant.
- Cases cover `fillet.txt`, `polygon_corpus.txt`, deterministic radial polygons
  with 16/64/256 vertices, and square/L/narrow-neck/hole/tiny-edge shapes at
  scales 1e-6, 1, and 1e6, with translations up to 1e8. Corpus and stress sweeps
  use radii 0.1/0.5/0.9/1.01 and -0.1/-0.5/-0.9 times the stored inscribed
  radius (or a chosen stress radius). Random and large fixtures use ±0.5.
  `--large` adds sponge, zebra, and zebra3, including 65,002-vertex input.
- Compare candidate polygons directly against reference polygons using Clipper2
  integer XOR at 1e12 units per input span. Report XOR area divided by input
  area, contour count, sharp signed turns using fillet2d's angle threshold,
  containment error, and signed-area change after NonZero union. The same checker
  evaluates all methods. It is independent of Boolean2 but shares Clipper2 with
  one candidate, so it is not an unbiased exact geometry oracle.
- Tiny filled area can make relative-area metrics ill-conditioned. The CSV's
  `oracle_error_bound` is a conservative grid-displacement estimate based on
  perimeter/area, **not a certified bound on the clipping algorithm**. Reports
  omit ratios with estimates above 1e-6. Empty fillet-reference outputs are counted
  separately because `Fillet` also returns empty on validation rejection and on
  no-change cases. Missing output is not automatically a robustness failure.
- A basic-valid result has finite vertices, nonzero edges, and at least three
  points per contour. These checks and union-area residuals do not prove freedom
  from self-intersections. Reported agreement at 1e-4 or 1e-6 is an explicit
  area/topology threshold, not exact equality or a Hausdorff-distance guarantee.

## Recorded run

The latest run uses commit `9b6ff7ef`: [old/new comparison](results/9b6ff7ef/comparison.md),
[Clipper2 GT benchmark](results/9b6ff7ef/clipper_gt.md),
[sharp-angle rejection diagnostics](results/9b6ff7ef/rejections.md), and
[environment](results/9b6ff7ef/environment.txt). All 969 cases were rerun.
Compare saved benchmark CSVs with:

```sh
python3 extras/fillet_benchmark/compare_runs.py \
  extras/fillet_benchmark/results/clipper_gt.csv \
  extras/fillet_benchmark/results/9b6ff7ef/clipper_gt.csv \
  > extras/fillet_benchmark/results/9b6ff7ef/comparison.md
```

See the [Clipper2 ground-truth report](results/clipper_gt.md), its raw
[CSV](results/clipper_gt.csv), and [environment metadata](results/environment.txt).
Runtime results are split by input size and include a direct comparison on the
same cases where both methods match Clipper2. The earlier
[fillet-reference report](results/current.md) is also retained.
The source implementation was not modified. The benchmark measures its current
behavior, including rejection behavior, and does not certify its mathematical
correctness.

## Measure sharp-angle rejection inside fillet2d

```sh
cmake --build build/fillet-benchmark --target fillet_rejections -j 4
taskset -c 0 build/fillet-benchmark/fillet_rejections "$PWD" \
  extras/fillet_benchmark/results/rejections.csv
python3 extras/fillet_benchmark/rejection_report.py \
  extras/fillet_benchmark/results/rejections.csv \
  > extras/fillet_benchmark/results/rejections.md
```

This target instruments a generated build-local source copy. Its validator
preserves the original branch order and strict comparisons, records geometry
before it is cleared, and classifies empty returns as empty trace, sharp angle,
invalid geometry, unchanged/wrong-direction area, input rejection, or exception.
It also evaluates every nonempty candidate's angle predicate even if another
check would have rejected it first. The production source and normal benchmark
library remain unchanged. Instrumentation timings are not performance results.
See the [rejection measurement](results/rejections.md).

## Offset thread scaling

The [parallelism experiment](results/offset_parallel/report.md) measures 1, 2,
4, and 8 TBB threads on eight physical P cores. It separates the existing
parallel cleanup from an experimental parallel contour-emission loop. The
extra loop produced no meaningful total-runtime improvement and was removed
from production; the tested patch is saved in the results directory.

To benchmark the existing parallel backend using the local TBB sources:

```sh
cmake -S extras/fillet_benchmark -B build/offset-parallel \
  -DFILLET_BENCHMARK_PAR=ON -DCMAKE_BUILD_TYPE=Release \
  -DMANIFOLD_USE_BUILTIN_TBB=ON \
  -DFETCHCONTENT_SOURCE_DIR_TBB="$PWD/build/_deps/tbb-src"
cmake --build build/offset-parallel --target offset_parallel -j 4
taskset -c 0,2,4,6,8,10,12,14 build/offset-parallel/offset_parallel "$PWD" \
  /tmp/offset-scaling.csv --large-only
taskset -c 0,2,4,6,8,10,12,14 build/offset-parallel/offset_parallel "$PWD" \
  /tmp/offset-verify.csv --verify-only
```

Choose affinity appropriate for your CPU. Normal Manifold builds enable this
backend with `-DMANIFOLD_PAR=ON`; `FILLET_BENCHMARK_PAR` configures only this
standalone benchmark project. Verification compares exact coordinates and
contour order across thread counts, not just areas. It covers the original
969 cases plus four disjoint-contour stress cases. Phase profiling is outside
the timed performance samples.
## Expanded stress on e69cb770

See the [combined report](results/stress_e69cb770_summary.md). The broad pass
covers 116 completed corpus entries, 256 seeds at 20 segments and 64 seeds at
8/32/64 segments. The deeper pass uses the original checks on 12 shapes,
1,024 seeds, and 60 transforms per seed. It reproduces all 65 newly found
empty-result failures and records their validator rejection reasons.

```sh
cmake -S extras/fillet_benchmark -B build/fillet-benchmark -DCMAKE_BUILD_TYPE=Release
cmake --build build/fillet-benchmark --target fillet_stress fillet_rejection_probe -j 4
python3 extras/fillet_benchmark/stress_run.py build/fillet-benchmark/fillet_stress . /tmp/fillet-broad --seconds 900 --seeds 256 --transforms 8
python3 extras/fillet_benchmark/stress_run.py build/fillet-benchmark/fillet_stress . /tmp/fillet-deep --seconds 900 --seeds 1024 --transforms 60 --segments 20 --strict --cases Comb Concave2 KissingZigzag Slit Sponge Sponge3 Sponge4 SpongeThin SquareHoles TShape TouchingHole2 Ugly
python3 extras/fillet_benchmark/stress_report.py /tmp/fillet-deep > /tmp/fillet-deep/report.md
python3 extras/fillet_benchmark/stress_replay.py /tmp/fillet-deep . build/fillet-benchmark/fillet_stress build/fillet-benchmark/fillet_rejection_probe
```

Run from the repository root. Each subprocess has a 120-second timeout; the
runner also enforces the overall deadline. Strict checks are appropriate only
for fixtures known to produce nonempty results at the selected radii. Both
passes use rotations over a full turn, scales exp(U[-2,2]), translations in
[-50,50] on each axis, negative radius factors -0.5/-1/-2/-5/-10/-20/-50/-100,
and positive factors 0.1/0.3/0.5/0.7/0.9 of the corpus maximum circle radius.
Saved doubles and polygons provide portable reproducers; RNG replay uses
`mt19937_64` and the standard library's uniform distribution, whose mapping
is not promised identical across different C++ standard libraries.

`jobs.jsonl` contains exact commands and authoritative counts. Event details
for generic observations are capped at 100 per job; strict failure CSVs are
complete. Recorded per-job CSVs and first-observation polygons are compressed
in `raw_details.tar.gz`; extract in the containing run directory before
regenerating reports or classifications. All strict failure polygons and
their rejection probes remain directly readable under the deep run's
`replays/` directory. Probe a single saved polygon with:

```sh
build/fillet-benchmark/fillet_rejection_probe extras/fillet_benchmark/results/stress_e69cb770_deep/replays/Sponge3-s36-t50.txt REPRO 20
```
