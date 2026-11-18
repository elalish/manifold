# Updated fillet2d stress rerun

Tested `674cb693` plus the local `TripleCut` changes in [worktree.patch](worktree.patch).
Exact source: [fillet2d_snapshot.cpp](fillet2d_snapshot.cpp).
SHA256: `537d3df2e81907696c21a5398e2b363105cf150625d319c46d36212335df2e12`. The source hash was unchanged through the run.
The production worktree was left as found; no source changes or commits were made.

**Deep stress failures: 65 → 2 (96.9% fewer) on the same 9,584,640 calls.**
All 65 prior failure polygons now return accepted geometry. The remaining two
are different test points that passed in the old run. Fixtures and harness
are unchanged from the previous benchmark.

| Shape | Previous failures | Updated failures |
|---|---:|---:|
| Sponge3 | 48 | 2 |
| Sponge | 15 | 0 |
| Sponge4 | 1 | 0 |
| Slit | 1 | 0 |
| Eight other selected shapes | 0 | 0 |

Each selected shape received 1,024 seeds × 60 transforms × 13 signed radii
at 20 arc segments: 798,720 calls per shape. All deep jobs completed.
Updated failure rate: 0.209 per million calls, compared with 6.782 previously;
these are empirical rates for this workload, not probability guarantees.

## Remaining failures

| Case | Seed | Transform (zero-based) | Radius factor | Rejection | Candidate sharp turns | Maximum turn |
|---|---:|---:|---:|---|---:|---:|
| Sponge3 | 97 | 27 | 0.5 | sharp_angle | 2 | 161.57° |
| Sponge3 | 227 | 47 | 0.5 | sharp_angle | 8 | 153.16° |

Both failures reproduce from the saved input polygons. Construction produces
nonempty candidate geometry, then the validator rejects a sharp angle and clears
the entire result. Neither is an empty trace. This identifies why the output is
empty; it does not establish which construction/tracing decision caused it.

See [classified failures](deep/classified_failures.csv),
[exact reproducers](deep/replays), [deep report](deep/report.md), and
[all 65 old-failure replay results](old_failure_replays/comparison.csv).
For example:

```sh
build/fillet-benchmark/fillet_rejection_probe extras/fillet_benchmark/results/stress_674cb693_updated/deep/replays/Sponge3-s97-t27.txt REPRO 20
```

## Broad sweep and verification

The broad pass completed the same 5,404,672 calls across 116 corpus entries:
256 seeds at 20 segments, 64 seeds at 8/32/64 segments, and 8 transforms per seed.
Combined completed calls across the broad and deep passes: **14,989,312**.

Across both passes, returned geometry produced zero invalid-loop findings,
zero positive-radius sharp-turn violations, and zero wrong-area-direction
findings. Rejected candidates can still have sharp turns, as the two failures
show. These checks do not establish equivalence or absence of intersections;
negative-radius output angles were not independently checked.

| Arc segments | Previous broad empty/nonempty flips | Updated flips |
|---|---:|---:|
| 8 | 912 | 909 |
| 20 | 3,232 | 3,238 |
| 32 | 769 | 769 |
| 64 | 307 | 307 |

These flips compare transformed and untransformed inputs. They mostly involve
degenerate fixtures and strict no-change rejection, and are not automatically
confirmed geometry defects. They remain separate from the strict deep failures.
See [broad details](broad/report.md).

Zebra and Zebra3 again exceeded the 120-second per-job cap at all four
resolutions; their partial work is excluded from completed stress coverage.
All **127 existing fillet unit tests passed**, including Zebra and Zebra3,
after rebuilding against the updated source; see [unit results](unit_summary.txt).

Release sequential library, debug/assertions off, GCC 11.4, eight subprocess
workers per sweep. Broad elapsed time: 300.7 seconds; deep: 464.1 seconds.
The passes and unit tests ran concurrently, so these times are not isolated
performance comparisons. Source snapshots, patch, metadata, complete per-job
counts and replay commands are retained. Raw per-job CSVs and first-observation
polygons are compressed in each run's `raw_details.tar.gz`; extract in that
directory before regenerating reports that read those raw files.
