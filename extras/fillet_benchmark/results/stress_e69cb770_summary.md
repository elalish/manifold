# Expanded fillet2d stress, e69cb770

Source snapshot: `e69cb770a9af1a47c116a7022d16adb2eeb20a3c`.
The previously uncommitted 19-line change was committed before building.
The historical implementation context was read from Paseo agents
`f7e7d56a-4d88-448e-a2e0-2f24bbc98aad` and
`79cb022a-763e-43de-8bc5-92194cbc7c6b`.

**Result: 14,989,312 completed transformed fillet calls. The deeper pass found
65 reproducible empty-result failures, all rejected by the sharp-angle
validator.** There were no failures in its seeds 1–32, matching the earlier
299,520-call result. More seeds expose rare failures that the earlier run missed.

| Pass | Coverage | Calls | Wall time |
|---|---|---:|---:|
| Broad | 116 corpus entries; 256 seeds at 20 segments, 64 seeds at 8/32/64; 8 transforms/seed | 5,404,672 | 261.0 s |
| Deep | 12 known-roundable shapes; 1,024 seeds; 60 transforms/seed; 20 segments | 9,584,640 | 336.3 s |

Both passes use 13 signed radii per transform. Counts exclude untransformed
reference calls, replay/probe calls, and partial work in timed-out jobs.
These elapsed times come from eight concurrent sequential-library processes;
they are not isolated performance measurements or comparisons with Boolean2
or Clipper2. Build: Release, GCC 11.4, debug/assertions off, TBB off.

## Strict failures

| Shape | Failed calls / 798,720 | Radius factors |
|---|---:|---|
| Sponge3 | 48 | +0.5: 45; +0.7: 2; -0.5: 1 |
| Sponge | 15 | -1: 15 |
| Sponge4 | 1 | -1: 1 |
| Slit | 1 | +0.7: 1 |
| Comb, Concave2, KissingZigzag, SpongeThin, SquareHoles, TShape, TouchingHole2, Ugly | 0 each | |

Failure rate in the deep pass: 6.78 per million calls. This is an empirical
rate for these fixtures and sampling choices, not a probability guarantee.
All 65 failures reproduced from saved polygon coordinates and radii. Build-local
instrumentation identifies `sharp_angle` for every one, with maximum candidate
turns between 108.43° and 163.71° versus the 19.8° limit at 20 segments.
They reach the validator with nonempty candidate geometry; the validator then
clears the result. This identifies the rejection reason, not the underlying
construction bug or a proof that a circle cannot fit.

First example: `Sponge3`, seed 36, transform 50 (zero-based), radius +0.5R.
Second example: `Sponge`, seed 55, transform 34, radius -R.
See [all classified failures](stress_e69cb770_deep/classified_failures.csv),
[the strict report](stress_e69cb770_deep/report.md), and exact input polygons
in [the replay directory](stress_e69cb770_deep/replays).

## Broader observations and limits

The broad sweep found 3,232 transformed/untransformed empty-result differences
at 20 segments, plus 912 at 8, 769 at 32, and 307 at 64. Most involve degenerate
fixtures such as Duplicate, Sliver5, Sliver6, and TouchingHole; ColinearY and
Sponge also differ at 8 segments. These are recorded separately from strict
failures: legitimate no-change rejection, tiny area roundoff, and changes in
input normalization can affect acceptance. For example, an accepted transformed
Sliver5 result differs from its input area by only 2.84e-14. We did not classify
all these observations as confirmed geometry bugs.

Across both completed passes: zero invalid-loop findings, zero positive-radius
sharp-turn findings in returned geometry, and zero wrong-area-direction findings.
Rejected candidates can still contain sharp turns, as the 65 strict failures
demonstrate. Negative-radius returned sharp turns were not independently checked.
Area drift under transforms is diagnostic because arc tessellation depends on
rotation. These checks do not prove geometric equivalence or absence of
self-intersections.

The broad sweep selected 118 entries; Zebra and Zebra3 timed out at all four
resolutions with the 120-second per-job cap. A reduced one-seed/one-transform
retry at 20/32 segments also timed out. They are excluded from completed stress
coverage. All **127 existing fillet unit tests passed**, including both large
Zebra tests after rerunning with a longer allowance (326.3 seconds combined).

See [broad details](stress_e69cb770/report.md),
[large-case timeout details](stress_e69cb770_large/report.md), and
[environment and verification](stress_e69cb770_environment.txt).
Full per-job data, replay inputs, and reproduction commands are retained.
