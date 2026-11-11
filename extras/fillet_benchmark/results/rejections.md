# fillet2d final-validation rejection measurement

Same 969 cases as the Clipper2-ground-truth benchmark, 32 segments per circle, Release, sequential backend. Instrumentation uses a generated build-local copy of fillet2d.cpp and preserves the original validation order and strict comparisons. Production source is unchanged. Diagnostic timings are not performance measurements.

**0 calls were rejected by the angle check.** Of 852 constructed nonempty candidates, 0 failed the angle predicate, including candidates that might have failed an earlier check.

| Final outcome | Cases | Positive radius | Negative radius |
|---|---:|---:|---:|
| accepted | 749 | 432 | 317 |
| trace_empty | 117 | 117 | 0 |
| area_unchanged | 103 | 0 | 103 |
| sharp_angle | 0 | 0 | 0 |
| invalid_geometry | 0 | 0 | 0 |
| area_wrong_direction | 0 | 0 | 0 |
| input_rejected | 0 | 0 | 0 |
| exception | 0 | 0 | 0 |

The angle threshold was **12.375°**. The largest measured turn in the rounded direction before final rejection was **11.25003501°**, in `stress/square/s=1.000000/t=100000000.000000/0.100000`. Corners in the opposite direction are outside this angle criterion.

`trace_empty` means tracing already returned no geometry. This stage does not distinguish genuine geometric extinction from tracing failure, so these calls cannot all be labeled circle-fit failures. `area_unchanged` means nonempty geometry passed geometry and angle checks but failed the requirement that positive radius strictly decrease area or negative radius strictly increase area.

| Outcome | Clipper2 empty | Clipper2 nonempty |
|---|---:|---:|
| accepted | 0 | 749 |
| trace_empty | 109 | 8 |
| area_unchanged | 6 | 97 |

Raw per-case diagnostics are in `rejections.csv`. Any nonempty angle-failing geometry is written before clearing to `rejections.csv.raw.txt` in polygon-corpus format. An empty raw file means no such cases were observed. This result is specific to these cases and this segment count; it is not a proof that angle rejection can never occur.
