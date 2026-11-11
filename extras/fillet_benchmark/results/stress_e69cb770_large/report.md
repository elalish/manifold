# Fillet stress: `e69cb770`

Completed 0 jobs; 4 incomplete jobs. Wall time: 240.3 seconds.

Calls below exclude the 13 untransformed reference calls per job and any work performed by timed-out processes.

| Segments | Cases completed | Calls | Empty results | Empty flips | Invalid loops | Sharp turns | Wrong area direction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Empty flips compare transformed inputs with the untransformed reference at the same normalized radius. They are observations, not automatically confirmed defects: Fillet deliberately rejects unchanged areas, and near-degenerate input cleanup can also change under transforms. Sharp turns use the existing positive-radius unit-test threshold (1.1 × 360° / segments). Negative-radius sharp turns are not checked.

| Case | Segments | Empty flips | Largest area drift / input area |
|---|---:|---:|---:|

Area drift is diagnostic: arc tessellation changes with rotation, especially at 8 segments. It is not a correctness verdict. Finite-loop checks do not prove absence of intersections or geometric equivalence.

Incomplete jobs (not counted as successful coverage):

- zebra3.txt:Zebra3, segments=20, seeds=1..1: timeout
- zebra.txt:Zebra, segments=20, seeds=1..1: timeout
- zebra.txt:Zebra, segments=32, seeds=1..1: timeout
- zebra3.txt:Zebra3, segments=32, seeds=1..1: timeout
