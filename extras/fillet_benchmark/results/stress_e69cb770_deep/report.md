# Fillet stress: `e69cb770`

Completed 768 jobs; 0 incomplete jobs. Wall time: 336.3 seconds.

Calls below exclude the 13 untransformed reference calls per job and any work performed by timed-out processes.

| Segments | Cases completed | Calls | Empty results | Empty flips | Invalid loops | Sharp turns | Wrong area direction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 12 | 9,584,640 | 65 | 65 | 0 | 0 | 0 |

Empty flips compare transformed inputs with the untransformed reference at the same normalized radius. They are observations, not automatically confirmed defects: Fillet deliberately rejects unchanged areas, and near-degenerate input cleanup can also change under transforms. Sharp turns use the existing positive-radius unit-test threshold (1.1 × 360° / segments). Negative-radius sharp turns are not checked.

| Case | Segments | Empty flips | Largest area drift / input area |
|---|---:|---:|---:|
| polygon_corpus.txt:Slit | 20 | 1 | 8.54342e-09 |
| polygon_corpus.txt:Sponge | 20 | 15 | 3.06758e-05 |
| sponge.txt:Sponge3 | 20 | 48 | 0.749189 |
| sponge.txt:Sponge4 | 20 | 1 | 0.000321201 |

Area drift is diagnostic: arc tessellation changes with rotation, especially at 8 segments. It is not a correctness verdict. Finite-loop checks do not prove absence of intersections or geometric equivalence.

Original strict checks require nonempty results at all 13 radii and nondecreasing area across the eight negative radii, allowing 1e-9 relative roundoff. These checks apply only to the selected known-roundable fixtures.

| Case | Calls | Strict failures |
|---|---:|---:|
| Comb | 798,720 | 0 |
| Concave2 | 798,720 | 0 |
| KissingZigzag | 798,720 | 0 |
| Slit | 798,720 | 1 |
| Sponge | 798,720 | 15 |
| Sponge3 | 798,720 | 48 |
| Sponge4 | 798,720 | 1 |
| SpongeThin | 798,720 | 0 |
| SquareHoles | 798,720 | 0 |
| TShape | 798,720 | 0 |
| TouchingHole2 | 798,720 | 0 |
| Ugly | 798,720 | 0 |

All 65 strict failure events are in `failures.csv`.
