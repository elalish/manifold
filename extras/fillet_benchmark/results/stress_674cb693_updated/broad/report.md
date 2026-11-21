# Fillet stress: `674cb693`

Completed 3,248 jobs; 8 incomplete jobs. Wall time: 300.7 seconds.

Calls below exclude the 13 untransformed reference calls per job and any work performed by timed-out processes.

| Segments | Cases completed | Calls | Empty results | Empty flips | Invalid loops | Sharp turns | Wrong area direction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 116 | 772,096 | 132,373 | 909 | 0 | 0 | 0 |
| 20 | 116 | 3,088,384 | 514,906 | 3,238 | 0 | 0 | 0 |
| 32 | 116 | 772,096 | 128,767 | 769 | 0 | 0 | 0 |
| 64 | 116 | 772,096 | 124,621 | 307 | 0 | 0 | 0 |

Empty flips compare transformed inputs with the untransformed reference at the same normalized radius. They are observations, not automatically confirmed defects: Fillet deliberately rejects unchanged areas, and near-degenerate input cleanup can also change under transforms. Sharp turns use the existing positive-radius unit-test threshold (1.1 × 360° / segments). Negative-radius sharp turns are not checked.

| Case | Segments | Empty flips | Largest area drift / input area |
|---|---:|---:|---:|
| polygon_corpus.txt:ColinearY | 8 | 81 | 0.0949998 |
| polygon_corpus.txt:Duplicate | 8 | 462 | 4.21823e-10 |
| polygon_corpus.txt:Duplicate | 20 | 1,956 | 4.23519e-10 |
| polygon_corpus.txt:Duplicate | 32 | 462 | 4.23594e-10 |
| polygon_corpus.txt:Sliver5 | 8 | 160 | 0.00156821 |
| polygon_corpus.txt:Sliver5 | 20 | 648 | 0.000143123 |
| polygon_corpus.txt:Sliver5 | 32 | 160 | 3.85904e-05 |
| polygon_corpus.txt:Sliver5 | 64 | 160 | 5.2623e-06 |
| polygon_corpus.txt:Sliver6 | 8 | 83 | 0 |
| polygon_corpus.txt:Sliver6 | 20 | 88 | 0 |
| polygon_corpus.txt:Sliver6 | 32 | 24 | 0 |
| polygon_corpus.txt:Sliver6 | 64 | 24 | 0 |
| polygon_corpus.txt:TouchingHole | 8 | 123 | 0.0173718 |
| polygon_corpus.txt:TouchingHole | 20 | 546 | 0.00158543 |
| polygon_corpus.txt:TouchingHole | 32 | 123 | 0.000427482 |
| polygon_corpus.txt:TouchingHole | 64 | 123 | 5.82928e-05 |

Area drift is diagnostic: arc tessellation changes with rotation, especially at 8 segments. It is not a correctness verdict. Finite-loop checks do not prove absence of intersections or geometric equivalence.

Incomplete jobs (not counted as successful coverage):

- zebra.txt:Zebra, segments=20, seeds=1..8: timeout
- zebra3.txt:Zebra3, segments=20, seeds=1..8: timeout
- zebra.txt:Zebra, segments=8, seeds=1..8: timeout
- zebra3.txt:Zebra3, segments=8, seeds=1..8: timeout
- zebra.txt:Zebra, segments=32, seeds=1..8: timeout
- zebra3.txt:Zebra3, segments=32, seeds=1..8: timeout
- zebra.txt:Zebra, segments=64, seeds=1..8: timeout
- zebra3.txt:Zebra3, segments=64, seeds=1..8: timeout
