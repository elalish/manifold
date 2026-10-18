### Weekly Benchmarks

Commit: `7aa3f48609a89be1ff77860b02f1c7e7e5021882`
Runner: `GitHub Actions 1000069681`
OS: `macOS`
Compiler: `AppleClang 17.0.0.17000013`
CPU: `Apple M1 (Virtual)`
CPU count: `3`
CPU model identifier: `VirtualMac2,1`
CPU physical cores: `3`
CPU performance cores: `3`
Repeats: `5`

#### Ember Phase Timings

| Case | Dominant phase | Full mean (ms) | Intersect12 share | P->Q | Q->P | Winding P | Winding Q | Runs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 667 | Intersect12 Q->P | 2859.00 | 0.998 | 1136.40 | 1717.00 | 0.00 | 5.60 | 5 |
| 695 | Intersect12 P->Q | 1138.80 | 0.989 | 772.60 | 353.60 | 12.20 | 0.40 | 5 |
| 16 | Intersect12 Q->P | 1041.60 | 0.998 | 399.20 | 640.00 | 0.20 | 2.20 | 5 |
| 84 | Intersect12 P->Q | 840.60 | 0.992 | 486.00 | 347.80 | 5.00 | 1.80 | 5 |
| 260 | Intersect12 Q->P | 410.80 | 0.981 | 179.80 | 223.40 | 2.40 | 5.20 | 5 |
| 406 | Intersect12 P->Q | 298.40 | 0.982 | 182.60 | 110.40 | 5.40 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 231.00 | 0.973 | 126.20 | 98.60 | 6.00 | 0.20 | 5 |
| 582 | Intersect12 P->Q | 117.40 | 0.994 | 66.80 | 49.80 | 0.80 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 3.45 | 3.04 | 1.67 | 5.80 | 4.61 | 4.47 | 4.84 | 5 |
| 2048 | 8.79 | 7.19 | 3.11 | 20.16 | 6.13 | 5.80 | 6.41 | 5 |
| 8192 | 18.24 | 17.12 | 13.99 | 26.23 | 12.95 | 12.11 | 13.62 | 5 |
| 32768 | 81.19 | 95.31 | 30.75 | 112.29 | 37.10 | 36.02 | 38.98 | 5 |
| 131072 | 246.82 | 265.32 | 103.88 | 337.93 | 125.84 | 118.23 | 134.72 | 5 |
| 524288 | 858.00 | 817.33 | 658.27 | 1058.56 | 526.55 | 521.89 | 538.80 | 5 |
| 2097152 | 2563.12 | 2488.86 | 2275.05 | 2873.72 | 2016.88 | 1741.81 | 2107.92 | 5 |
| 8388608 | 28522.36 | 27548.50 | 24533.00 | 36605.80 | 3708.95 | 3109.22 | 3998.34 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 3660.40 | 3570.00 | 3494.00 | 4012.00 | 5 |
| Boolean.BatchBoolean | 8.40 | 3.00 | 2.00 | 29.00 | 5 |
| CrossSection.BatchBoolean | 5.20 | 0.00 | 0.00 | 25.00 | 5 |
| Polygon.Sponge4 | 1.20 | 1.00 | 1.00 | 2.00 | 5 |
| Polygon.Zebra1 | 2.80 | 2.00 | 2.00 | 5.00 | 5 |
| Polygon.Zebra3 | 1983.60 | 1986.00 | 1795.00 | 2304.00 | 5 |

