### Weekly Benchmarks

Commit: `f3597e73cd58487b836a8600022bad3e8ed668a0`
Runner: `GitHub Actions 1000065503`
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
| 667 | Intersect12 Q->P | 1261.20 | 0.996 | 490.80 | 765.80 | 0.00 | 4.60 | 5 |
| 695 | Intersect12 P->Q | 517.00 | 0.990 | 357.20 | 154.80 | 5.00 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 417.80 | 0.998 | 167.40 | 249.40 | 0.00 | 1.00 | 5 |
| 84 | Intersect12 P->Q | 401.60 | 0.990 | 234.80 | 162.80 | 2.80 | 1.20 | 5 |
| 260 | Intersect12 Q->P | 171.00 | 0.968 | 75.80 | 89.80 | 1.80 | 3.60 | 5 |
| 406 | Intersect12 P->Q | 139.40 | 0.977 | 88.20 | 48.00 | 3.20 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 110.60 | 0.960 | 62.20 | 44.00 | 4.40 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 45.20 | 0.996 | 25.00 | 20.00 | 0.20 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.52 | 1.70 | 0.99 | 1.78 | 4.79 | 4.66 | 4.94 | 5 |
| 2048 | 3.29 | 2.96 | 1.85 | 4.70 | 6.18 | 5.95 | 6.27 | 5 |
| 8192 | 8.66 | 6.59 | 5.06 | 14.98 | 12.24 | 11.34 | 14.00 | 5 |
| 32768 | 27.05 | 21.30 | 12.29 | 46.91 | 33.80 | 29.52 | 39.08 | 5 |
| 131072 | 106.58 | 107.02 | 50.94 | 180.16 | 124.67 | 117.53 | 132.58 | 5 |
| 524288 | 535.87 | 492.53 | 383.23 | 777.74 | 531.23 | 525.12 | 540.48 | 5 |
| 2097152 | 1534.54 | 1607.00 | 1123.46 | 1915.87 | 1952.11 | 1450.42 | 2081.42 | 5 |
| 8388608 | 16951.18 | 15333.10 | 13560.10 | 25164.80 | 3785.86 | 2965.19 | 4074.11 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 2202.60 | 2151.00 | 1666.00 | 2726.00 | 5 |
| Boolean.BatchBoolean | 2.40 | 2.00 | 2.00 | 3.00 | 5 |
| CrossSection.BatchBoolean | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| Polygon.Sponge4 | 1.00 | 1.00 | 1.00 | 1.00 | 5 |
| Polygon.Zebra1 | 2.20 | 2.00 | 2.00 | 3.00 | 5 |
| Polygon.Zebra3 | 1262.20 | 1290.00 | 1093.00 | 1449.00 | 5 |

