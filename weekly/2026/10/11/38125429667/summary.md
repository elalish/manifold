### Weekly Benchmarks

Commit: `7917236a960c2e1dc9f1cf003e63a27352b0cea6`
Runner: `GitHub Actions 1000068641`
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
| 667 | Intersect12 Q->P | 1785.20 | 0.998 | 747.40 | 1033.40 | 0.00 | 4.40 | 5 |
| 695 | Intersect12 P->Q | 850.20 | 0.991 | 548.60 | 294.20 | 7.00 | 0.40 | 5 |
| 16 | Intersect12 Q->P | 620.20 | 0.998 | 237.20 | 381.60 | 0.00 | 1.40 | 5 |
| 84 | Intersect12 P->Q | 546.00 | 0.992 | 309.40 | 232.20 | 3.20 | 1.20 | 5 |
| 260 | Intersect12 Q->P | 274.20 | 0.968 | 109.40 | 156.20 | 3.00 | 5.60 | 5 |
| 406 | Intersect12 P->Q | 235.40 | 0.976 | 134.60 | 95.20 | 5.60 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 173.60 | 0.960 | 98.40 | 68.40 | 6.60 | 0.20 | 5 |
| 582 | Intersect12 P->Q | 63.80 | 1.000 | 34.00 | 29.80 | 0.00 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.90 | 1.53 | 1.42 | 2.82 | 4.74 | 4.56 | 4.92 | 5 |
| 2048 | 4.06 | 3.44 | 2.69 | 6.31 | 6.21 | 5.92 | 7.14 | 5 |
| 8192 | 14.48 | 12.46 | 8.78 | 26.41 | 12.82 | 11.98 | 13.45 | 5 |
| 32768 | 34.27 | 35.62 | 24.63 | 45.68 | 36.52 | 33.56 | 38.97 | 5 |
| 131072 | 149.10 | 120.31 | 72.04 | 241.19 | 125.34 | 119.70 | 130.33 | 5 |
| 524288 | 668.04 | 625.92 | 545.19 | 825.16 | 533.38 | 522.44 | 554.77 | 5 |
| 2097152 | 2165.78 | 2118.24 | 1492.34 | 3221.56 | 1994.51 | 1614.64 | 2100.03 | 5 |
| 8388608 | 20150.18 | 17851.10 | 15392.80 | 31445.50 | 3829.51 | 2912.09 | 4132.30 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 2393.00 | 2477.00 | 1627.00 | 3029.00 | 5 |
| Boolean.BatchBoolean | 6.20 | 2.00 | 2.00 | 23.00 | 5 |
| CrossSection.BatchBoolean | 0.40 | 0.00 | 0.00 | 2.00 | 5 |
| Polygon.Sponge4 | 1.00 | 1.00 | 0.00 | 2.00 | 5 |
| Polygon.Zebra1 | 2.80 | 2.00 | 2.00 | 5.00 | 5 |
| Polygon.Zebra3 | 1288.00 | 1382.00 | 859.00 | 1604.00 | 5 |

