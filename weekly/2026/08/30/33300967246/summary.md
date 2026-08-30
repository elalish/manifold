### Weekly Benchmarks

Commit: `0d9abcd72a95a7b3c0ee937e16b644ef4028d40f`
Runner: `GitHub Actions 1000063488`
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
| 667 | Intersect12 Q->P | 1244.60 | 0.997 | 477.80 | 763.40 | 0.00 | 3.40 | 5 |
| 695 | Intersect12 P->Q | 522.00 | 0.990 | 365.80 | 151.00 | 5.20 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 485.00 | 0.998 | 191.40 | 292.60 | 0.00 | 1.00 | 5 |
| 84 | Intersect12 P->Q | 403.80 | 0.990 | 229.20 | 170.80 | 2.80 | 1.00 | 5 |
| 260 | Intersect12 Q->P | 189.40 | 0.972 | 79.60 | 104.60 | 2.00 | 3.20 | 5 |
| 406 | Intersect12 P->Q | 147.20 | 0.978 | 89.20 | 54.80 | 3.20 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 112.60 | 0.959 | 61.00 | 47.00 | 4.40 | 0.20 | 5 |
| 582 | Intersect12 P->Q | 45.20 | 1.000 | 26.80 | 18.40 | 0.00 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.40 | 1.27 | 0.85 | 2.14 | 4.59 | 4.45 | 4.77 | 5 |
| 2048 | 3.15 | 2.61 | 2.02 | 5.72 | 6.01 | 5.94 | 6.23 | 5 |
| 8192 | 7.54 | 6.78 | 4.94 | 12.62 | 12.48 | 11.31 | 14.11 | 5 |
| 32768 | 21.71 | 24.99 | 13.79 | 27.91 | 36.08 | 32.17 | 37.95 | 5 |
| 131072 | 86.77 | 79.58 | 48.89 | 145.72 | 121.40 | 115.84 | 126.28 | 5 |
| 524288 | 463.03 | 445.93 | 323.96 | 629.90 | 541.30 | 515.05 | 579.20 | 5 |
| 2097152 | 1235.51 | 1237.67 | 712.54 | 1626.14 | 1997.70 | 1637.70 | 2092.48 | 5 |
| 8388608 | 14007.30 | 12666.10 | 11721.70 | 19803.10 | 3795.89 | 3252.08 | 4067.70 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 1854.60 | 1857.00 | 1732.00 | 1949.00 | 5 |
| Boolean.BatchBoolean | 1.80 | 2.00 | 1.00 | 2.00 | 5 |
| CrossSection.BatchBoolean | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| Polygon.Sponge4 | 0.60 | 1.00 | 0.00 | 1.00 | 5 |
| Polygon.Zebra1 | 2.00 | 2.00 | 2.00 | 2.00 | 5 |
| Polygon.Zebra3 | 1076.00 | 1066.00 | 936.00 | 1295.00 | 5 |

