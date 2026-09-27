### Weekly Benchmarks

Commit: `269c662c2ef1c968d13e7f04eadc6916963aa1be`
Runner: `GitHub Actions 1000066586`
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
| 667 | Intersect12 Q->P | 1402.80 | 0.997 | 557.20 | 842.00 | 0.00 | 3.60 | 5 |
| 695 | Intersect12 P->Q | 622.60 | 0.991 | 433.00 | 184.00 | 5.40 | 0.20 | 5 |
| 16 | Intersect12 Q->P | 516.00 | 0.995 | 203.80 | 309.60 | 0.20 | 2.40 | 5 |
| 84 | Intersect12 P->Q | 452.20 | 0.990 | 264.80 | 182.80 | 3.60 | 1.00 | 5 |
| 260 | Intersect12 Q->P | 205.20 | 0.973 | 88.00 | 111.60 | 2.20 | 3.40 | 5 |
| 406 | Intersect12 P->Q | 160.00 | 0.972 | 95.40 | 60.20 | 4.40 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 135.40 | 0.959 | 68.20 | 61.80 | 5.40 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 45.40 | 1.000 | 25.80 | 19.60 | 0.00 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 5.48 | 4.31 | 0.98 | 15.44 | 4.60 | 4.45 | 4.77 | 5 |
| 2048 | 2.95 | 3.17 | 2.15 | 3.76 | 6.15 | 5.61 | 6.34 | 5 |
| 8192 | 9.82 | 8.35 | 5.83 | 17.14 | 11.77 | 11.12 | 12.89 | 5 |
| 32768 | 22.27 | 20.64 | 13.39 | 32.82 | 35.28 | 31.66 | 38.31 | 5 |
| 131072 | 137.32 | 114.41 | 102.64 | 208.21 | 119.21 | 113.00 | 129.09 | 5 |
| 524288 | 582.83 | 417.35 | 340.20 | 1311.26 | 521.95 | 514.36 | 527.89 | 5 |
| 2097152 | 1891.30 | 1546.65 | 1216.86 | 3762.25 | 1983.22 | 1557.42 | 2103.27 | 5 |
| 8388608 | 15987.60 | 13958.60 | 12667.90 | 24190.70 | 3802.70 | 2881.56 | 4185.59 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 1932.60 | 1907.00 | 1727.00 | 2131.00 | 5 |
| Boolean.BatchBoolean | 2.20 | 2.00 | 1.00 | 5.00 | 5 |
| CrossSection.BatchBoolean | 0.40 | 0.00 | 0.00 | 2.00 | 5 |
| Polygon.Sponge4 | 0.80 | 1.00 | 0.00 | 2.00 | 5 |
| Polygon.Zebra1 | 2.00 | 2.00 | 2.00 | 2.00 | 5 |
| Polygon.Zebra3 | 1142.20 | 1072.00 | 1022.00 | 1381.00 | 5 |

