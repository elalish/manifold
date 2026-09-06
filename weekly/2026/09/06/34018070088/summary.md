### Weekly Benchmarks

Commit: `2e2ed36968fb0a1c24e95d14584bf614eb5b8f16`
Runner: `GitHub Actions 1000063661`
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
| 667 | Intersect12 Q->P | 1172.20 | 0.997 | 472.20 | 696.60 | 0.00 | 3.40 | 5 |
| 695 | Intersect12 P->Q | 484.40 | 0.991 | 337.60 | 142.20 | 4.60 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 422.20 | 0.998 | 167.80 | 253.40 | 0.00 | 1.00 | 5 |
| 84 | Intersect12 P->Q | 370.80 | 0.991 | 218.20 | 149.40 | 2.20 | 1.00 | 5 |
| 260 | Intersect12 Q->P | 168.40 | 0.973 | 74.40 | 89.40 | 1.60 | 3.00 | 5 |
| 406 | Intersect12 P->Q | 135.40 | 0.976 | 85.00 | 47.20 | 3.20 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 105.20 | 0.960 | 58.00 | 43.00 | 4.20 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 42.60 | 1.000 | 24.00 | 18.60 | 0.00 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.29 | 1.19 | 0.93 | 1.88 | 4.82 | 4.75 | 4.94 | 5 |
| 2048 | 2.69 | 2.59 | 1.90 | 3.90 | 6.26 | 6.22 | 6.36 | 5 |
| 8192 | 6.67 | 7.14 | 5.16 | 7.77 | 11.37 | 11.31 | 11.48 | 5 |
| 32768 | 19.19 | 16.01 | 13.41 | 34.05 | 35.56 | 29.73 | 37.61 | 5 |
| 131072 | 61.13 | 61.65 | 48.25 | 77.65 | 128.15 | 115.77 | 144.05 | 5 |
| 524288 | 313.24 | 297.20 | 266.19 | 378.51 | 525.77 | 515.17 | 536.52 | 5 |
| 2097152 | 1010.36 | 983.99 | 718.73 | 1420.38 | 1967.62 | 1526.72 | 2089.72 | 5 |
| 8388608 | 14047.96 | 12337.10 | 10475.80 | 20699.50 | 3955.13 | 3278.09 | 4179.75 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 1955.40 | 1926.00 | 1845.00 | 2059.00 | 5 |
| Boolean.BatchBoolean | 2.20 | 2.00 | 2.00 | 3.00 | 5 |
| CrossSection.BatchBoolean | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| Polygon.Sponge4 | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| Polygon.Zebra1 | 2.00 | 2.00 | 2.00 | 2.00 | 5 |
| Polygon.Zebra3 | 976.20 | 958.00 | 861.00 | 1119.00 | 5 |

