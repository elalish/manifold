### Weekly Benchmarks

Commit: `b490dacf1e67715808851a7f690fddaf790bef2c`
Runner: `GitHub Actions 1000067898`
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
| 667 | Intersect12 Q->P | 1333.80 | 0.998 | 538.80 | 792.00 | 0.00 | 3.00 | 5 |
| 695 | Intersect12 P->Q | 534.00 | 0.986 | 361.80 | 164.60 | 7.40 | 0.20 | 5 |
| 16 | Intersect12 Q->P | 496.40 | 0.998 | 183.20 | 312.00 | 0.00 | 1.20 | 5 |
| 84 | Intersect12 P->Q | 412.40 | 0.989 | 231.20 | 176.60 | 3.20 | 1.40 | 5 |
| 260 | Intersect12 Q->P | 192.60 | 0.968 | 85.20 | 101.20 | 2.00 | 4.20 | 5 |
| 406 | Intersect12 P->Q | 162.20 | 0.976 | 105.20 | 53.20 | 3.80 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 120.20 | 0.956 | 63.80 | 51.00 | 5.40 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 47.80 | 0.997 | 27.20 | 20.40 | 0.20 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 3.60 | 1.58 | 0.92 | 11.85 | 4.60 | 4.52 | 4.72 | 5 |
| 2048 | 3.93 | 3.52 | 2.69 | 5.56 | 6.25 | 6.14 | 6.56 | 5 |
| 8192 | 8.57 | 8.10 | 7.17 | 10.71 | 12.35 | 11.80 | 13.64 | 5 |
| 32768 | 28.61 | 21.46 | 15.09 | 50.23 | 35.02 | 30.95 | 38.05 | 5 |
| 131072 | 112.68 | 115.08 | 65.45 | 162.79 | 128.18 | 121.33 | 133.70 | 5 |
| 524288 | 477.93 | 454.71 | 303.33 | 715.79 | 520.51 | 481.59 | 534.61 | 5 |
| 2097152 | 1397.54 | 1473.37 | 897.40 | 1957.65 | 1956.82 | 1458.91 | 2092.97 | 5 |
| 8388608 | 15494.18 | 14696.00 | 12346.40 | 20044.80 | 3747.42 | 2912.83 | 4091.70 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 2205.60 | 2130.00 | 1941.00 | 2441.00 | 5 |
| Boolean.BatchBoolean | 2.20 | 2.00 | 2.00 | 3.00 | 5 |
| CrossSection.BatchBoolean | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| Polygon.Sponge4 | 1.00 | 1.00 | 1.00 | 1.00 | 5 |
| Polygon.Zebra1 | 2.80 | 2.00 | 2.00 | 5.00 | 5 |
| Polygon.Zebra3 | 1324.40 | 1319.00 | 1239.00 | 1390.00 | 5 |

