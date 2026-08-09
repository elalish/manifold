### Weekly Benchmarks

Commit: `ff42ddc885e2287faa176873e38e795572e95992`
Runner: `GitHub Actions 1000060601`
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
| 667 | Intersect12 Q->P | 1225.00 | 0.997 | 515.40 | 706.20 | 0.00 | 3.40 | 5 |
| 695 | Intersect12 P->Q | 545.60 | 0.989 | 373.20 | 166.20 | 6.20 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 454.20 | 0.998 | 178.60 | 274.60 | 0.00 | 1.00 | 5 |
| 84 | Intersect12 P->Q | 393.00 | 0.991 | 229.40 | 160.00 | 2.60 | 1.00 | 5 |
| 260 | Intersect12 Q->P | 183.80 | 0.971 | 79.80 | 98.60 | 1.80 | 3.60 | 5 |
| 406 | Intersect12 P->Q | 139.60 | 0.977 | 87.60 | 48.80 | 3.20 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 115.60 | 0.957 | 65.00 | 45.60 | 5.00 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 45.40 | 0.996 | 25.40 | 19.80 | 0.20 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.84 | 1.65 | 1.05 | 2.99 | 4.81 | 4.78 | 4.86 | 5 |
| 2048 | 3.99 | 3.75 | 2.26 | 7.11 | 6.25 | 6.02 | 6.58 | 5 |
| 8192 | 9.13 | 9.69 | 6.56 | 12.74 | 12.25 | 11.41 | 13.20 | 5 |
| 32768 | 33.06 | 35.06 | 15.85 | 54.86 | 36.29 | 35.64 | 37.02 | 5 |
| 131072 | 104.12 | 97.82 | 65.70 | 157.14 | 124.95 | 117.81 | 131.28 | 5 |
| 524288 | 467.53 | 456.45 | 275.89 | 616.61 | 533.98 | 526.17 | 547.52 | 5 |
| 2097152 | 1498.90 | 1700.34 | 935.39 | 1870.04 | 1966.46 | 1492.47 | 2088.41 | 5 |
| 8388608 | 19258.78 | 17599.80 | 13314.40 | 26321.30 | 3924.27 | 3052.00 | 4181.62 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 2365.40 | 2273.00 | 2037.00 | 2788.00 | 5 |
| ExecutionContextFromMeshGL.CancelConcurrent | 89.60 | 94.00 | 51.00 | 147.00 | 5 |
| Boolean.BatchBoolean | 4.40 | 2.00 | 1.00 | 14.00 | 5 |
| CrossSection.BatchBoolean | 2.00 | 0.00 | 0.00 | 10.00 | 5 |
| Polygon.Sponge4 | 0.80 | 1.00 | 0.00 | 1.00 | 5 |
| Polygon.Zebra1 | 2.60 | 3.00 | 2.00 | 3.00 | 5 |
| Polygon.Zebra3 | 1180.20 | 1094.00 | 993.00 | 1475.00 | 5 |

