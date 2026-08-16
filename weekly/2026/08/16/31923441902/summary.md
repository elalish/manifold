### Weekly Benchmarks

Commit: `d38122c5ab515eae9f74273f53ed299e49b4f744`
Runner: `GitHub Actions 1000061618`
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
| 667 | Intersect12 Q->P | 1037.20 | 0.997 | 417.00 | 617.20 | 0.00 | 3.00 | 5 |
| 695 | Intersect12 P->Q | 439.80 | 0.991 | 307.60 | 128.20 | 4.00 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 378.20 | 0.997 | 150.20 | 227.00 | 0.00 | 1.00 | 5 |
| 84 | Intersect12 P->Q | 337.80 | 0.990 | 196.60 | 137.80 | 2.40 | 1.00 | 5 |
| 260 | Intersect12 Q->P | 151.00 | 0.974 | 66.60 | 80.40 | 1.00 | 3.00 | 5 |
| 406 | Intersect12 P->Q | 118.60 | 0.975 | 74.20 | 41.40 | 3.00 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 96.40 | 0.958 | 53.80 | 38.60 | 4.00 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 39.80 | 1.000 | 22.40 | 17.40 | 0.00 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.77 | 0.77 | 0.76 | 0.78 | 4.77 | 4.73 | 4.81 | 5 |
| 2048 | 1.85 | 1.85 | 1.83 | 1.87 | 6.23 | 6.22 | 6.25 | 5 |
| 8192 | 4.90 | 4.89 | 4.88 | 4.91 | 11.32 | 11.30 | 11.36 | 5 |
| 32768 | 11.82 | 11.93 | 11.55 | 12.13 | 34.36 | 31.75 | 36.92 | 5 |
| 131072 | 41.42 | 40.96 | 40.35 | 42.66 | 118.53 | 101.20 | 131.03 | 5 |
| 524288 | 165.96 | 154.38 | 152.94 | 214.37 | 537.85 | 524.33 | 554.95 | 5 |
| 2097152 | 667.39 | 608.52 | 608.26 | 903.16 | 1974.44 | 1471.12 | 2112.33 | 5 |
| 8388608 | 9842.05 | 9537.08 | 9025.10 | 11199.80 | 3994.66 | 3363.78 | 4169.94 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 1278.40 | 1277.00 | 1276.00 | 1281.00 | 5 |
| ExecutionContextFromMeshGL.CancelConcurrent | 45.40 | 44.00 | 43.00 | 51.00 | 5 |
| Boolean.BatchBoolean | 1.20 | 1.00 | 1.00 | 2.00 | 5 |
| CrossSection.BatchBoolean | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| Polygon.Sponge4 | 0.00 | 0.00 | 0.00 | 0.00 | 5 |
| Polygon.Zebra1 | 2.00 | 2.00 | 2.00 | 2.00 | 5 |
| Polygon.Zebra3 | 767.60 | 757.00 | 745.00 | 825.00 | 5 |

