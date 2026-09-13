### Weekly Benchmarks

Commit: `a9cc3c658b8fbaceabf56886c467ad766cf71036`
Runner: `GitHub Actions 1000064421`
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
| 667 | Intersect12 Q->P | 1158.20 | 0.997 | 463.00 | 692.00 | 0.00 | 3.20 | 5 |
| 695 | Intersect12 P->Q | 499.20 | 0.990 | 345.80 | 148.40 | 5.00 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 421.20 | 0.998 | 167.80 | 252.40 | 0.00 | 1.00 | 5 |
| 84 | Intersect12 P->Q | 368.00 | 0.991 | 212.40 | 152.20 | 2.40 | 1.00 | 5 |
| 260 | Intersect12 Q->P | 173.60 | 0.969 | 75.20 | 93.00 | 2.00 | 3.40 | 5 |
| 406 | Intersect12 P->Q | 133.80 | 0.976 | 83.80 | 46.80 | 3.20 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 111.20 | 0.962 | 61.60 | 45.40 | 4.20 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 46.60 | 1.000 | 25.80 | 20.80 | 0.00 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 3.61 | 1.18 | 0.94 | 12.45 | 4.78 | 4.62 | 4.92 | 5 |
| 2048 | 2.67 | 2.50 | 2.27 | 3.78 | 6.22 | 6.06 | 6.28 | 5 |
| 8192 | 7.87 | 6.73 | 5.87 | 13.71 | 11.95 | 11.33 | 12.97 | 5 |
| 32768 | 19.83 | 18.89 | 14.33 | 30.69 | 34.57 | 31.88 | 36.25 | 5 |
| 131072 | 59.88 | 61.86 | 46.00 | 75.25 | 120.49 | 113.67 | 129.61 | 5 |
| 524288 | 387.27 | 346.28 | 220.72 | 653.01 | 527.49 | 517.27 | 540.42 | 5 |
| 2097152 | 1041.76 | 877.21 | 756.03 | 1446.65 | 1963.44 | 1472.67 | 2089.03 | 5 |
| 8388608 | 15571.84 | 13393.60 | 11329.50 | 21408.70 | 3900.29 | 3030.73 | 4210.17 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 2890.60 | 2709.00 | 2628.00 | 3358.00 | 5 |
| Boolean.BatchBoolean | 3.80 | 3.00 | 2.00 | 9.00 | 5 |
| CrossSection.BatchBoolean | 0.40 | 0.00 | 0.00 | 2.00 | 5 |
| Polygon.Sponge4 | 1.20 | 1.00 | 1.00 | 2.00 | 5 |
| Polygon.Zebra1 | 2.80 | 3.00 | 2.00 | 4.00 | 5 |
| Polygon.Zebra3 | 1267.80 | 1224.00 | 971.00 | 1546.00 | 5 |

