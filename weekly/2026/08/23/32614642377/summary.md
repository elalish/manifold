### Weekly Benchmarks

Commit: `15224126a430dbfaf8ffc4a18d07935cdc5c48ec`
Runner: `GitHub Actions 1000062496`
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
| 667 | Intersect12 Q->P | 1078.40 | 0.997 | 428.00 | 647.20 | 0.00 | 3.20 | 5 |
| 695 | Intersect12 P->Q | 477.60 | 0.990 | 338.40 | 134.60 | 4.60 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 388.00 | 0.997 | 152.60 | 234.40 | 0.00 | 1.00 | 5 |
| 84 | Intersect12 P->Q | 353.00 | 0.991 | 203.20 | 146.80 | 2.00 | 1.00 | 5 |
| 260 | Intersect12 Q->P | 161.20 | 0.971 | 69.80 | 86.80 | 1.40 | 3.20 | 5 |
| 406 | Intersect12 P->Q | 125.60 | 0.976 | 78.80 | 43.80 | 3.00 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 106.80 | 0.957 | 56.80 | 45.40 | 4.60 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 41.00 | 1.000 | 23.00 | 18.00 | 0.00 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.15 | 1.14 | 0.82 | 1.64 | 4.72 | 4.55 | 4.94 | 5 |
| 2048 | 2.23 | 2.10 | 2.02 | 2.90 | 6.11 | 5.62 | 6.25 | 5 |
| 8192 | 5.58 | 5.53 | 5.37 | 5.90 | 11.90 | 11.33 | 12.94 | 5 |
| 32768 | 14.52 | 14.81 | 13.48 | 15.27 | 32.68 | 29.59 | 35.62 | 5 |
| 131072 | 60.45 | 55.91 | 46.41 | 89.29 | 123.26 | 108.23 | 129.62 | 5 |
| 524288 | 336.46 | 316.16 | 240.17 | 524.85 | 534.05 | 523.27 | 541.81 | 5 |
| 2097152 | 1010.17 | 1071.80 | 720.51 | 1332.29 | 2040.50 | 1848.03 | 2096.81 | 5 |
| 8388608 | 13095.64 | 11961.10 | 11458.50 | 17956.30 | 3775.45 | 2819.70 | 4092.58 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 1972.20 | 1832.00 | 1644.00 | 2522.00 | 5 |
| Boolean.BatchBoolean | 1.60 | 1.00 | 1.00 | 3.00 | 5 |
| CrossSection.BatchBoolean | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| Polygon.Sponge4 | 0.80 | 1.00 | 0.00 | 1.00 | 5 |
| Polygon.Zebra1 | 2.00 | 2.00 | 2.00 | 2.00 | 5 |
| Polygon.Zebra3 | 982.40 | 955.00 | 865.00 | 1143.00 | 5 |

