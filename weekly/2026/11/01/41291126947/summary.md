### Weekly Benchmarks

Commit: `2f33e782c3c0f0b3f56f581398622d6441755244`
Runner: `GitHub Actions 1000070989`
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
| 667 | Intersect12 Q->P | 2415.20 | 0.999 | 931.80 | 1480.00 | 0.00 | 3.40 | 5 |
| 695 | Intersect12 P->Q | 951.00 | 0.995 | 658.80 | 287.00 | 5.20 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 856.80 | 0.999 | 325.40 | 530.20 | 0.00 | 1.20 | 5 |
| 84 | Intersect12 P->Q | 765.40 | 0.991 | 440.00 | 318.80 | 5.60 | 1.00 | 5 |
| 260 | Intersect12 Q->P | 311.80 | 0.983 | 150.00 | 156.40 | 2.00 | 3.40 | 5 |
| 406 | Intersect12 P->Q | 258.80 | 0.988 | 165.20 | 90.60 | 3.00 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 202.20 | 0.980 | 112.20 | 86.00 | 4.00 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 86.00 | 0.995 | 49.20 | 36.40 | 0.40 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 2.16 | 1.55 | 1.15 | 4.66 | 4.82 | 4.75 | 4.97 | 5 |
| 2048 | 2.97 | 3.06 | 2.62 | 3.37 | 6.17 | 6.00 | 6.31 | 5 |
| 8192 | 8.60 | 7.94 | 6.22 | 10.83 | 12.88 | 11.25 | 13.88 | 5 |
| 32768 | 30.41 | 31.10 | 15.51 | 52.93 | 36.42 | 33.30 | 42.17 | 5 |
| 131072 | 96.39 | 100.96 | 57.57 | 125.50 | 127.20 | 120.92 | 139.86 | 5 |
| 524288 | 592.28 | 574.56 | 406.06 | 868.94 | 533.81 | 524.47 | 539.83 | 5 |
| 2097152 | 2710.50 | 1830.91 | 1456.72 | 6786.13 | 2026.10 | 1718.66 | 2116.83 | 5 |
| 8388608 | 18497.22 | 17482.60 | 14009.90 | 26537.10 | 3778.94 | 3114.31 | 4025.22 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 2026.60 | 1990.00 | 1847.00 | 2186.00 | 5 |
| Boolean.BatchBoolean | 2.20 | 2.00 | 1.00 | 4.00 | 5 |
| CrossSection.BatchBoolean | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsLongLoop2048 | 16.80 | 17.00 | 16.00 | 17.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsLongLoop8192 | 17.00 | 17.00 | 15.00 | 20.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsManySmallContours | 24.00 | 24.00 | 23.00 | 25.00 | 5 |
| Polygon.Sponge4 | 0.40 | 0.00 | 0.00 | 1.00 | 5 |
| Polygon.Zebra1 | 2.00 | 2.00 | 2.00 | 2.00 | 5 |
| Polygon.Zebra3 | 1011.40 | 983.00 | 957.00 | 1093.00 | 5 |

