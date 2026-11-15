### Weekly Benchmarks

Commit: `9e23eff6f7edc7dddde38cbcf0b407ee93758e3d`
Runner: `GitHub Actions 1000074945`
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
| 667 | Intersect12 Q->P | 2699.40 | 0.999 | 1089.20 | 1606.60 | 0.00 | 3.60 | 5 |
| 695 | Intersect12 P->Q | 1169.20 | 0.996 | 800.80 | 363.40 | 5.00 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 1126.80 | 0.999 | 434.60 | 690.60 | 0.00 | 1.60 | 5 |
| 84 | Intersect12 P->Q | 916.80 | 0.991 | 523.80 | 385.20 | 6.60 | 1.20 | 5 |
| 260 | Intersect12 Q->P | 408.20 | 0.985 | 186.00 | 216.00 | 2.00 | 4.20 | 5 |
| 406 | Intersect12 P->Q | 368.40 | 0.990 | 221.80 | 143.00 | 3.60 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 273.80 | 0.977 | 152.20 | 115.40 | 6.20 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 107.80 | 0.996 | 65.60 | 41.80 | 0.40 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.59 | 1.66 | 0.97 | 2.22 | 4.54 | 4.38 | 4.78 | 5 |
| 2048 | 3.69 | 3.32 | 2.17 | 5.90 | 6.10 | 5.61 | 6.72 | 5 |
| 8192 | 10.63 | 11.13 | 5.75 | 13.80 | 12.35 | 11.22 | 14.56 | 5 |
| 32768 | 33.16 | 28.09 | 15.32 | 65.47 | 36.14 | 30.34 | 40.31 | 5 |
| 131072 | 170.47 | 123.87 | 57.85 | 339.57 | 128.25 | 121.39 | 134.06 | 5 |
| 524288 | 796.21 | 629.16 | 484.26 | 1452.81 | 533.74 | 524.80 | 549.00 | 5 |
| 2097152 | 2969.23 | 2252.44 | 1874.29 | 6321.93 | 2032.44 | 1737.75 | 2116.66 | 5 |
| 8388608 | 20570.04 | 18693.10 | 17935.30 | 27175.40 | 3807.99 | 3152.14 | 4040.97 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 3037.80 | 3067.00 | 2673.00 | 3401.00 | 5 |
| Boolean.BatchBoolean | 2.80 | 3.00 | 2.00 | 4.00 | 5 |
| CrossSection.BatchBoolean | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| CrossSectionComplex.MergeVertsSharedColumns | 32.80 | 35.00 | 27.00 | 38.00 | 5 |
| CrossSectionComplex.MergeVertsGrid | 111.40 | 123.00 | 85.00 | 127.00 | 5 |
| CrossSectionComplex.IncidenceClusterInRandomOrder | 303.00 | 295.00 | 268.00 | 349.00 | 5 |
| CrossSectionComplex.IncidenceParallelDiagonalSlivers | 799.00 | 783.00 | 752.00 | 865.00 | 5 |
| CrossSectionComplex.IncidenceUnevenLayout | 258.00 | 233.00 | 226.00 | 346.00 | 5 |
| CrossSectionComplex.IncidenceBrickWall | 381.40 | 406.00 | 297.00 | 432.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsLongLoop2048 | 8.80 | 9.00 | 7.00 | 11.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsLongLoop8192 | 8.00 | 7.00 | 7.00 | 10.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsManySmallContours | 12.00 | 9.00 | 8.00 | 20.00 | 5 |
| CrossSectionComplex.SweepManySeparatedContours | 138.20 | 131.00 | 107.00 | 189.00 | 5 |
| CrossSectionComplex.SweepManySeparatedContoursRotated90 | 74.40 | 70.00 | 62.00 | 94.00 | 5 |
| CrossSectionComplex.SweepCrossingContours | 62.00 | 62.00 | 53.00 | 72.00 | 5 |
| CrossSectionComplex.OffsetRoundMergingContours | 48.80 | 49.00 | 40.00 | 59.00 | 5 |
| Polygon.Sponge4 | 1.00 | 1.00 | 1.00 | 1.00 | 5 |
| Polygon.Zebra1 | 2.20 | 2.00 | 2.00 | 3.00 | 5 |
| Polygon.Zebra3 | 1400.20 | 1341.00 | 1275.00 | 1640.00 | 5 |

