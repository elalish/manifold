### Weekly Benchmarks

Commit: `69f3c2fe66ead694e4e645d3168cfb5d68d3ed27`
Runner: `GitHub Actions 1000072403`
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
| 667 | Intersect12 Q->P | 2874.80 | 0.999 | 1149.20 | 1721.40 | 0.00 | 4.20 | 5 |
| 695 | Intersect12 P->Q | 1243.40 | 0.995 | 865.40 | 371.60 | 6.40 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 988.20 | 0.999 | 389.20 | 597.80 | 0.00 | 1.20 | 5 |
| 84 | Intersect12 P->Q | 880.60 | 0.991 | 503.20 | 369.60 | 6.20 | 1.60 | 5 |
| 260 | Intersect12 Q->P | 382.80 | 0.980 | 180.80 | 194.60 | 3.20 | 4.20 | 5 |
| 406 | Intersect12 P->Q | 341.00 | 0.991 | 220.40 | 117.60 | 3.00 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 229.80 | 0.978 | 130.40 | 94.40 | 5.00 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 111.00 | 0.997 | 64.20 | 46.40 | 0.40 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 3.48 | 2.56 | 1.24 | 7.34 | 4.77 | 4.61 | 4.92 | 5 |
| 2048 | 4.21 | 3.60 | 3.05 | 7.24 | 6.21 | 5.69 | 6.75 | 5 |
| 8192 | 13.99 | 15.29 | 6.98 | 18.23 | 13.25 | 12.05 | 14.12 | 5 |
| 32768 | 68.19 | 59.79 | 46.13 | 99.64 | 37.38 | 34.50 | 40.81 | 5 |
| 131072 | 200.46 | 209.64 | 147.57 | 255.62 | 126.24 | 114.23 | 133.23 | 5 |
| 524288 | 694.55 | 746.86 | 502.47 | 843.05 | 534.16 | 525.88 | 549.02 | 5 |
| 2097152 | 2817.10 | 2464.12 | 2314.70 | 3914.99 | 2041.69 | 1778.22 | 2115.55 | 5 |
| 8388608 | 28005.72 | 26753.50 | 24867.80 | 36095.70 | 3717.04 | 2994.34 | 3931.23 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 2969.00 | 2894.00 | 2725.00 | 3480.00 | 5 |
| Boolean.BatchBoolean | 2.60 | 2.00 | 2.00 | 4.00 | 5 |
| CrossSection.BatchBoolean | 0.40 | 0.00 | 0.00 | 1.00 | 5 |
| CrossSectionComplex.MergeVertsSharedColumns | 26.60 | 24.00 | 20.00 | 43.00 | 5 |
| CrossSectionComplex.MergeVertsGrid | 115.20 | 125.00 | 83.00 | 140.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsLongLoop2048 | 20.80 | 21.00 | 16.00 | 25.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsLongLoop8192 | 21.60 | 19.00 | 18.00 | 27.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsManySmallContours | 36.60 | 31.00 | 25.00 | 62.00 | 5 |
| CrossSectionComplex.SweepManySeparatedContours | 141.40 | 142.00 | 134.00 | 152.00 | 5 |
| CrossSectionComplex.SweepManySeparatedContoursRotated90 | 68.80 | 68.00 | 64.00 | 78.00 | 5 |
| CrossSectionComplex.SweepCrossingContours | 64.80 | 68.00 | 53.00 | 71.00 | 5 |
| CrossSectionComplex.OffsetRoundMergingContours | 62.80 | 63.00 | 61.00 | 64.00 | 5 |
| Polygon.Sponge4 | 1.20 | 1.00 | 1.00 | 2.00 | 5 |
| Polygon.Zebra1 | 3.00 | 2.00 | 2.00 | 5.00 | 5 |
| Polygon.Zebra3 | 1384.00 | 1365.00 | 1307.00 | 1523.00 | 5 |

