### Weekly Benchmarks

Commit: `5d35cedabad4b77c025ae7dff6f9f86af0579369`
Runner: `GitHub Actions 1000082921`
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
| 667 | Intersect12 Q->P | 2976.60 | 0.998 | 1158.00 | 1813.40 | 0.00 | 5.20 | 5 |
| 695 | Intersect12 P->Q | 1201.60 | 0.995 | 821.40 | 374.80 | 5.40 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 1073.80 | 0.999 | 397.40 | 675.20 | 0.20 | 1.00 | 5 |
| 84 | Intersect12 P->Q | 890.80 | 0.993 | 512.00 | 372.40 | 4.60 | 1.80 | 5 |
| 260 | Intersect12 Q->P | 428.00 | 0.986 | 208.00 | 214.20 | 2.00 | 3.80 | 5 |
| 406 | Intersect12 P->Q | 276.60 | 0.986 | 177.20 | 95.40 | 4.00 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 239.00 | 0.980 | 128.00 | 106.20 | 4.80 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 107.60 | 0.996 | 62.80 | 44.20 | 0.60 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 6.91 | 5.11 | 1.15 | 14.03 | 4.72 | 4.48 | 4.88 | 5 |
| 2048 | 3.41 | 3.43 | 2.57 | 4.75 | 6.23 | 6.05 | 6.53 | 5 |
| 8192 | 10.53 | 8.24 | 6.39 | 17.28 | 12.35 | 10.69 | 13.59 | 5 |
| 32768 | 41.24 | 33.49 | 17.08 | 69.48 | 35.26 | 32.67 | 37.30 | 5 |
| 131072 | 184.35 | 132.34 | 118.46 | 354.48 | 127.65 | 119.45 | 132.12 | 5 |
| 524288 | 716.49 | 549.81 | 453.51 | 1070.00 | 539.29 | 534.55 | 551.75 | 5 |
| 2097152 | 2560.94 | 2524.87 | 1844.61 | 3673.18 | 2048.46 | 1785.94 | 2126.88 | 5 |
| 8388608 | 23296.84 | 20598.10 | 18161.90 | 30766.50 | 3761.69 | 3135.92 | 3982.25 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Triangulation.LargeSmoothContour | 57.40 | 61.00 | 47.00 | 67.00 | 5 |
| Manifold.DeepChainDoesNotOverflowNumLeaves | 2338.00 | 2398.00 | 2039.00 | 2505.00 | 5 |
| Boolean.SmoothExtrusionCrossing | 50.80 | 48.00 | 30.00 | 68.00 | 5 |
| Boolean.BatchBoolean | 1.80 | 2.00 | 1.00 | 3.00 | 5 |
| CrossSection.BatchBoolean | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| CrossSectionComplex.MergeVertsSharedColumns | 23.40 | 22.00 | 20.00 | 30.00 | 5 |
| CrossSectionComplex.MergeVertsGrid | 96.80 | 84.00 | 67.00 | 134.00 | 5 |
| CrossSectionComplex.IncidenceClusterInRandomOrder | 216.60 | 229.00 | 186.00 | 235.00 | 5 |
| CrossSectionComplex.IncidenceParallelDiagonalSlivers | 690.60 | 670.00 | 652.00 | 759.00 | 5 |
| CrossSectionComplex.IncidenceUnevenLayout | 191.40 | 180.00 | 165.00 | 234.00 | 5 |
| CrossSectionComplex.IncidenceBrickWall | 309.80 | 295.00 | 255.00 | 375.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsLongLoop2048 | 8.40 | 9.00 | 6.00 | 10.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsLongLoop8192 | 7.80 | 7.00 | 7.00 | 9.00 | 5 |
| CrossSectionComplex.OutEdgesToPolygonsManySmallContours | 10.80 | 10.00 | 7.00 | 15.00 | 5 |
| CrossSectionComplex.SweepManySeparatedContours | 117.40 | 117.00 | 98.00 | 136.00 | 5 |
| CrossSectionComplex.SweepManySeparatedContoursRotated90 | 56.60 | 54.00 | 47.00 | 66.00 | 5 |
| CrossSectionComplex.SweepCrossingContours | 50.00 | 52.00 | 40.00 | 62.00 | 5 |
| CrossSectionComplex.OffsetRoundMergingContours | 48.60 | 46.00 | 44.00 | 58.00 | 5 |
| CrossSectionComplex.OffsetMiterWavyRing | 76.20 | 67.00 | 63.00 | 119.00 | 5 |
| CrossSectionComplex.DifferenceManyContoursFewCrossings | 185.60 | 180.00 | 144.00 | 226.00 | 5 |
| Polygon.Sponge4 | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| Polygon.Zebra1 | 2.40 | 2.00 | 2.00 | 4.00 | 5 |
| Polygon.Zebra3 | 1312.40 | 1252.00 | 1230.00 | 1562.00 | 5 |

