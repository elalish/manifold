### Weekly Benchmarks

Commit: `d056b8fb70de8c8e6d6bee2e639b71c20aa08d01`
Runner: `GitHub Actions 1000070563`
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
| 667 | Intersect12 Q->P | 2953.40 | 0.999 | 1137.60 | 1811.80 | 0.00 | 4.00 | 5 |
| 695 | Intersect12 P->Q | 1191.40 | 0.995 | 803.20 | 382.20 | 6.00 | 0.00 | 5 |
| 16 | Intersect12 Q->P | 1100.00 | 0.999 | 411.00 | 687.60 | 0.00 | 1.40 | 5 |
| 84 | Intersect12 P->Q | 967.80 | 0.994 | 573.00 | 389.60 | 3.80 | 1.40 | 5 |
| 260 | Intersect12 Q->P | 421.00 | 0.986 | 191.80 | 223.20 | 2.40 | 3.60 | 5 |
| 406 | Intersect12 P->Q | 324.20 | 0.987 | 199.60 | 120.40 | 4.20 | 0.00 | 5 |
| 551 | Intersect12 P->Q | 254.00 | 0.979 | 136.60 | 112.00 | 5.40 | 0.00 | 5 |
| 582 | Intersect12 P->Q | 103.80 | 0.996 | 58.20 | 45.20 | 0.40 | 0.00 | 5 |

Note: phase timings cover `Intersect12` and `Winding03` only; `Intersections (total)` is excluded from the denominator.

#### perfTest Size Sweep

| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Peak RSS mean (MB) | Peak RSS min (MB) | Peak RSS max (MB) | Runs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 3.14 | 2.80 | 2.39 | 4.37 | 4.71 | 4.52 | 4.89 | 5 |
| 2048 | 2.82 | 2.89 | 2.46 | 3.14 | 6.15 | 5.92 | 6.23 | 5 |
| 8192 | 10.18 | 10.06 | 7.17 | 13.49 | 12.42 | 11.30 | 13.42 | 5 |
| 32768 | 32.11 | 36.76 | 17.17 | 43.95 | 34.53 | 29.66 | 37.97 | 5 |
| 131072 | 105.03 | 107.54 | 71.86 | 150.49 | 123.82 | 113.53 | 138.48 | 5 |
| 524288 | 548.03 | 500.41 | 457.70 | 790.09 | 530.17 | 492.41 | 552.47 | 5 |
| 2097152 | 1857.96 | 1772.53 | 1549.77 | 2325.02 | 1991.68 | 1527.70 | 2120.61 | 5 |
| 8388608 | 19015.04 | 16475.50 | 16037.80 | 27466.30 | 3939.94 | 3169.00 | 4170.98 | 5 |

#### Existing Regression Tests

| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |
|---|---:|---:|---:|---:|---:|
| Manifold.DeepChainDoesNotOverflowNumLeaves | 2442.20 | 2435.00 | 2211.00 | 2839.00 | 5 |
| Boolean.BatchBoolean | 4.00 | 5.00 | 2.00 | 6.00 | 5 |
| CrossSection.BatchBoolean | 0.20 | 0.00 | 0.00 | 1.00 | 5 |
| Polygon.Sponge4 | 1.00 | 1.00 | 0.00 | 2.00 | 5 |
| Polygon.Zebra1 | 2.60 | 2.00 | 2.00 | 5.00 | 5 |
| Polygon.Zebra3 | 1197.80 | 1169.00 | 1129.00 | 1287.00 | 5 |

