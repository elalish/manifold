# Updated fillet2d versus previous benchmark

Previous source: `6bd50c7f`. Updated source: `9b6ff7ef`. Same 969 cases, 32 segments/circle, Release, sequential backend, CPU 0, 5–21 timed calls per case, Clipper2 ground truth. Previous measurements are from a separate run; unchanged methods provide timing controls.

## Old/new fillet2d runtime

Speedup is old time / updated time; above 1 means faster. Compare the same nonempty outputs in both versions. Workload time sums the median time of one invocation per case.

| Input vertices | Cases | Old workload, ms | Updated workload, ms | Workload speedup | Median per-case speedup |
|---|---:|---:|---:|---:|---:|
| ≤100 | 707 | 46.028 | 42.654 | 1.079× | 1.022× |
| 101–1,000 | 36 | 50.031 | 46.546 | 1.075× | 1.037× |
| 1,001–10,000 | 2 | 19.034 | 18.714 | 1.017× | 1.017× |
| >10,000 | 4 | 5697.666 | 4468.034 | 1.275× | 1.280× |
| All paired | 749 | 5812.758 | 4575.947 | 1.270× | 1.022× |

## Unchanged timing controls

| Method | Old/updated workload ratio, all cases | Median old/updated per-case ratio |
|---|---:|---:|
| boolean2_offset | 0.996× | 1.002× |
| clipper2_offset | 0.999× | 1.003× |

## Geometry versus Clipper2 ground truth

The same 934 resolved cases are used for both versions. A match requires identical empty status and contour count, and XOR/input area ≤ 0.01%. Empty returns include no-change rejections.

| Version | Matches | Empty when GT nonempty | Contour mismatch, both nonempty | Worst XOR/input, both nonempty |
|---|---:|---:|---:|---:|
| Previous | 564 | 88 | 9 | 0.0113821 |
| Updated | 564 | 88 | 9 | 0.0113821 |

## Changed output summaries

0 cases changed empty status, vertex/contour count, or area beyond 1e-10 relative to output area. This compares output summaries, not exact polygon identity.

| Case | Old/new vertices | Old/new contours | Old/new XOR/input |
|---|---:|---:|---:|
