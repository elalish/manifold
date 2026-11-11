# fillet2d versus Boolean2 offset, Clipper2 ground truth

969 cases, 32 segments per circle, Release, sequential backend, CPU 0. Each time is the median of 5–21 calls after warmup. Ground truth is Clipper2's round inset/outset (or outset/inset for negative radii), with adaptive integer scaling.

Geometry match requires equal empty/nonempty status, identical contour count, and symmetric-difference area / input area ≤ 1e-4 (0.01%). Strict match uses 1e-6 (0.0001%); loose match uses 1e-3 (0.1%). These are area and contour criteria, not exact vertices or a pointwise distance guarantee. Fillet2d's angle limit is diagnostic only; Clipper2 ground truth itself may exceed it.

All methods share 934 resolved cases; 35 ill-conditioned area comparisons are excluded. Of these, 835 have nonempty Clipper2 ground truth and 99 have empty ground truth.

| Method | Matches, all resolved | Strict matches | Loose matches | Empty despite nonempty GT | Nonempty despite empty GT | Contour mismatch, both nonempty |
|---|---:|---:|---:|---:|---:|---:|
| fillet2d | 564/934 | 217 | 806 | 88 | 0 | 9 |
| boolean2_offset | 633/934 | 333 | 847 | 0 | 5 | 54 |

Fillet2d returns empty on some no-change or rejected cases, while offsets can return the input unchanged. Empty-status mismatches therefore include API behavior differences, not only numerical failures.

## Accuracy where both methods and ground truth return geometry

| Method | Cases | Matches | Median XOR/input | Worst XOR/input | Sharp-angle cases |
|---|---:|---:|---:|---:|---:|
| fillet2d | 747 | 465 | 3.23617e-05 | 0.0113821 | 0 |
| boolean2_offset | 747 | 451 | 2.22842e-05 | 1.15973 | 69 |

## Direct runtime comparison by input size

Timings below use the same cases for both methods, with nonempty ground truth and nonempty outputs. They are **not restricted to matching geometry**. Sum of medians means one invocation per case in the given workload; the median ratio weights each case equally. Offset/fillet below 1 means Boolean2 offset is faster.

| Input vertices | Cases | Fillet sum of medians, ms | Offset sum of medians, ms | Offset/fillet workload time | Median per-case offset/fillet |
|---|---:|---:|---:|---:|---:|
| ≤100 | 705 | 42.605 | 603.309 | 14.160× | 8.573× |
| 101–1,000 | 36 | 46.546 | 347.076 | 7.457× | 7.054× |
| 1,001–10,000 | 2 | 18.714 | 174.082 | 9.302× | 6.948× |
| >10,000 | 4 | 4468.034 | 1290.790 | 0.289× | 0.287× |
| All paired | 747 | 4575.898 | 2415.257 | 0.528× | 7.983× |

On the same **428 cases where both match Clipper2 GT**, offset/fillet workload time is **12.392×** and median per-case ratio is **13.785×**.

## Large zebra results

| Case | Method | Time, ms | XOR/input | Contours (GT) |
|---|---|---:|---:|---:|
| zebra/Zebra/0.500000 | fillet2d | 1129.562 | 0.000247597 | 38 (38) |
| zebra/Zebra/0.500000 | boolean2_offset | 303.404 | 0.0311615 | 54 (38) |
| zebra/Zebra/0.500000 | clipper2_offset | 2716.542 | 0 | 38 (38) |
| zebra/Zebra/-0.500000 | fillet2d | 1020.003 | 0.000241498 | 26 (26) |
| zebra/Zebra/-0.500000 | boolean2_offset | 333.595 | 0.0781108 | 72 (26) |
| zebra/Zebra/-0.500000 | clipper2_offset | 2908.504 | 0 | 26 (26) |
| zebra3/Zebra3/0.500000 | fillet2d | 1207.539 | 0.000247016 | 38 (38) |
| zebra3/Zebra3/0.500000 | boolean2_offset | 313.949 | 0.0296155 | 51 (38) |
| zebra3/Zebra3/0.500000 | clipper2_offset | 2829.049 | 0 | 38 (38) |
| zebra3/Zebra3/-0.500000 | fillet2d | 1110.929 | 0.000240845 | 26 (26) |
| zebra3/Zebra3/-0.500000 | boolean2_offset | 339.843 | 0.0715021 | 70 (26) |
| zebra3/Zebra3/-0.500000 | clipper2_offset | 2996.303 | 0 | 26 (26) |

## Largest resolved differences

| Method | Case | XOR/input | Contours (GT) |
|---|---|---:|---:|
| fillet2d | stress/neck/s=0.000001/t=0.000000/0.100000 | 0.0113821 | 1 (2) |
| fillet2d | stress/neck/s=1.000000/t=0.000000/0.100000 | 0.0113821 | 1 (2) |
| fillet2d | stress/neck/s=1000000.000000/t=0.000000/0.100000 | 0.0113821 | 1 (2) |
| fillet2d | stress/neck/s=1000000.000000/t=100000000.000000/0.100000 | 0.0113821 | 1 (2) |
| fillet2d | polygon_corpus/KissingZigzag/0.900000 | 0.00214467 | 1 (1) |
| boolean2_offset | polygon_corpus/KissingZigzag/-0.900000 | 1.15973 | 5 (1) |
| boolean2_offset | polygon_corpus/Sponge4b/-0.900000 | 0.851344 | 47 (6) |
| boolean2_offset | polygon_corpus/SimpleHoles2/-0.900000 | 0.810512 | 1 (1) |
| boolean2_offset | polygon_corpus/Holes3/-0.900000 | 0.763318 | 2 (1) |
| boolean2_offset | polygon_corpus/Holes7/-0.900000 | 0.719143 | 3 (1) |

Validation uses Clipper2 XOR at 1e12 units per input span, sharing an implementation with the ground truth. The conditioning estimate is not a certified clipping-error bound. See README.md for case construction, scaling, timing inclusions, and validation limits.
