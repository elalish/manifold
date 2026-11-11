# Current fillet2d comparison

969 cases; 32 segments per circle. Reference: `fillet2d`. Times are medians of 5–21 calls per case, one warmup, one CPU, Release, sequential backend, debug disabled.

Agreement means symmetric-difference area / input area ≤ 1e-4 and identical contour count. Strict agreement uses 1e-6. These are area-based criteria, not pointwise or vertex-identical guarantees. An angle pass also requires the reference's 1.1 × segment-angle limit.

Empty reference outputs are reported separately: they can mean rejection, no change, or geometric extinction. They do not establish the correct geometric result. Cases whose estimated validation-grid area uncertainty exceeds 1e-6 of input area are excluded from agreement and containment statistics. The estimate is a conditioning diagnostic, not a certified clipping-error bound.

| Method | Empty | Malformed/exception | Nonempty reference, resolved | Strict agreement | Agreement | Agreement + angle pass | Sharp-angle cases | Containment violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fillet2d | 220 | 0 | 747 | 747 | 747 | 747 | 0 | 0 |
| boolean2_offset | 112 | 0 | 747 | 273 | 559 | 535 | 69 | 406 |
| clipper2_offset | 115 | 0 | 747 | 118 | 465 | 429 | 92 | 402 |

Malformed checks cover finite coordinates, nonzero edges, and at least three vertices per contour. Containment violations mean material outside the input for positive radius, or input material missing for negative radius, above 1e-6 of input area. Regularization-area residuals are in the CSV; these checks do not prove absence of self-intersections.

| Candidate | Median candidate/reference time, all paired nonempty reference cases | Same ratio, agreement + angle pass | Paired cases | Qualified cases | Nonempty candidate when reference empty |
|---|---:|---:|---:|---:|---:|
| boolean2_offset | 7.40× | 6.24× | 749 | 535 | 108 |
| clipper2_offset | 0.66× | 1.00× | 749 | 429 | 105 |

A ratio below 1 means faster than fillet2d. Qualified speed ratios use each candidate's own passing subset; subsets differ. Timing includes conversion to output polygons, fillet decomposition, both offset passes and cleanup, and Clipper integer conversion. Input construction and validation are excluded.

On the same 386 cases where **both** candidates meet agreement and angle criteria, median candidate/reference time ratios are boolean2_offset: 13.06×, clipper2_offset: 0.85×.

## Positive radii

| Candidate | Resolved nonempty-reference cases | XOR/input median | XOR/input max | Contour mismatches |
|---|---:|---:|---:|---:|
| boolean2_offset | 430 | 1.05887e-06 | 0.679553 | 18 |
| clipper2_offset | 430 | 4.47399e-05 | 0.0113821 | 4 |

## Negative radii

| Candidate | Resolved nonempty-reference cases | XOR/input median | XOR/input max | Contour mismatches |
|---|---:|---:|---:|---:|
| boolean2_offset | 317 | 1.89433e-05 | 1.16065 | 38 |
| clipper2_offset | 317 | 2.60739e-05 | 0.00129779 | 5 |

## Largest resolved differences

| Candidate | Case | XOR/input | Reference/candidate contours | Sharp turns |
|---|---|---:|---:|---:|
| boolean2_offset | polygon_corpus/KissingZigzag/-0.900000 | 1.16065 | 1/5 | 46 |
| boolean2_offset | polygon_corpus/Sponge4b/-0.900000 | 0.851374 | 6/47 | 0 |
| boolean2_offset | polygon_corpus/SimpleHoles2/-0.900000 | 0.810586 | 1/1 | 0 |
| boolean2_offset | polygon_corpus/Holes3/-0.900000 | 0.763326 | 1/2 | 0 |
| boolean2_offset | polygon_corpus/Holes7/-0.900000 | 0.719179 | 1/3 | 0 |
| clipper2_offset | stress/neck/s=0.000001/t=0.000000/0.100000 | 0.0113821 | 1/2 | 0 |
| clipper2_offset | stress/neck/s=1.000000/t=0.000000/0.100000 | 0.0113821 | 1/2 | 0 |
| clipper2_offset | stress/neck/s=1000000.000000/t=0.000000/0.100000 | 0.0113821 | 1/2 | 0 |
| clipper2_offset | stress/neck/s=1000000.000000/t=100000000.000000/0.100000 | 0.0113821 | 1/2 | 0 |
| clipper2_offset | polygon_corpus/KissingZigzag/0.900000 | 0.00214467 | 1/1 | 0 |

## Large fixtures

| Case | Input vertices | Method | Median ms | XOR/input | Contours |
|---|---:|---|---:|---:|---:|
| sponge/Sponge4/0.500000 | 184 | fillet2d | 1.525 | 0 | 7 |
| sponge/Sponge4/0.500000 | 184 | boolean2_offset | 5.625 | 0.000581162 | 9 |
| sponge/Sponge4/0.500000 | 184 | clipper2_offset | 0.630 | 0.000750143 | 7 |
| sponge/Sponge4/-0.500000 | 184 | fillet2d | 0.695 | 0 | 3 |
| sponge/Sponge4/-0.500000 | 184 | boolean2_offset | 3.495 | 0.000134029 | 3 |
| sponge/Sponge4/-0.500000 | 184 | clipper2_offset | 0.299 | 8.58424e-05 | 3 |
| sponge/Sponge3/0.500000 | 84 | fillet2d | 0.780 | 0 | 5 |
| sponge/Sponge3/0.500000 | 84 | boolean2_offset | 5.114 | 0.0277913 | 11 |
| sponge/Sponge3/0.500000 | 84 | clipper2_offset | 0.558 | 0.000415552 | 5 |
| sponge/Sponge3/-0.500000 | 84 | fillet2d | 0.888 | 0 | 3 |
| sponge/Sponge3/-0.500000 | 84 | boolean2_offset | 2.183 | 2.24827e-05 | 3 |
| sponge/Sponge3/-0.500000 | 84 | clipper2_offset | 0.183 | 2.23142e-05 | 3 |
| sponge/SpongeMore/0.500000 | 15 | fillet2d | 0.093 | 0 | 1 |
| sponge/SpongeMore/0.500000 | 15 | boolean2_offset | 0.932 | 0.000223335 | 1 |
| sponge/SpongeMore/0.500000 | 15 | clipper2_offset | 0.060 | 0.000388932 | 1 |
| sponge/SpongeMore/-0.500000 | 15 | fillet2d | 0.015 | 0 | 1 |
| sponge/SpongeMore/-0.500000 | 15 | boolean2_offset | 0.799 | 8.54687e-06 | 1 |
| sponge/SpongeMore/-0.500000 | 15 | clipper2_offset | 0.052 | 7.95157e-06 | 1 |
| sponge/Sponge3b/0.500000 | 132 | fillet2d | 0.982 | 0 | 7 |
| sponge/Sponge3b/0.500000 | 132 | boolean2_offset | 4.841 | 0.00021027 | 7 |
| sponge/Sponge3b/0.500000 | 132 | clipper2_offset | 0.366 | 0.000387266 | 7 |
| sponge/Sponge3b/-0.500000 | 132 | fillet2d | 0.792 | 0 | 16 |
| sponge/Sponge3b/-0.500000 | 132 | boolean2_offset | 8.734 | 7.33003e-05 | 16 |
| sponge/Sponge3b/-0.500000 | 132 | clipper2_offset | 0.639 | 7.98483e-05 | 16 |
| sponge/Sponge3a/0.500000 | 159 | fillet2d | 2.347 | 0 | 7 |
| sponge/Sponge3a/0.500000 | 159 | boolean2_offset | 13.486 | 0.000687146 | 7 |
| sponge/Sponge3a/0.500000 | 159 | clipper2_offset | 1.895 | 0.000737376 | 7 |
| sponge/Sponge3a/-0.500000 | 159 | fillet2d | 0.580 | 0 | 4 |
| sponge/Sponge3a/-0.500000 | 159 | boolean2_offset | 2.536 | 5.40505e-06 | 4 |
| sponge/Sponge3a/-0.500000 | 159 | clipper2_offset | 0.198 | 7.41902e-06 | 4 |
| sponge/BigSponge/0.500000 | 1197 | fillet2d | 14.901 | 0 | 57 |
| sponge/BigSponge/0.500000 | 1197 | boolean2_offset | 163.708 | 0.000648139 | 57 |
| sponge/BigSponge/0.500000 | 1197 | clipper2_offset | 23.432 | 0.000604211 | 57 |
| sponge/BigSponge/-0.500000 | 1197 | fillet2d | 4.294 | 0 | 34 |
| sponge/BigSponge/-0.500000 | 1197 | boolean2_offset | 11.122 | 3.76849e-05 | 34 |
| sponge/BigSponge/-0.500000 | 1197 | clipper2_offset | 0.906 | 8.56171e-06 | 34 |
| zebra/Zebra1/0.500000 | 569 | fillet2d | 2.649 | 0 | 2 |
| zebra/Zebra1/0.500000 | 569 | boolean2_offset | 1.650 | 0.147477 | 4 |
| zebra/Zebra1/0.500000 | 569 | clipper2_offset | 0.392 | 0.000740941 | 2 |
| zebra/Zebra1/-0.500000 | 569 | fillet2d | 2.183 | 0 | 1 |
| zebra/Zebra1/-0.500000 | 569 | boolean2_offset | 7.626 | 0.016305 | 2 |
| zebra/Zebra1/-0.500000 | 569 | clipper2_offset | 2.128 | 0.000487777 | 1 |
| zebra/Zebra/0.500000 | 62270 | fillet2d | 1427.932 | 0 | 38 |
| zebra/Zebra/0.500000 | 62270 | boolean2_offset | 306.069 | 0.0311211 | 54 |
| zebra/Zebra/0.500000 | 62270 | clipper2_offset | 2766.832 | 0.000247597 | 38 |
| zebra/Zebra/-0.500000 | 62270 | fillet2d | 1323.359 | 0 | 26 |
| zebra/Zebra/-0.500000 | 62270 | boolean2_offset | 334.039 | 0.0780749 | 72 |
| zebra/Zebra/-0.500000 | 62270 | clipper2_offset | 2963.537 | 0.000241498 | 26 |
| zebra3/Zebra3/0.500000 | 65002 | fillet2d | 1541.616 | 0 | 38 |
| zebra3/Zebra3/0.500000 | 65002 | boolean2_offset | 316.091 | 0.0295742 | 51 |
| zebra3/Zebra3/0.500000 | 65002 | clipper2_offset | 2895.569 | 0.000247016 | 38 |
| zebra3/Zebra3/-0.500000 | 65002 | fillet2d | 1440.417 | 0 | 26 |
| zebra3/Zebra3/-0.500000 | 65002 | boolean2_offset | 339.198 | 0.0714626 | 70 |
| zebra3/Zebra3/-0.500000 | 65002 | clipper2_offset | 3048.693 | 0.000240845 | 26 |
