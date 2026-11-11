#!/usr/bin/env python3
"""Compare fillet2d and Boolean2 round offsets with Clipper2 ground truth."""
import csv
import statistics
import sys
from pathlib import Path

METHODS = ("fillet2d", "boolean2_offset")


def matches(row, tolerance=1e-4):
    return (row["valid"] == "1"
            and row["empty"] == row["reference_empty"]
            and row["contours"] == row["reference_contours"]
            and float(row["xor_over_input"]) <= tolerance)


def main():
    rows = list(csv.DictReader(Path(sys.argv[1]).open()))
    assert rows and all(r["reference_method"] == "clipper2_offset" for r in rows)
    by_method = {m: {r["case"]: r for r in rows if r["method"] == m}
                 for m in (*METHODS, "clipper2_offset")}
    gt = by_method["clipper2_offset"]
    # The same denominator for both methods, including candidate empty returns.
    resolved = [case for case in gt if all(
        float(by_method[m][case]["oracle_error_bound"]) <= 1e-6
        for m in (*METHODS, "clipper2_offset"))]
    nonempty = [case for case in resolved if gt[case]["empty"] == "0"]
    paired = [case for case in nonempty if all(
        by_method[m][case]["empty"] == "0" and by_method[m][case]["valid"] == "1"
        for m in METHODS)]
    print("# fillet2d versus Boolean2 offset, Clipper2 ground truth\n")
    print(f"{len(gt)} cases, 32 segments per circle, Release, sequential backend, "
          "CPU 0. Each time is the median of 5–21 calls after warmup. "
          "Ground truth is Clipper2's round inset/outset (or outset/inset for "
          "negative radii), with adaptive integer scaling.\n")
    print("Geometry match requires equal empty/nonempty status, identical contour "
          "count, and symmetric-difference area / input area ≤ 1e-4 (0.01%). "
          "Strict match uses 1e-6 (0.0001%); loose match uses 1e-3 (0.1%). "
          "These are area and contour criteria, not exact vertices or a pointwise "
          "distance guarantee. Fillet2d's angle limit is diagnostic only; "
          "Clipper2 ground truth itself may exceed it.\n")
    print(f"All methods share {len(resolved)} resolved cases; "
          f"{len(gt)-len(resolved)} ill-conditioned area comparisons are excluded. "
          f"Of these, {len(nonempty)} have nonempty Clipper2 ground truth and "
          f"{len(resolved)-len(nonempty)} have empty ground truth.\n")
    print("| Method | Matches, all resolved | Strict matches | Loose matches | Empty despite nonempty GT | Nonempty despite empty GT | Contour mismatch, both nonempty |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for method in METHODS:
        data = [by_method[method][case] for case in resolved]
        print(f"| {method} | {sum(matches(r) for r in data)}/{len(data)} | "
              f"{sum(matches(r,1e-6) for r in data)} | "
              f"{sum(matches(r,1e-3) for r in data)} | "
              f"{sum(r['empty']=='1' and r['reference_empty']=='0' for r in data)} | "
              f"{sum(r['empty']=='0' and r['reference_empty']=='1' for r in data)} | "
              f"{sum(r['empty']=='0' and r['reference_empty']=='0' and r['contours']!=r['reference_contours'] for r in data)} |")
    print("\nFillet2d returns empty on some no-change or rejected cases, while "
          "offsets can return the input unchanged. Empty-status mismatches therefore "
          "include API behavior differences, not only numerical failures.\n")
    print("## Accuracy where both methods and ground truth return geometry\n")
    print("| Method | Cases | Matches | Median XOR/input | Worst XOR/input | Sharp-angle cases |")
    print("|---|---:|---:|---:|---:|---:|")
    for method in METHODS:
        data = [by_method[method][case] for case in paired]
        values = [float(r["xor_over_input"]) for r in data]
        print(f"| {method} | {len(data)} | {sum(matches(r) for r in data)} | "
              f"{statistics.median(values):.6g} | {max(values):.6g} | "
              f"{sum(int(r['sharp_turns'])>0 for r in data)} |")
    print("\n## Direct runtime comparison by input size\n")
    print("Timings below use the same cases for both methods, with nonempty ground "
          "truth and nonempty outputs. They are **not restricted to matching "
          "geometry**. Sum of medians means one invocation per case in the given "
          "workload; the median ratio weights each case equally. "
          "Offset/fillet below 1 means Boolean2 offset is faster.\n")
    print("| Input vertices | Cases | Fillet sum of medians, ms | Offset sum of medians, ms | Offset/fillet workload time | Median per-case offset/fillet |")
    print("|---|---:|---:|---:|---:|---:|")
    bands = [("≤100", lambda n: n <= 100),
             ("101–1,000", lambda n: 100 < n <= 1000),
             ("1,001–10,000", lambda n: 1000 < n <= 10000),
             (">10,000", lambda n: n > 10000),
             ("All paired", lambda n: True)]
    for label, predicate in bands:
        cases = [case for case in paired if predicate(int(gt[case]["input_vertices"]))]
        if not cases:
            continue
        f = [float(by_method["fillet2d"][case]["median_us"]) for case in cases]
        o = [float(by_method["boolean2_offset"][case]["median_us"]) for case in cases]
        print(f"| {label} | {len(cases)} | {sum(f)/1000:.3f} | {sum(o)/1000:.3f} | "
              f"{sum(o)/sum(f):.3f}× | {statistics.median(b/a for a,b in zip(f,o)):.3f}× |")
    common = [case for case in paired if all(matches(by_method[m][case]) for m in METHODS)]
    if common:
        f = [float(by_method["fillet2d"][case]["median_us"]) for case in common]
        o = [float(by_method["boolean2_offset"][case]["median_us"]) for case in common]
        print(f"\nOn the same **{len(common)} cases where both match Clipper2 GT**, "
              f"offset/fillet workload time is **{sum(o)/sum(f):.3f}×** "
              f"and median per-case ratio is **{statistics.median(b/a for a,b in zip(f,o)):.3f}×**.\n")
    print("## Large zebra results\n")
    print("| Case | Method | Time, ms | XOR/input | Contours (GT) |")
    print("|---|---|---:|---:|---:|")
    for case in gt:
        if int(gt[case]["input_vertices"]) <= 10000:
            continue
        for method in (*METHODS, "clipper2_offset"):
            r = by_method[method][case]
            print(f"| {case} | {method} | {float(r['median_us'])/1000:.3f} | "
                  f"{float(r['xor_over_input']):.6g} | {r['contours']} ({r['reference_contours']}) |")
    print("\n## Largest resolved differences\n")
    print("| Method | Case | XOR/input | Contours (GT) |")
    print("|---|---|---:|---:|")
    for method in METHODS:
        data = [by_method[method][case] for case in paired]
        for r in sorted(data, key=lambda r: float(r["xor_over_input"]), reverse=True)[:5]:
            print(f"| {method} | {r['case']} | {float(r['xor_over_input']):.6g} | "
                  f"{r['contours']} ({r['reference_contours']}) |")
    print("\nValidation uses Clipper2 XOR at 1e12 units per input span, sharing an "
          "implementation with the ground truth. The conditioning estimate is "
          "not a certified clipping-error bound. See README.md for case "
          "construction, scaling, timing inclusions, and validation limits.")


if __name__ == "__main__":
    main()
