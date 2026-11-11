#!/usr/bin/env python3
"""Compare saved Clipper2-ground-truth runs on identical cases."""
import csv
import statistics
import sys
from pathlib import Path

from clipper_gt_report import matches


def read(path):
    rows = list(csv.DictReader(Path(path).open()))
    assert rows and all(r["reference_method"] == "clipper2_offset" for r in rows)
    return {(r["case"], r["method"]): r for r in rows}


def main():
    old, new = (read(p) for p in sys.argv[1:3])
    assert old.keys() == new.keys()
    cases = [case for case, method in new if method == "fillet2d"]
    paired = [case for case in cases if all(
        data[case, "fillet2d"]["empty"] == "0" for data in (old, new))]
    resolved = [case for case in cases if all(
        float(data[case, m]["oracle_error_bound"]) <= 1e-6
        for data in (old, new) for m in ("fillet2d", "boolean2_offset", "clipper2_offset"))]
    print("# Updated fillet2d versus previous benchmark\n")
    print("Previous source: `6bd50c7f`. Updated source: `9b6ff7ef`. "
          "Same 969 cases, 32 segments/circle, Release, sequential backend, "
          "CPU 0, 5–21 timed calls per case, Clipper2 ground truth. "
          "Previous measurements are from a separate run; unchanged methods "
          "provide timing controls.\n")
    print("## Old/new fillet2d runtime\n")
    print("Speedup is old time / updated time; above 1 means faster. "
          "Compare the same nonempty outputs in both versions. Workload time "
          "sums the median time of one invocation per case.\n")
    print("| Input vertices | Cases | Old workload, ms | Updated workload, ms | Workload speedup | Median per-case speedup |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, predicate in (("≤100", lambda n: n <= 100),
                             ("101–1,000", lambda n: 100 < n <= 1000),
                             ("1,001–10,000", lambda n: 1000 < n <= 10000),
                             (">10,000", lambda n: n > 10000),
                             ("All paired", lambda n: True)):
        group = [c for c in paired if predicate(int(new[c, "fillet2d"]["input_vertices"]))]
        if not group:
            continue
        a = [float(old[c, "fillet2d"]["median_us"]) for c in group]
        b = [float(new[c, "fillet2d"]["median_us"]) for c in group]
        print(f"| {label} | {len(group)} | {sum(a)/1000:.3f} | {sum(b)/1000:.3f} | "
              f"{sum(a)/sum(b):.3f}× | {statistics.median(x/y for x,y in zip(a,b)):.3f}× |")
    print("\n## Unchanged timing controls\n")
    print("| Method | Old/updated workload ratio, all cases | Median old/updated per-case ratio |")
    print("|---|---:|---:|")
    for method in ("boolean2_offset", "clipper2_offset"):
        a = [float(old[c, method]["median_us"]) for c in cases]
        b = [float(new[c, method]["median_us"]) for c in cases]
        print(f"| {method} | {sum(a)/sum(b):.3f}× | "
              f"{statistics.median(x/y for x,y in zip(a,b)):.3f}× |")
    print("\n## Geometry versus Clipper2 ground truth\n")
    print(f"The same {len(resolved)} resolved cases are used for both versions. "
          "A match requires identical empty status and contour count, and "
          "XOR/input area ≤ 0.01%. Empty returns include no-change rejections.\n")
    print("| Version | Matches | Empty when GT nonempty | Contour mismatch, both nonempty | Worst XOR/input, both nonempty |")
    print("|---|---:|---:|---:|---:|")
    for name, data in (("Previous", old), ("Updated", new)):
        r = [data[c, "fillet2d"] for c in resolved]
        nonempty = [x for x in r if x["empty"] == x["reference_empty"] == "0"]
        print(f"| {name} | {sum(matches(x) for x in r)} | "
              f"{sum(x['empty']=='1' and x['reference_empty']=='0' for x in r)} | "
              f"{sum(x['contours']!=x['reference_contours'] for x in nonempty)} | "
              f"{max(float(x['xor_over_input']) for x in nonempty):.6g} |")
    print("\n## Changed output summaries\n")
    # Area differences smaller than reported rounding are omitted here.
    changed = []
    for c in cases:
        a, b = old[c, "fillet2d"], new[c, "fillet2d"]
        count_changed = any(a[k] != b[k] for k in ("empty", "vertices", "contours"))
        area_changed = abs(float(a["area"]) - float(b["area"])) > 1e-10 * max(
            abs(float(a["area"])), abs(float(b["area"])), 1e-30)
        if count_changed or area_changed:
            changed.append(c)
    print(f"{len(changed)} cases changed empty status, vertex/contour count, or "
          "area beyond 1e-10 relative to output area. This compares output "
          "summaries, not exact polygon identity.\n")
    print("| Case | Old/new vertices | Old/new contours | Old/new XOR/input |")
    print("|---|---:|---:|---:|")
    for c in changed:
        a, b = old[c, "fillet2d"], new[c, "fillet2d"]
        print(f"| {c} | {a['vertices']}/{b['vertices']} | {a['contours']}/{b['contours']} | "
              f"{float(a['xor_over_input']):.6g}/{float(b['xor_over_input']):.6g} |")


if __name__ == "__main__":
    main()
