#!/usr/bin/env python3
"""Summarize reference agreement; never treat empty fillet output as an oracle."""
import csv
import statistics
import sys
from pathlib import Path


def median(values):
    return f"{statistics.median(values):.2f}" if values else "n/a"


def main():
    rows = list(csv.DictReader(Path(sys.argv[1]).open()))
    assert all(r.get("reference_method", "fillet2d") == "fillet2d" for r in rows), (
        "Use clipper_gt_report.py for Clipper2 ground-truth results")
    grouped = {method: [r for r in rows if r["method"] == method]
               for method in ("fillet2d", "boolean2_offset", "clipper2_offset")}
    reference = {r["case"]: r for r in grouped["fillet2d"]}
    print("# Current fillet2d comparison\n")
    print(f"{len(reference)} cases; 32 segments per circle. Reference: `fillet2d`. "
          "Times are medians of 5–21 calls per case, one warmup, one CPU, "
          "Release, sequential backend, debug disabled.\n")
    print("Agreement means symmetric-difference area / input area ≤ 1e-4 "
          "and identical contour count. Strict agreement uses 1e-6. These are "
          "area-based criteria, not pointwise or vertex-identical guarantees. "
          "An angle pass also requires the reference's 1.1 × segment-angle limit.\n")
    print("Empty reference outputs are reported separately: they can mean "
          "rejection, no change, or geometric extinction. They do not establish "
          "the correct geometric result. Cases whose estimated validation-grid "
          "area uncertainty exceeds 1e-6 of input area are excluded from agreement "
          "and containment statistics. The estimate is a conditioning diagnostic, "
          "not a certified clipping-error bound.\n")
    print("| Method | Empty | Malformed/exception | Nonempty reference, resolved | Strict agreement | Agreement | Agreement + angle pass | Sharp-angle cases | Containment violations |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    qualified = {}
    for method, data in grouped.items():
        active = [r for r in data if r["reference_empty"] == "0"
                  and float(r["oracle_error_bound"]) <= 1e-6]
        def matches(r, tol):
            return (r["valid"] == "1" and r["empty"] == "0"
                    and r["contours"] == r["reference_contours"]
                    and float(r["xor_over_input"]) <= tol)
        qualified[method] = [r for r in active if matches(r, 1e-4)
                             and r["sharp_turns"] == "0"]
        print(f"| {method} | {sum(r['empty']=='1' for r in data)} | "
              f"{sum(r['valid']=='0' for r in data)} | {len(active)} | "
              f"{sum(matches(r,1e-6) for r in active)} | "
              f"{sum(matches(r,1e-4) for r in active)} | {len(qualified[method])} | "
              f"{sum(int(r['sharp_turns'])>0 for r in active)} | "
              f"{sum(float(r['containment_over_input'])>1e-6 for r in active)} |")
    print("\nMalformed checks cover finite coordinates, nonzero edges, and at "
          "least three vertices per contour. Containment violations mean material "
          "outside the input for positive radius, or input material missing for "
          "negative radius, above 1e-6 of input area. Regularization-area residuals "
          "are in the CSV; these checks do not prove absence of self-intersections.\n")
    print("| Candidate | Median candidate/reference time, all paired nonempty reference cases | Same ratio, agreement + angle pass | Paired cases | Qualified cases | Nonempty candidate when reference empty |")
    print("|---|---:|---:|---:|---:|---:|")
    for method in ("boolean2_offset", "clipper2_offset"):
        data = grouped[method]
        paired = [r for r in data if r["reference_empty"] == "0" and r["empty"] == "0"
                  and r["valid"] == "1"]
        def ratios(data):
            return [float(r["median_us"]) / float(reference[r["case"]]["median_us"])
                    for r in data]
        print(f"| {method} | {median(ratios(paired))}× | "
              f"{median(ratios(qualified[method]))}× | {len(paired)} | "
              f"{len(qualified[method])} | "
              f"{sum(r['reference_empty']=='1' and r['empty']=='0' for r in data)} |")
    print("\nA ratio below 1 means faster than fillet2d. Qualified speed ratios "
          "use each candidate's own passing subset; subsets differ. Timing "
          "includes conversion to output polygons, fillet decomposition, both "
          "offset passes and cleanup, and Clipper integer conversion. Input "
          "construction and validation are excluded.\n")
    common = set(r["case"] for r in qualified["boolean2_offset"]) & set(
        r["case"] for r in qualified["clipper2_offset"])
    print(f"On the same {len(common)} cases where **both** candidates meet "
          "agreement and angle criteria, median candidate/reference time ratios "
          "are " + ", ".join(
              f"{method}: {median([float(r['median_us']) / float(reference[r['case']]['median_us']) for r in grouped[method] if r['case'] in common])}×"
              for method in ("boolean2_offset", "clipper2_offset")) + ".\n")
    for sign, predicate in (("Positive radii", lambda r: float(r["radius"]) > 0),
                            ("Negative radii", lambda r: float(r["radius"]) < 0)):
        print(f"## {sign}\n")
        print("| Candidate | Resolved nonempty-reference cases | XOR/input median | XOR/input max | Contour mismatches |")
        print("|---|---:|---:|---:|---:|")
        for method in ("boolean2_offset", "clipper2_offset"):
            data = [r for r in grouped[method] if predicate(r)
                    and r["reference_empty"] == "0" and r["empty"] == "0"
                    and float(r["oracle_error_bound"]) <= 1e-6]
            vals = [float(r["xor_over_input"]) for r in data]
            print(f"| {method} | {len(data)} | {statistics.median(vals):.6g} | "
                  f"{max(vals):.6g} | {sum(r['contours']!=r['reference_contours'] for r in data)} |")
        print()
    print("## Largest resolved differences\n")
    print("| Candidate | Case | XOR/input | Reference/candidate contours | Sharp turns |")
    print("|---|---|---:|---:|---:|")
    for method in ("boolean2_offset", "clipper2_offset"):
        data = [r for r in grouped[method] if r["reference_empty"] == "0"
                and r["empty"] == "0" and float(r["oracle_error_bound"]) <= 1e-6]
        for r in sorted(data, key=lambda r: float(r["xor_over_input"]), reverse=True)[:5]:
            print(f"| {method} | {r['case']} | {float(r['xor_over_input']):.6g} | "
                  f"{r['reference_contours']}/{r['contours']} | {r['sharp_turns']} |")
    large = [r for r in rows if r["case"].startswith(("sponge/", "zebra/", "zebra3/"))]
    if large:
        print("\n## Large fixtures\n")
        print("| Case | Input vertices | Method | Median ms | XOR/input | Contours |")
        print("|---|---:|---|---:|---:|---:|")
        for r in large:
            print(f"| {r['case']} | {r['input_vertices']} | {r['method']} | "
                  f"{float(r['median_us'])/1000:.3f} | "
                  f"{float(r['xor_over_input']):.6g} | {r['contours']} |")


if __name__ == "__main__":
    main()
