#!/usr/bin/env python3
"""Summarize the exact final-validator rejection reasons."""
import collections
import csv
import sys
from pathlib import Path


def main():
    rows = list(csv.DictReader(Path(sys.argv[1]).open()))
    counts = collections.Counter(r["reason"] for r in rows)
    constructed = [r for r in rows if int(r["raw_vertices"]) > 0]
    angle_failed = [r for r in constructed if r["angles_valid"] == "0"]
    print("# fillet2d final-validation rejection measurement\n")
    print(f"Same {len(rows)} cases as the Clipper2-ground-truth benchmark, "
          "32 segments per circle, Release, sequential backend. Instrumentation "
          "uses a generated build-local copy of fillet2d.cpp and preserves the "
          "original validation order and strict comparisons. Production source "
          "is unchanged. Diagnostic timings are not performance measurements.\n")
    print(f"**{counts['sharp_angle']} calls were rejected by the angle check.** "
          f"Of {len(constructed)} constructed nonempty candidates, "
          f"{len(angle_failed)} failed the angle predicate, including candidates "
          "that might have failed an earlier check.\n")
    print("| Final outcome | Cases | Positive radius | Negative radius |")
    print("|---|---:|---:|---:|")
    for reason in ("accepted", "trace_empty", "area_unchanged", "sharp_angle",
                   "invalid_geometry", "area_wrong_direction", "input_rejected", "exception"):
        data = [r for r in rows if r["reason"] == reason]
        print(f"| {reason} | {len(data)} | "
              f"{sum(float(r['radius'])>0 for r in data)} | "
              f"{sum(float(r['radius'])<0 for r in data)} |")
    limit = float(rows[0]["angle_limit_degrees"])
    worst = max(constructed, key=lambda r: float(r["max_turn_degrees"]))
    print(f"\nThe angle threshold was **{limit:g}°**. The largest measured turn "
          f"in the rounded direction before final rejection was "
          f"**{float(worst['max_turn_degrees']):.8f}°**, in `{worst['case']}`. "
          "Corners in the opposite direction are outside this angle criterion.\n")
    print("`trace_empty` means tracing already returned no geometry. This stage "
          "does not distinguish genuine geometric extinction from tracing "
          "failure, so these calls cannot all be labeled circle-fit failures. "
          "`area_unchanged` means nonempty geometry passed geometry and angle "
          "checks but failed the requirement that positive radius strictly "
          "decrease area or negative radius strictly increase area.\n")
    print("| Outcome | Clipper2 empty | Clipper2 nonempty |")
    print("|---|---:|---:|")
    for reason in counts:
        data = [r for r in rows if r["reason"] == reason]
        print(f"| {reason} | {sum(r['clipper_empty']=='1' for r in data)} | "
              f"{sum(r['clipper_empty']=='0' for r in data)} |")
    if angle_failed:
        print("\n## Nonempty candidates failing the angle check\n")
        print("| Case | Rejection reason | Sharp turns | Largest turn, degrees |")
        print("|---|---|---:|---:|")
        for r in angle_failed:
            print(f"| {r['case']} | {r['reason']} | {r['sharp_turns']} | "
                  f"{float(r['max_turn_degrees']):.8f} |")
    print("\nRaw per-case diagnostics are in `rejections.csv`. "
          "Any nonempty angle-failing geometry is written before clearing to "
          "`rejections.csv.raw.txt` in polygon-corpus format. An empty raw file "
          "means no such cases were observed. This result is specific to these "
          "cases and this segment count; it is not a proof that angle rejection "
          "can never occur.")


if __name__ == "__main__":
    main()
