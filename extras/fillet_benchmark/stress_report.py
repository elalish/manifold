#!/usr/bin/env python3
"""Summarize stress_run.py output without treating every empty as a defect."""
import collections
import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "jobs.jsonl").read_text().splitlines()]
meta = json.loads((root / "metadata.json").read_text())
ok = [r for r in rows if r["status"] == "ok"]
print(f"# Fillet stress: `{meta['source_commit'][:8]}`\n")
print(f"Completed {len(ok):,} jobs; {len(rows)-len(ok):,} incomplete jobs. "
      f"Wall time: {meta.get('wall_seconds', 0):.1f} seconds.\n")
print("Calls below exclude the 13 untransformed reference calls per job and "
      "any work performed by timed-out processes.\n")
print("| Segments | Cases completed | Calls | Empty results | Empty flips | "
      "Invalid loops | Sharp turns | Wrong area direction |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|")
for segments in sorted({r['segments'] for r in rows}):
    group = [r for r in ok if r['segments'] == segments]
    cases = len({(r['file'], r['case']) for r in group})
    keys = ('calls', 'empty', 'empty_flips', 'invalid', 'sharp', 'wrong_direction')
    print(f"| {segments} | {cases} | " + " | ".join(
        f"{sum(r.get(k, 0) for r in group):,.0f}" for k in keys) + " |")
print("\nEmpty flips compare transformed inputs with the untransformed "
      "reference at the same normalized radius. They are observations, not "
      "automatically confirmed defects: Fillet deliberately rejects unchanged "
      "areas, and near-degenerate input cleanup can also change under transforms. "
      "Sharp turns use the existing positive-radius unit-test threshold "
      "(1.1 × 360° / segments). Negative-radius sharp turns are not checked.\n")
print("| Case | Segments | Empty flips | Largest area drift / input area |")
print("|---|---:|---:|---:|")
groups = collections.defaultdict(list)
for r in ok:
    groups[(r['file'], r['case'], r['segments'])].append(r)
for (file, name, segments), group in sorted(groups.items()):
    flips = sum(r['empty_flips'] for r in group)
    if flips:
        print(f"| {file}:{name} | {segments} | {flips:,.0f} | "
              f"{max(r['max_area_drift'] for r in group):.6g} |")
print("\nArea drift is diagnostic: arc tessellation changes with rotation, "
      "especially at 8 segments. It is not a correctness verdict. Finite-loop "
      "checks do not prove absence of intersections or geometric equivalence.\n")
if meta['arguments'].get('strict'):
    print("Original strict checks require nonempty results at all 13 radii "
          "and nondecreasing area across the eight negative radii, allowing "
          "1e-9 relative roundoff. These checks apply only to the selected "
          "known-roundable fixtures.\n")
    print("| Case | Calls | Strict failures |")
    print("|---|---:|---:|")
    for name in sorted({r['case'] for r in ok}):
        group = [r for r in ok if r['case'] == name]
        print(f"| {name} | {sum(r['calls'] for r in group):,.0f} | "
              f"{sum(r['strict_failures'] for r in group):,} |")
    failures = []
    for r in ok:
        prefix = Path(r['replay'][7]).name
        path = root / (prefix + '.strict.csv')
        if path.exists():
            for event in csv.DictReader(path.open()):
                failures.append(dict(file=r['file'], case=r['case'],
                                     segments=r['segments'], **event))
    if failures:
        with (root / 'failures.csv').open('w') as out:
            writer = csv.DictWriter(out, fieldnames=list(failures[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(failures)
        print(f"\nAll {len(failures):,} strict failure events are in `failures.csv`.\n")
bad = [r for r in rows if r['status'] != 'ok']
if bad:
    print("Incomplete jobs (not counted as successful coverage):\n")
    for r in bad:
        print(f"- {r['file']}:{r['case']}, segments={r['segments']}, "
              f"seeds={r['first_seed']}..{r['first_seed']+r['seeds']-1}: "
              f"{r['status']}")
