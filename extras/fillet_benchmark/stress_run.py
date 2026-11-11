#!/usr/bin/env python3
"""Deadline-bounded stress sweep; each subprocess has an independent replay key."""
import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("binary", type=Path)
p.add_argument("root", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--seconds", type=int, default=1200)
p.add_argument("--seeds", type=int, default=256)
p.add_argument("--transforms", type=int, default=4)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--cases", nargs="+", help="Optional exact corpus names")
p.add_argument("--segments", nargs="+", type=int, default=[20, 8, 32, 64])
p.add_argument("--strict", action="store_true", help="Original nonempty/negative monotonic-area checks; use selected known-roundable cases")
args = p.parse_args()
args.binary = args.binary.resolve()
args.root = args.root.resolve()
args.output.mkdir(parents=True, exist_ok=True)
start = time.monotonic()
deadline = start + args.seconds
cases = []
for file in sorted((args.root / "test/polygons").glob("*.txt")):
    listing = subprocess.check_output([args.binary, file, "--list"], text=True)
    for line in listing.splitlines():
        name, vertices = line.split()
        if args.cases and name not in args.cases:
            continue
        cases.append((file, name, int(vertices)))
# Breadth first: cover every shape before extending seed coverage. Larger cases
# get a smaller quota so Zebra cannot monopolize the deadline.
jobs = []
for segments in args.segments:
    budget = args.seeds if segments == 20 else min(args.seeds, 64)
    for first in range(1, budget + 1, 16):
        for file, name, vertices in cases:
            limit = min(8, budget) if vertices > 5000 else budget
            if first > limit:
                continue
            seeds = min(16, limit-first+1)
            count = 1 if vertices > 5000 else args.transforms
            jobs.append((file, name, segments, first, seeds, count))

def run(job):
    file, name, segments, first, seeds, count = job
    prefix = args.output / f"{file.stem}-{name}-n{segments}-s{first}"
    command = [str(args.binary), str(file), name, str(first), str(seeds),
               str(count), str(segments), str(prefix)]
    if args.strict:
        command.append("--strict")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    row = dict(file=file.name, case=name, segments=segments, first_seed=first,
               seeds=seeds, transforms=count, replay=command)
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              timeout=min(120, remaining))
        values = proc.stdout.strip().split(',')
        row.update(status="ok" if proc.returncode == 0 and len(values) == 9
                   else "error", returncode=proc.returncode)
        if row["status"] == "ok":
            keys = ("calls", "empty", "empty_flips", "invalid", "sharp",
                    "wrong_direction", "area_drift", "max_area_drift", "seconds")
            row.update(zip(keys, map(float, values)))
            if args.strict:
                row["strict_failures"] = sum(1 for _ in
                    Path(str(prefix)+".strict.csv").open())-1
        else:
            row.update(stdout=proc.stdout, stderr=proc.stderr[-4000:])
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
    return row

metadata = dict(source_commit=subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=args.root, text=True).strip(),
    arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
    case_count=len(cases), planned_jobs=len(jobs), factors=[-.5,-1,-2,-5,-10,-20,
    -50,-100,.1,.3,.5,.7,.9], build="Release, sequential, assertions/debug off")
(args.output / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
completed = 0
with (args.output / "jobs.jsonl").open("w") as log:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = set()
        iterator = iter(jobs)
        while True:
            while len(pending) < args.workers and time.monotonic() < deadline:
                job = next(iterator, None)
                if job is None:
                    break
                pending.add(pool.submit(run, job))
            if not pending:
                break
            done, pending = concurrent.futures.wait(
                pending, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                row = future.result()
                if row:
                    log.write(json.dumps(row)+"\n")
                    log.flush()
                    completed += 1
                    if completed % 100 == 0 or row["status"] != "ok":
                        print(f"{completed}/{len(jobs)} jobs; "
                              f"{time.monotonic()-start:.1f}s; "
                              f"{row['case']} {row['status']}", flush=True)
metadata.update(completed_jobs=completed, wall_seconds=time.monotonic()-start,
                deadline_reached=time.monotonic() >= deadline)
(args.output / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
print(json.dumps(metadata, indent=2), flush=True)
