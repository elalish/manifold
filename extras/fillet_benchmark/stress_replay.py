#!/usr/bin/env python3
"""Save and classify every strict failure using build-local instrumentation."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('root', type=Path)
p.add_argument('stress', type=Path)
p.add_argument('probe', type=Path)
args = p.parse_args()
jobs = [json.loads(line) for line in (args.run/'jobs.jsonl').read_text().splitlines()]
failures = list(csv.DictReader((args.run/'failures.csv').open()))
out = args.run/'replays'
out.mkdir(exist_ok=True)
classified = []
for event in failures:
    seed, transform = int(event['seed']), int(event['transform'])
    job = next(j for j in jobs if j['file'] == event['file'] and
               j['case'] == event['case'] and j['first_seed'] <= seed <
               j['first_seed']+j['seeds'])
    name = f"{event['case']}-s{seed}-t{transform}"
    prefix = out/name
    saved = out/(name+'.txt')
    original = args.run/(Path(job['replay'][7]).name+
                        f'.failure-s{seed}-t{transform}.txt')
    if original.exists():
        shutil.copyfile(original, saved)
    else:
        subprocess.run([str(args.stress.resolve()),
                        str(args.root/'test/polygons'/event['file']),
                        event['case'], str(seed), '1', str(transform+1),
                        event['segments'], str(prefix), '--strict',
                        str(transform)], check=True, capture_output=True)
        shutil.copyfile(str(prefix)+'.replay.txt', saved)
    result = subprocess.check_output([str(args.probe.resolve()), str(saved),
                                      'REPRO', event['segments']], text=True)
    (out/(name+'.reasons.csv')).write_text(result)
    observed = next(row for row in csv.DictReader(result.splitlines())
                    if abs(float(row['factor'])-float(event['factor'])) < 1e-12)
    if int(observed['returned_pieces']) != 0:
        raise RuntimeError(f"Empty failure did not reproduce: {event}")
    classified.append(dict(**event, reason=observed['reason'],
                           sharp_turns=observed['sharp_turns'],
                           max_turn_degrees=observed['max_turn_degrees'],
                           input_area=observed['input_area'],
                           raw_area=observed['raw_area'], polygon=str(saved)))
if classified:
    with (args.run/'classified_failures.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(classified[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(classified)
print(f"Reproduced and classified {len(classified)} failures")
