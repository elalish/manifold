#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

THINGI10K_URL = (
    "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main/raw_meshes"
)


def parse_cases(raw: str) -> list[int]:
    cases = []
    for part in raw.replace(",", " ").split():
        cases.append(int(part))
    if not cases:
        raise ValueError("No weekly benchmark cases were provided")
    return cases


def transform_args(prefix: str, transform: dict) -> list[str]:
    args = [prefix]
    for col in ("col0", "col1", "col2", "col3"):
        for row in ("x", "y", "z"):
            args.append(f"{float(transform[col][row]):f}")
    return args


def download_mesh(mesh_id: int, mesh_dir: Path) -> Path:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    path = mesh_dir / f"{mesh_id}.stl"
    if path.exists() and path.stat().st_size > 0:
        return path

    tmp = path.with_suffix(".stl.part")
    url = f"{THINGI10K_URL}/{mesh_id}.stl"
    with urllib.request.urlopen(url, timeout=120) as response:
        with tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
    tmp.replace(path)
    return path


def load_specs(spec_path: Path, cases: list[int]) -> list[dict]:
    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    selected = []
    for case_index in cases:
        if case_index < 0 or case_index >= len(specs):
            raise IndexError(f"Case index {case_index} is outside 0..{len(specs) - 1}")
        spec = dict(specs[case_index])
        spec["case_index"] = case_index
        selected.append(spec)
    return selected


def write_cases_metadata(out_dir: Path, specs: list[dict]) -> None:
    with (out_dir / "cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_index", "id_a", "id_b"])
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "case_index": spec["case_index"],
                    "id_a": spec["id_a"],
                    "id_b": spec["id_b"],
                }
            )


def run_case(
    binary: Path, spec: dict, mesh_dir: Path, threads: int, verbose: int, cwd: Path
) -> str:
    mesh_a = download_mesh(int(spec["id_a"]), mesh_dir)
    mesh_b = download_mesh(int(spec["id_b"]), mesh_dir)
    args = [
        str(binary),
        str(mesh_a),
        str(mesh_b),
        *transform_args("-t1", spec["transform_a"]),
        *transform_args("-t2", spec["transform_b"]),
        "--threads",
        str(threads),
        "--verbose",
        str(verbose),
    ]
    result = subprocess.run(args, check=True, capture_output=True, text=True, cwd=cwd)
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Run selected Ember weekly benchmarks.")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--mesh-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--verbose", default=2, type=int)
    args = parser.parse_args()

    cases = parse_cases(args.cases)
    specs = load_specs(args.spec, cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_cases_metadata(args.out_dir, specs)

    for run_index in range(1, args.repeats + 1):
        run_dir = args.out_dir / f"run{run_index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_file = args.out_dir / f"run{run_index}.txt"
        benchmark_csv = args.out_dir / "benchmark.csv"
        benchmark_csv.unlink(missing_ok=True)
        with run_file.open("w", encoding="utf-8") as out:
            for spec in specs:
                out.write(
                    f"### case {spec['case_index']} "
                    f"({spec['id_a']} vs {spec['id_b']})\n"
                )
                stdout = run_case(
                    args.binary.resolve(),
                    spec,
                    args.mesh_dir.resolve(),
                    args.threads,
                    args.verbose,
                    args.out_dir.resolve(),
                )
                out.write(stdout)
                if not stdout.endswith("\n"):
                    out.write("\n")

        if benchmark_csv.exists():
            benchmark_csv.replace(args.out_dir / f"benchmark_run{run_index}.csv")

        print(f"completed weekly benchmark run {run_index}/{args.repeats}")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
