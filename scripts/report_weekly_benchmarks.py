#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def print_file(path: Path) -> None:
    sys.stdout.write(path.read_text(encoding="utf-8-sig"))


def raw_run_files(suite_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in suite_dir.rglob("run*.txt")
        if path.is_file() and "build" not in path.relative_to(suite_dir).parts
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Report weekly benchmark results.")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("repeats", type=int)
    args = parser.parse_args()

    summary_path = args.suite_dir / "summary.md"
    result_path = args.suite_dir / "result.json"

    subprocess.run(
        [
            sys.executable,
            "./scripts/parse_weekly_benchmarks.py",
            "--source-dir",
            str(args.source_dir),
            "--suite-dir",
            str(args.suite_dir),
            "--repeats",
            str(args.repeats),
            "--markdown-out",
            str(summary_path),
            "--json-out",
            str(result_path),
        ],
        check=True,
    )

    print("::group::Weekly benchmark summary")
    print_file(summary_path)
    print("::endgroup::")

    print("::group::Weekly benchmark result.json")
    print_file(result_path)
    print("::endgroup::")

    print("::group::Weekly benchmark raw outputs")
    for run_file in raw_run_files(args.suite_dir):
        print(f"--- {run_file} ---")
        print_file(run_file)
    print("::endgroup::")

    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with Path(step_summary).open("a", encoding="utf-8") as out:
            out.write(summary_path.read_text(encoding="utf-8"))
            out.write("\n")
            out.write(
                "Raw logs: open this step and expand `Weekly benchmark result.json` "
                "and `Weekly benchmark raw outputs` groups.\n"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
