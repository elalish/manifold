#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import parse_weekly_benchmarks as pwb


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

    import argparse as _argparse
    parse_ns = _argparse.Namespace(
        suite_dir=args.suite_dir,
        source_dir=args.source_dir,
        commit_sha=None,
        workflow=None,
        runner=None,
        os_name=None,
        compiler=None,
        cpu_model=None,
        cpu_count=None,
    )
    metadata = pwb.resolve_metadata(parse_ns)
    suites = pwb.parse_suites(args.suite_dir, args.source_dir)
    markdown = pwb.build_summary(suites, metadata, args.repeats)
    payload = {
        "metadata": metadata,
        "config": {"repeats": args.repeats, "suites": list(suites.keys())},
        "suites": suites,
    }
    summary_path.write_text(markdown + "\n", encoding="utf-8")
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

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
