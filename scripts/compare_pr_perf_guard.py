#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

# fallback: extract only time value when nTri label is not present
TIME_PATTERN = re.compile(r"time\s*=\s*([0-9]*\.?[0-9]+)\s*sec")
# primary: extract both nTri bucket and timing from perfTest output
TRI_TIME_PATTERN = re.compile(
    r"nTri\s*=\s*([0-9]+)\s*,\s*time\s*=\s*([0-9]*\.?[0-9]+)\s*sec"
)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    # keep stdev defined even for single-sample cases
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def parse_run(run_path: Path, run_index: int) -> dict:
    # parse one run*.txt into ordered benchmark samples
    benchmarks = []
    for line in run_path.read_text(encoding="utf-8").splitlines():
        tri_match = TRI_TIME_PATTERN.search(line)
        if tri_match:
            benchmark_key = f"nTri={tri_match.group(1)}"
            benchmarks.append(
                {"benchmark": benchmark_key, "time_sec": float(tri_match.group(2))}
            )
            continue

        time_match = TIME_PATTERN.search(line)
        if time_match:
            benchmark_key = f"benchmark_{len(benchmarks) + 1}"
            benchmarks.append(
                {"benchmark": benchmark_key, "time_sec": float(time_match.group(1))}
            )

    if not benchmarks:
        raise RuntimeError(f"No perf timing lines found in {run_path}")

    benchmark_names = [entry["benchmark"] for entry in benchmarks]
    if len(set(benchmark_names)) != len(benchmark_names):
        raise RuntimeError(f"Duplicate benchmark keys found in {run_path}")

    return {
        "path": str(run_path),
        "run_index": run_index,
        "benchmarks": benchmarks,
    }


def parse_suite(suite_dir: Path) -> dict:
    # parse all run*.txt and build per-benchmark aggregates across repeats
    run_files = sorted(suite_dir.glob("run*.txt"))
    if not run_files:
        raise RuntimeError(f"No run*.txt files found in {suite_dir}")

    runs = [parse_run(run_file, i + 1) for i, run_file in enumerate(run_files)]
    benchmark_order = [entry["benchmark"] for entry in runs[0]["benchmarks"]]

    for run in runs[1:]:
        run_order = [entry["benchmark"] for entry in run["benchmarks"]]
        if run_order != benchmark_order:
            raise RuntimeError(
                f"Benchmark layout mismatch in {run['path']}: expected {benchmark_order}, got {run_order}"
            )

    benchmark_samples = {benchmark: [] for benchmark in benchmark_order}
    for run in runs:
        for entry in run["benchmarks"]:
            benchmark_samples[entry["benchmark"]].append(entry["time_sec"])

    benchmarks = {}
    for benchmark in benchmark_order:
        samples = benchmark_samples[benchmark]
        benchmarks[benchmark] = {
            "samples_sec": samples,
            "mean_sec": mean(samples),
            "median_sec": statistics.median(samples),
            "stdev_sec": stdev(samples),
            "max_sec": max(samples),
            "n_runs": len(samples),
        }

    return {
        "runs": runs,
        "benchmarks": benchmarks,
        "benchmark_order": benchmark_order,
    }


def build_summary(
    base: dict, head: dict, warn_pct: float, warn_abs_ms: float
) -> tuple[str, bool, dict]:
    # Compare benchmark medians between base and head with dual-threshold warnings.
    if base["benchmark_order"] != head["benchmark_order"]:
        raise RuntimeError(
            "Benchmark set/order mismatch between base and head: "
            f"{base['benchmark_order']} vs {head['benchmark_order']}"
        )

    lines = []
    lines.append("### PR Benchmark Guard (perfTest)")
    lines.append("")
    lines.append("| Benchmark | Base median (sec) | Head median (sec) | Delta | Base ±stdev | Head ±stdev | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    per_benchmark = []
    regressed = False
    for benchmark in base["benchmark_order"]:
        base_mean = base["benchmarks"][benchmark]["mean_sec"]
        head_mean = head["benchmarks"][benchmark]["mean_sec"]
        base_median = base["benchmarks"][benchmark]["median_sec"]
        head_median = head["benchmarks"][benchmark]["median_sec"]
        base_stdev = base["benchmarks"][benchmark]["stdev_sec"]
        head_stdev = head["benchmarks"][benchmark]["stdev_sec"]
        delta_sec = head_median - base_median
        delta_ms = delta_sec * 1000.0
        pct = delta_sec / base_median * 100.0
        this_regressed = (pct >= warn_pct) and (delta_ms >= warn_abs_ms)
        regressed = regressed or this_regressed
        status = "WARNING" if this_regressed else "OK"

        lines.append(
            f"| {benchmark} | {base_median:.6f} | {head_median:.6f} | {delta_sec:+.6f} ({pct:+.2f}%) | ±{base_stdev:.6f} | ±{head_stdev:.6f} | {status} |"
        )

        per_benchmark.append(
            {
                "benchmark": benchmark,
                "metric": "median_sec",
                "base_mean_sec": base_mean,
                "head_mean_sec": head_mean,
                "base_median_sec": base_median,
                "head_median_sec": head_median,
                "base_stdev_sec": base_stdev,
                "head_stdev_sec": head_stdev,
                "delta_median_sec": delta_sec,
                "delta_median_pct": pct,
                "regressed": this_regressed,
            }
        )

    lines.append("")
    lines.append(f"Thresholds: warn if regression >= {warn_pct:.1f}% and >= {warn_abs_ms:.1f} ms.")
    lines.append(
        f"Result: {'WARNING (one or more benchmark regressions detected)' if regressed else 'OK (no threshold breach)'}"
    )
    lines.append("")

    regressed_rows = [row for row in per_benchmark if row["regressed"]]
    worst_regression = max(regressed_rows, key=lambda row: row["delta_median_sec"]) if regressed_rows else None

    payload = {
        "primary_metric": "median_sec",
        "base": base,
        "head": head,
        "per_benchmark": per_benchmark,
        "regressed_count": len(regressed_rows),
        "worst_regression": worst_regression,
        "warn_pct": warn_pct,
        "warn_abs_ms": warn_abs_ms,
        "regressed": regressed,
    }
    return "\n".join(lines), regressed, payload


def build_invalid_summary(reason: str) -> tuple[str, dict]:
    # non-blocking fallback payload when data is missing/invalid
    lines = []
    lines.append("### PR Benchmark Guard (perfTest)")
    lines.append("")
    lines.append("Result: WARNING (benchmark data invalid/skipped)")
    lines.append("")
    lines.append(f"Reason: {reason}")
    lines.append("")

    payload = {
        "base": None,
        "head": None,
        "per_benchmark": [],
        "regressed_count": 0,
        "worst_regression": None,
        "warn_pct": None,
        "warn_abs_ms": None,
        "regressed": False,
        "data_valid": False,
        "reason": reason,
    }
    return "\n".join(lines), payload


def detect_compiler() -> str | None:
    # best compiler fingerprint for metadata (first available binary wins)
    for binary in ("c++", "g++", "clang++"):
        try:
            result = subprocess.run(
                [binary, "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, OSError):
            continue
        first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
        if first_line:
            return first_line
    return None


def resolve_metadata(args: argparse.Namespace) -> dict:
    # resolve metadata from explicit args first, then GitHub env vars
    timestamp = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    return {
        "commit_sha": args.commit_sha or os.getenv("GITHUB_SHA"),
        "workflow": args.workflow or os.getenv("GITHUB_WORKFLOW"),
        "runner": args.runner or os.getenv("RUNNER_NAME"),
        "os": args.os_name or os.getenv("RUNNER_OS"),
        "compiler": args.compiler or detect_compiler(),
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare perfTest runs for PR benchmark guard.")
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--head-dir", required=True, type=Path)
    parser.add_argument("--warn-pct", type=float, default=20.0)
    parser.add_argument("--warn-abs-ms", type=float, default=10.0)
    parser.add_argument("--markdown-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--commit-sha")
    parser.add_argument("--workflow")
    parser.add_argument("--runner")
    parser.add_argument("--os-name")
    parser.add_argument("--compiler")
    args = parser.parse_args()
    metadata = resolve_metadata(args)

    try:
        base = parse_suite(args.base_dir)
        head = parse_suite(args.head_dir)
        if len(base["runs"]) != len(head["runs"]):
            raise RuntimeError(
                f"Run count mismatch: base has {len(base['runs'])}, head has {len(head['runs'])}."
            )
        markdown, regressed, payload = build_summary(
            base, head, args.warn_pct, args.warn_abs_ms
        )
        payload["data_valid"] = True
    except Exception as exc:
        markdown, payload = build_invalid_summary(str(exc))
        regressed = False
        print(f"::warning::PR benchmark guard data invalid: {exc}")

    payload["metadata"] = metadata
    args.markdown_out.write_text(markdown + "\n", encoding="utf-8")
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if regressed:
        worst = payload.get("worst_regression")
        if worst:
            print(
                "::warning::PR benchmark regression detected: "
                f"{payload['regressed_count']} benchmark(s) exceeded thresholds. "
                f"Worst: {worst['benchmark']} {worst['delta_median_pct']:.2f}% ({worst['delta_median_sec'] * 1000:.2f} ms) slower."
            )
        else:
            print("::warning::PR benchmark regression detected.")
    elif payload.get("data_valid", False):
        print("No benchmark regression above warning thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
