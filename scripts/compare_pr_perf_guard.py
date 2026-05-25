#!/usr/bin/env python3
import argparse
import json
import re
import statistics
from pathlib import Path

TIME_PATTERN = re.compile(r"time\s*=\s*([0-9]*\.?[0-9]+)\s*sec") #only extracts time values(for fallback)
TRI_TIME_PATTERN = re.compile(
    r"nTri\s*=\s*([0-9]+)\s*,\s*time\s*=\s*([0-9]*\.?[0-9]+)\s*sec"#extracts nTri and time values
)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def parse_run(run_path: Path) -> dict:#parses run*.txt files to extract benchmark timings
    benchmarks = []
    for line in run_path.read_text(encoding="utf-8").splitlines():
        tri_match = TRI_TIME_PATTERN.search(line)
        if tri_match:
            benchmark_key = f"nTri={tri_match.group(1)}"
            benchmarks.append({"benchmark": benchmark_key, "time_sec": float(tri_match.group(2))})
            continue
            
        time_match = TIME_PATTERN.search(line)
        if time_match:
            benchmark_key = f"benchmark_{len(benchmarks) + 1}"
            benchmarks.append({"benchmark": benchmark_key, "time_sec": float(time_match.group(1))})

    if not benchmarks:
        raise RuntimeError(f"No perf timing lines found in {run_path}")

    benchmark_names = [entry["benchmark"] for entry in benchmarks]
    if len(set(benchmark_names)) != len(benchmark_names):
        raise RuntimeError(f"Duplicate benchmark keys found in {run_path}")

    return {
        "path": str(run_path),
        "benchmarks": benchmarks,
    }


def parse_suite(suite_dir: Path) -> dict:#calls parse_run for each run*.txt file in the given directory and arranges the data for head and base runs
    run_files = sorted(suite_dir.glob("run*.txt"))
    if not run_files:
        raise RuntimeError(f"No run*.txt files found in {suite_dir}")

    runs = [parse_run(run_file) for run_file in run_files]
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
            "max_sec": max(samples),
        }

    return {
        "runs": runs,
        "benchmarks": benchmarks,
        "benchmark_order": benchmark_order,
    }


def build_summary(base: dict, head: dict, warn_pct: float, warn_abs_ms: float) -> tuple[str, bool, dict]:#arranges everything
    if base["benchmark_order"] != head["benchmark_order"]:
        raise RuntimeError(
            "Benchmark set/order mismatch between base and head: "
            f"{base['benchmark_order']} vs {head['benchmark_order']}"
        )

    lines = []
    lines.append("### PR Benchmark Guard (perfTest)")
    lines.append("")
    lines.append("| Benchmark | Base mean (sec) | Head mean (sec) | Delta | Status |")
    lines.append("|---|---:|---:|---:|")

    per_benchmark = []
    regressed = False
    for benchmark in base["benchmark_order"]:
        base_mean = base["benchmarks"][benchmark]["mean_sec"]
        head_mean = head["benchmarks"][benchmark]["mean_sec"]
        delta_sec = head_mean - base_mean
        delta_ms = delta_sec * 1000.0
        pct = (delta_sec / base_mean * 100.0) if base_mean > 0 else 0.0
        this_regressed = (pct >= warn_pct) and (delta_ms >= warn_abs_ms)
        regressed = regressed or this_regressed
        status = "WARNING" if this_regressed else "OK"

        lines.append(
            f"| {benchmark} | {base_mean:.6f} | {head_mean:.6f} | {delta_sec:+.6f} ({pct:+.2f}%) | {status} |"
        )

        per_benchmark.append(
            {
                "benchmark": benchmark,
                "base_mean_sec": base_mean,
                "head_mean_sec": head_mean,
                "delta_sec": delta_sec,
                "delta_ms": delta_ms,
                "delta_pct": pct,
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
    worst_regression = max(regressed_rows, key=lambda row: row["delta_ms"]) if regressed_rows else None

    payload = {
        "primary_metric": "per_benchmark_mean_sec",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare perfTest runs for PR benchmark guard.")
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--head-dir", required=True, type=Path)
    parser.add_argument("--warn-pct", type=float, default=20.0)
    parser.add_argument("--warn-abs-ms", type=float, default=10.0)
    parser.add_argument("--markdown-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    try:
        base = parse_suite(args.base_dir)
        head = parse_suite(args.head_dir)
        if len(base["runs"]) != len(head["runs"]):
            raise RuntimeError(
                f"Run count mismatch: base has {len(base['runs'])}, head has {len(head['runs'])}."
            )
        markdown, regressed, payload = build_summary(base, head, args.warn_pct, args.warn_abs_ms)
        payload["data_valid"] = True
    except Exception as exc:
        markdown, payload = build_invalid_summary(str(exc))
        regressed = False
        print(f"::warning::PR benchmark guard data invalid: {exc}")

    args.markdown_out.write_text(markdown + "\n", encoding="utf-8")
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if regressed:
        worst = payload.get("worst_regression")
        if worst:
            print(
                "::warning::PR benchmark regression detected: "
                f"{payload['regressed_count']} benchmark(s) exceeded thresholds. "
                f"Worst: {worst['benchmark']} {worst['delta_pct']:.2f}% ({worst['delta_ms']:.2f} ms) slower."
            )
        else:
            print("::warning::PR benchmark regression detected.")
    elif payload.get("data_valid", False):
        print("No benchmark regression above warning thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
