#!/usr/bin/env python3
import argparse
import json
import os
import re
import statistics
from pathlib import Path

from system_metadata import resolve_metadata

# fallback: extract only time value when nTri label is not present
TIME_PATTERN = re.compile(r"time\s*=\s*([0-9]*\.?[0-9]+)\s*sec")
# primary: extract both nTri bucket and timing from perfTest output
TRI_TIME_PATTERN = re.compile(
    r"nTri\s*=\s*([0-9]+)\s*,\s*time\s*=\s*([0-9]*\.?[0-9]+)\s*sec"
)
PEAK_RSS_PATTERN = re.compile(
    r"PEAK_RSS\s+nTri=([^\s]+)\s+size_index=([0-9]+)\s+"
    r"peak_rss_mb=([0-9]*\.?[0-9]+)\s+peak_rss_bytes=([0-9]+)"
)


def sd(values: list[float]) -> float:
    # keep sd defined even for single-sample cases
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def compute_stats(samples: list[float]) -> dict:
    return {
        "samples": samples,
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "sd": sd(samples),
        "min": min(samples),
        "max": max(samples),
        "n_runs": len(samples),
    }


def parse_run(run_path: Path, run_index: int) -> dict:
    # parse one run*.txt into ordered benchmark samples
    benchmarks = []
    peak_rss_by_benchmark = {}
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
            continue

        peak_rss_match = PEAK_RSS_PATTERN.search(line)
        if peak_rss_match:
            ntri, size_index, peak_rss_mb, _peak_rss_bytes = peak_rss_match.groups()
            benchmark_key = (
                f"nTri={ntri}" if ntri != "unknown" else f"size_index={size_index}"
            )
            peak_rss_by_benchmark[benchmark_key] = {
                "peak_rss_mb": float(peak_rss_mb),
            }

    if not benchmarks:
        raise RuntimeError(f"No perf timing lines found in {run_path}")

    benchmark_names = [entry["benchmark"] for entry in benchmarks]
    if len(set(benchmark_names)) != len(benchmark_names):
        raise RuntimeError(f"Duplicate benchmark keys found in {run_path}")

    for entry in benchmarks:
        peak_rss = peak_rss_by_benchmark.get(entry["benchmark"])
        if peak_rss is not None:
            entry.update(peak_rss)

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
    peak_rss_samples = {benchmark: [] for benchmark in benchmark_order}
    for run in runs:
        for entry in run["benchmarks"]:
            benchmark_samples[entry["benchmark"]].append(entry["time_sec"])
            if "peak_rss_mb" in entry:
                peak_rss_samples[entry["benchmark"]].append(entry["peak_rss_mb"])

    for benchmark in benchmark_order:
        if not peak_rss_samples[benchmark]:
            raise RuntimeError(
                f"No peak RSS samples for benchmark {benchmark!r} in {suite_dir}"
            )

    benchmarks = {
        benchmark: {
            "timing_sec": compute_stats(benchmark_samples[benchmark]),
            "peak_rss_mb": compute_stats(peak_rss_samples[benchmark]),
        }
        for benchmark in benchmark_order
    }

    return {
        "runs": runs,
        "benchmarks": benchmarks,
        "benchmark_order": benchmark_order,
    }


def build_summary(
    base: dict,
    head: dict,
    warn_percent: float,
    warn_abs_ms: float,
    memory_warn_percent: float,
    memory_warn_abs_mb: float,
) -> tuple[str, bool, dict]:
    # Compare benchmark minimum time and minimum peak RSS with dual thresholds.
    # Memory uses the same min-min comparison as time
    if base["benchmark_order"] != head["benchmark_order"]:
        raise RuntimeError(
            "Benchmark set/order mismatch between base and head: "
            f"{base['benchmark_order']} vs {head['benchmark_order']}"
        )

    lines = []
    lines.append("### PR Benchmark Guard (perfTest)")
    lines.append("")
    lines.append(
        "| Benchmark | Base min (sec) | Head min (sec) | Time delta | "
        "Base +/-sd | Head +/-sd | Base peak RSS (MB) | "
        "Head peak RSS (MB) | RSS delta | Status |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    per_benchmark = []
    time_regressed = False
    memory_regressed = False
    for benchmark in base["benchmark_order"]:
        base_timing = base["benchmarks"][benchmark]["timing_sec"]
        head_timing = head["benchmarks"][benchmark]["timing_sec"]
        base_min = base_timing["min"]
        head_min = head_timing["min"]
        base_sd = base_timing["sd"]
        head_sd = head_timing["sd"]
        delta_sec = head_min - base_min
        delta_ms = delta_sec * 1000.0
        percent = delta_sec / base_min * 100.0
        time_this_regressed = (percent >= warn_percent) and (delta_ms >= warn_abs_ms)
        time_regressed = time_regressed or time_this_regressed

        base_rss = base["benchmarks"][benchmark]["peak_rss_mb"]["min"]
        head_rss = head["benchmarks"][benchmark]["peak_rss_mb"]["min"]
        rss_delta_mb = head_rss - base_rss
        rss_delta_percent = rss_delta_mb / base_rss * 100.0 if base_rss else 0.0
        memory_this_regressed = (
            rss_delta_percent >= memory_warn_percent
            and rss_delta_mb >= memory_warn_abs_mb
        )
        memory_regressed = memory_regressed or memory_this_regressed

        status_parts = []
        if time_this_regressed:
            status_parts.append("TIME WARNING")
        if memory_this_regressed:
            status_parts.append("MEMORY WARNING")
        status = ", ".join(status_parts) if status_parts else "OK"

        lines.append(
            f"| {benchmark} | {base_min:.6f} | {head_min:.6f} | "
            f"{delta_sec:+.6f} ({percent:+.2f}%) | +/-{base_sd:.6f} | "
            f"+/-{head_sd:.6f} | {base_rss:.2f} | {head_rss:.2f} | "
            f"{rss_delta_mb:+.2f} ({rss_delta_percent:+.2f}%) | {status} |"
        )

        per_benchmark.append(
            {
                "benchmark": benchmark,
                "metric": "timing_sec.min",
                "memory_metric": "peak_rss_mb.min",
                "base_min_sec": base_min,
                "head_min_sec": head_min,
                "base_sd_sec": base_sd,
                "head_sd_sec": head_sd,
                "delta_min_sec": delta_sec,
                "delta_min_percent": percent,
                "time_regressed": time_this_regressed,
                "base_peak_rss_mb": base_rss,
                "head_peak_rss_mb": head_rss,
                "delta_peak_rss_mb": rss_delta_mb,
                "delta_peak_rss_percent": rss_delta_percent,
                "memory_regressed": memory_this_regressed,
                "regressed": time_this_regressed or memory_this_regressed,
            }
        )

    lines.append("")
    lines.append(
        f"Thresholds: warn if regression >= {warn_percent:.1f}% "
        f"and >= {warn_abs_ms:.1f} ms."
    )
    lines.append(
        f"Memory thresholds: warn if peak RSS regression >= {memory_warn_percent:.1f}% "
        f"and >= {memory_warn_abs_mb:.1f} MB."
    )
    regressed = time_regressed or memory_regressed
    lines.append(
        f"Result: {'WARNING (one or more benchmark regressions detected)' if regressed else 'OK (no threshold breach)'}"
    )

    regressed_rows = [row for row in per_benchmark if row["regressed"]]
    time_regressed_rows = [row for row in per_benchmark if row["time_regressed"]]
    memory_regressed_rows = [row for row in per_benchmark if row["memory_regressed"]]
    worst_regression = (
        max(time_regressed_rows, key=lambda row: row["delta_min_sec"])
        if time_regressed_rows
        else None
    )
    worst_memory_regression = (
        max(memory_regressed_rows, key=lambda row: row["delta_peak_rss_mb"])
        if memory_regressed_rows
        else None
    )

    lines.append("")

    payload = {
        "primary_metric": "timing_sec.min",
        "memory_metric": "peak_rss_mb.min",
        "base": base,
        "head": head,
        "per_benchmark": per_benchmark,
        "regressed_count": len(regressed_rows),
        "time_regressed_count": len(time_regressed_rows),
        "memory_regressed_count": len(memory_regressed_rows),
        "worst_regression": worst_regression,
        "worst_memory_regression": worst_memory_regression,
        "warn_percent": warn_percent,
        "warn_abs_ms": warn_abs_ms,
        "memory_warn_percent": memory_warn_percent,
        "memory_warn_abs_mb": memory_warn_abs_mb,
        "time_regressed": time_regressed,
        "memory_regressed": memory_regressed,
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
        "time_regressed_count": 0,
        "memory_regressed_count": 0,
        "worst_regression": None,
        "worst_memory_regression": None,
        "warn_percent": None,
        "warn_abs_ms": None,
        "memory_warn_percent": None,
        "memory_warn_abs_mb": None,
        "time_regressed": False,
        "memory_regressed": False,
        "regressed": False,
        "data_valid": False,
        "reason": reason,
    }
    return "\n".join(lines), payload


def print_github_group(title: str, content: str) -> None:
    print(f"::group::{title}")
    print(content)
    print("::endgroup::")


def emit_ci_reporting(
    markdown: str, json_text: str, base_dir: Path, head_dir: Path
) -> None:
    print_github_group("PR benchmark summary", markdown)
    print_github_group("PR benchmark result.json", json_text)

    raw_output_lines = []
    for suite_dir in (base_dir, head_dir):
        for run_file in sorted(suite_dir.glob("run*.txt")):
            raw_output_lines.append(f"--- {run_file} ---")
            raw_output_lines.append(run_file.read_text(encoding="utf-8"))
    print_github_group("PR benchmark raw outputs", "\n".join(raw_output_lines))

    step_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not step_summary_path:
        return
    with open(step_summary_path, "a", encoding="utf-8") as handle:
        handle.write(markdown + "\n\n")
        handle.write(
            "Raw logs: open this step and expand `PR benchmark result.json` "
            "and `PR benchmark raw outputs` groups.\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare perfTest runs for PR benchmark guard."
    )
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--head-dir", required=True, type=Path)
    parser.add_argument("--warn-percent", type=float, required=True)
    parser.add_argument("--warn-abs-ms", type=float, required=True)
    parser.add_argument("--memory-warn-percent", type=float, required=True)
    parser.add_argument("--memory-warn-abs-mb", type=float, required=True)
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
            base,
            head,
            args.warn_percent,
            args.warn_abs_ms,
            args.memory_warn_percent,
            args.memory_warn_abs_mb,
        )
        payload["data_valid"] = True
    except RuntimeError as exc:
        markdown, payload = build_invalid_summary(str(exc))
        regressed = False
        print(f"::warning::PR benchmark guard data invalid: {exc}")

    payload["metadata"] = metadata
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(markdown + "\n", encoding="utf-8")
    json_text = json.dumps(payload, indent=2) + "\n"
    args.json_out.write_text(json_text, encoding="utf-8")

    emit_ci_reporting(markdown, json_text, args.base_dir, args.head_dir)

    if regressed:
        worst = payload.get("worst_regression")
        if worst:
            print(
                "::warning::PR benchmark time regression detected: "
                f"{payload['time_regressed_count']} benchmark(s) exceeded thresholds. "
                f"Worst: {worst['benchmark']} {worst['delta_min_percent']:.2f}% ({worst['delta_min_sec'] * 1000:.2f} ms) slower."
            )
        worst_memory = payload.get("worst_memory_regression")
        if worst_memory:
            print(
                "::warning::PR benchmark memory regression detected: "
                f"{payload['memory_regressed_count']} benchmark(s) exceeded thresholds. "
                f"Worst: {worst_memory['benchmark']} {worst_memory['delta_peak_rss_percent']:.2f}% "
                f"({worst_memory['delta_peak_rss_mb']:.2f} MB) higher peak RSS."
            )
        if not worst and not worst_memory:
            print("::warning::PR benchmark regression detected.")
    elif payload.get("data_valid", False):
        print("No benchmark regression above warning thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
