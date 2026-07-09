#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import platform
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
PEAK_RSS_PATTERN = re.compile(
    r"PEAK_RSS\s+nTri=([^\s]+)\s+size_index=([0-9]+)\s+"
    r"peak_rss_mb=([0-9]*\.?[0-9]+)\s+peak_rss_bytes=([0-9]+)"
)


def stdev(values: list[float]) -> float:
    # keep stdev defined even for single-sample cases
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


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
            ntri, size_index, peak_rss_mb, peak_rss_bytes = peak_rss_match.groups()
            benchmark_key = (
                f"nTri={ntri}" if ntri != "unknown" else f"size_index={size_index}"
            )
            peak_rss_by_benchmark[benchmark_key] = {
                "peak_rss_mb": float(peak_rss_mb),
                "peak_rss_bytes": int(peak_rss_bytes),
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

    benchmarks = {}
    for benchmark in benchmark_order:
        samples = benchmark_samples[benchmark]
        benchmark_stats = {
            "samples_sec": samples,
            "mean_sec": statistics.fmean(samples),
            "median_sec": statistics.median(samples),
            "stdev_sec": stdev(samples),
            "min_sec": min(samples),
            "max_sec": max(samples),
            "n_runs": len(samples),
        }
        rss_samples = peak_rss_samples[benchmark]
        if rss_samples:
            benchmark_stats.update(
                {
                    "peak_rss_samples_mb": rss_samples,
                    "peak_rss_mean_mb": statistics.fmean(rss_samples),
                    "peak_rss_median_mb": statistics.median(rss_samples),
                    "peak_rss_stdev_mb": stdev(rss_samples),
                    "peak_rss_min_mb": min(rss_samples),
                    "peak_rss_max_mb": max(rss_samples),
                    "peak_rss_n_runs": len(rss_samples),
                    "peak_rss_complete": len(rss_samples) == len(samples),
                }
            )
        else:
            benchmark_stats["peak_rss_complete"] = False
        benchmarks[benchmark] = benchmark_stats

    return {
        "runs": runs,
        "benchmarks": benchmarks,
        "benchmark_order": benchmark_order,
    }


def build_summary(
    base: dict,
    head: dict,
    warn_pct: float,
    warn_abs_ms: float,
    memory_warn_pct: float,
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
        "Base +/-stdev | Head +/-stdev | Base peak RSS (MB) | "
        "Head peak RSS (MB) | RSS delta | Status |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    per_benchmark = []
    time_regressed = False
    memory_regressed = False
    for benchmark in base["benchmark_order"]:
        base_min = base["benchmarks"][benchmark]["min_sec"]
        head_min = head["benchmarks"][benchmark]["min_sec"]
        base_stdev = base["benchmarks"][benchmark]["stdev_sec"]
        head_stdev = head["benchmarks"][benchmark]["stdev_sec"]
        delta_sec = head_min - base_min
        delta_ms = delta_sec * 1000.0
        pct = delta_sec / base_min * 100.0
        time_this_regressed = (pct >= warn_pct) and (delta_ms >= warn_abs_ms)
        time_regressed = time_regressed or time_this_regressed

        base_rss = base["benchmarks"][benchmark].get("peak_rss_min_mb")
        head_rss = head["benchmarks"][benchmark].get("peak_rss_min_mb")
        rss_delta_mb = None
        rss_delta_pct = None
        memory_this_regressed = False
        if base_rss is not None and head_rss is not None:
            rss_delta_mb = head_rss - base_rss
            rss_delta_pct = rss_delta_mb / base_rss * 100.0 if base_rss else 0.0
            memory_this_regressed = (
                rss_delta_pct >= memory_warn_pct
                and rss_delta_mb >= memory_warn_abs_mb
            )
            memory_regressed = memory_regressed or memory_this_regressed

        # peak_rss_complete is False when a benchmark is missing an RSS sample
        # for one or more repeats on either side; surface that instead of
        # letting the memory guard silently go inactive for this benchmark.
        base_rss_complete = base["benchmarks"][benchmark].get(
            "peak_rss_complete", False
        )
        head_rss_complete = head["benchmarks"][benchmark].get(
            "peak_rss_complete", False
        )
        memory_data_incomplete = not (base_rss_complete and head_rss_complete)

        status_parts = []
        if time_this_regressed:
            status_parts.append("TIME WARNING")
        if memory_this_regressed:
            status_parts.append("MEMORY WARNING")
        if memory_data_incomplete:
            if base_rss is None or head_rss is None:
                status_parts.append("MEMORY DATA MISSING")
            else:
                status_parts.append("MEMORY DATA PARTIAL")
        status = ", ".join(status_parts) if status_parts else "OK"

        base_rss_display = f"{base_rss:.2f}" if base_rss is not None else "N/A"
        head_rss_display = f"{head_rss:.2f}" if head_rss is not None else "N/A"
        rss_delta_display = "N/A"
        if rss_delta_mb is not None and rss_delta_pct is not None:
            rss_delta_display = f"{rss_delta_mb:+.2f} ({rss_delta_pct:+.2f}%)"

        lines.append(
            f"| {benchmark} | {base_min:.6f} | {head_min:.6f} | "
            f"{delta_sec:+.6f} ({pct:+.2f}%) | +/-{base_stdev:.6f} | "
            f"+/-{head_stdev:.6f} | {base_rss_display} | {head_rss_display} | "
            f"{rss_delta_display} | {status} |"
        )

        per_benchmark.append(
            {
                "benchmark": benchmark,
                "metric": "min_sec",
                "memory_metric": "peak_rss_min_mb",
                "base_min_sec": base_min,
                "head_min_sec": head_min,
                "base_stdev_sec": base_stdev,
                "head_stdev_sec": head_stdev,
                "delta_min_sec": delta_sec,
                "delta_min_pct": pct,
                "time_regressed": time_this_regressed,
                "base_peak_rss_mb": base_rss,
                "head_peak_rss_mb": head_rss,
                "delta_peak_rss_mb": rss_delta_mb,
                "delta_peak_rss_pct": rss_delta_pct,
                "memory_regressed": memory_this_regressed,
                "memory_data_incomplete": memory_data_incomplete,
                "regressed": time_this_regressed or memory_this_regressed,
            }
        )

    lines.append("")
    lines.append(
        f"Thresholds: warn if regression >= {warn_pct:.1f}% "
        f"and >= {warn_abs_ms:.1f} ms."
    )
    lines.append(
        f"Memory thresholds: warn if peak RSS regression >= {memory_warn_pct:.1f}% "
        f"and >= {memory_warn_abs_mb:.1f} MB."
    )
    regressed = time_regressed or memory_regressed
    lines.append(
        f"Result: {'WARNING (one or more benchmark regressions detected)' if regressed else 'OK (no threshold breach)'}"
    )

    regressed_rows = [row for row in per_benchmark if row["regressed"]]
    time_regressed_rows = [row for row in per_benchmark if row["time_regressed"]]
    memory_regressed_rows = [row for row in per_benchmark if row["memory_regressed"]]
    memory_incomplete_rows = [
        row for row in per_benchmark if row["memory_data_incomplete"]
    ]
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

    if memory_incomplete_rows:
        lines.append("")
        missing_names = ", ".join(
            row["benchmark"]
            for row in memory_incomplete_rows
            if row["base_peak_rss_mb"] is None or row["head_peak_rss_mb"] is None
        )
        partial_names = ", ".join(
            row["benchmark"]
            for row in memory_incomplete_rows
            if row["base_peak_rss_mb"] is not None
            and row["head_peak_rss_mb"] is not None
        )
        if missing_names:
            lines.append(
                f"Note: peak RSS data missing entirely (all repeats) for base "
                f"or head, memory guard skipped for: {missing_names}."
            )
        if partial_names:
            lines.append(
                "Note: peak RSS data has fewer samples than time samples "
                "(some repeats missing RSS) for: "
                f"{partial_names}. Memory comparison for these ran on the "
                "available samples but may be less reliable."
            )
    lines.append("")

    payload = {
        "primary_metric": "min_sec",
        "memory_metric": "peak_rss_min_mb",
        "base": base,
        "head": head,
        "per_benchmark": per_benchmark,
        "regressed_count": len(regressed_rows),
        "time_regressed_count": len(time_regressed_rows),
        "memory_regressed_count": len(memory_regressed_rows),
        "memory_incomplete_count": len(memory_incomplete_rows),
        "worst_regression": worst_regression,
        "worst_memory_regression": worst_memory_regression,
        "warn_pct": warn_pct,
        "warn_abs_ms": warn_abs_ms,
        "memory_warn_pct": memory_warn_pct,
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
        "memory_incomplete_count": 0,
        "worst_regression": None,
        "worst_memory_regression": None,
        "warn_pct": None,
        "warn_abs_ms": None,
        "memory_warn_pct": None,
        "memory_warn_abs_mb": None,
        "time_regressed": False,
        "memory_regressed": False,
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


def sysctl_value(name: str) -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None

    value = result.stdout.strip()
    return value or None


def int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def default_cpu_model() -> str | None:
    # Prefer /proc/cpuinfo on Linux because it gives a much better CPU name
    # than platform.processor() on GitHub-hosted Ubuntu runners.
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return platform.processor() or None
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def cpu_details() -> dict:
    if platform.system() == "Darwin":
        brand = sysctl_value("machdep.cpu.brand_string")
        model = sysctl_value("hw.model")
        return {
            "model": brand or model or platform.processor() or None,
            "brand": brand,
            "model_identifier": model,
            "arch": platform.machine() or None,
            "logical_count": int_or_none(sysctl_value("hw.logicalcpu"))
            or os.cpu_count(),
            "physical_count": int_or_none(sysctl_value("hw.physicalcpu")),
            "performance_core_count": int_or_none(
                sysctl_value("hw.perflevel0.physicalcpu")
            ),
            "efficiency_core_count": int_or_none(
                sysctl_value("hw.perflevel1.physicalcpu")
            ),
        }

    return {
        "model": default_cpu_model(),
        "brand": None,
        "model_identifier": None,
        "arch": platform.machine() or None,
        "logical_count": os.cpu_count(),
        "physical_count": None,
        "performance_core_count": None,
        "efficiency_core_count": None,
    }


def resolve_metadata(args: argparse.Namespace) -> dict:
    # resolve metadata from explicit args first, then GitHub env vars
    timestamp = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    cpu = cpu_details()
    return {
        "commit_sha": args.commit_sha or os.getenv("GITHUB_SHA"),
        "workflow": args.workflow or os.getenv("GITHUB_WORKFLOW"),
        "runner": args.runner or os.getenv("RUNNER_NAME"),
        "os": args.os_name or os.getenv("RUNNER_OS"),
        "compiler": args.compiler or detect_compiler(),
        "cpu_model": cpu["model"],
        "cpu_brand": cpu["brand"],
        "cpu_model_identifier": cpu["model_identifier"],
        "cpu_arch": cpu["arch"],
        "cpu_logical_count": cpu["logical_count"],
        "cpu_physical_count": cpu["physical_count"],
        "cpu_performance_core_count": cpu["performance_core_count"],
        "cpu_efficiency_core_count": cpu["efficiency_core_count"],
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare perfTest runs for PR benchmark guard.")
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--head-dir", required=True, type=Path)
    parser.add_argument("--warn-pct", type=float, required=True)
    parser.add_argument("--warn-abs-ms", type=float, required=True)
    parser.add_argument("--memory-warn-pct", type=float, required=True)
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
            args.warn_pct,
            args.warn_abs_ms,
            args.memory_warn_pct,
            args.memory_warn_abs_mb,
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
                "::warning::PR benchmark time regression detected: "
                f"{payload['time_regressed_count']} benchmark(s) exceeded thresholds. "
                f"Worst: {worst['benchmark']} {worst['delta_min_pct']:.2f}% ({worst['delta_min_sec'] * 1000:.2f} ms) slower."
            )
        worst_memory = payload.get("worst_memory_regression")
        if worst_memory:
            print(
                "::warning::PR benchmark memory regression detected: "
                f"{payload['memory_regressed_count']} benchmark(s) exceeded thresholds. "
                f"Worst: {worst_memory['benchmark']} {worst_memory['delta_peak_rss_pct']:.2f}% "
                f"({worst_memory['delta_peak_rss_mb']:.2f} MB) higher peak RSS."
            )
        if not worst and not worst_memory:
            print("::warning::PR benchmark regression detected.")
    elif payload.get("data_valid", False):
        print("No benchmark regression above warning thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
