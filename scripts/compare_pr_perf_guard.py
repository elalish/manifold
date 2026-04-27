#!/usr/bin/env python3
import argparse
import json
import re
import statistics
from pathlib import Path

TIME_PATTERN = re.compile(r"time\s*=\s*([0-9]*\.?[0-9]+)\s*sec")


def parse_run(run_path: Path) -> dict:
    times = []
    for line in run_path.read_text(encoding="utf-8").splitlines():
        match = TIME_PATTERN.search(line)
        if match:
            times.append(float(match.group(1)))
    if not times:
        raise RuntimeError(f"No perf timing lines found in {run_path}")
    return {
        "path": str(run_path),
        "samples_sec": times,
        "mean_sec": sum(times) / len(times),
        "max_sec": max(times),
    }


def parse_suite(suite_dir: Path) -> dict:
    run_files = sorted(suite_dir.glob("run*.txt"))
    if not run_files:
        raise RuntimeError(f"No run*.txt files found in {suite_dir}")
    runs = [parse_run(run_file) for run_file in run_files]
    run_means = [run["mean_sec"] for run in runs]
    return {
        "runs": runs,
        "run_means_sec": run_means,
        "median_run_mean_sec": statistics.median(run_means),
        "mean_of_run_means_sec": sum(run_means) / len(run_means),
    }


def build_summary(base: dict, head: dict, warn_pct: float, warn_abs_ms: float) -> tuple[str, bool, dict]:
    base_mean = base["median_run_mean_sec"]
    head_mean = head["median_run_mean_sec"]
    delta_sec = head_mean - base_mean
    delta_ms = delta_sec * 1000.0
    pct = (delta_sec / base_mean * 100.0) if base_mean > 0 else 0.0
    regressed = (pct >= warn_pct) and (delta_ms >= warn_abs_ms)

    lines = []
    lines.append("### PR Benchmark Guard (perfTest)")
    lines.append("")
    lines.append("| Metric | Base | Head | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Median of run means (sec) | {base_mean:.6f} | {head_mean:.6f} | {delta_sec:+.6f} ({pct:+.2f}%) |"
    )
    lines.append(
        f"| Mean of run means (sec) | {base['mean_of_run_means_sec']:.6f} | {head['mean_of_run_means_sec']:.6f} | "
        f"{(head['mean_of_run_means_sec'] - base['mean_of_run_means_sec']):+.6f} |"
    )
    lines.append("")
    lines.append(f"Thresholds: warn if regression >= {warn_pct:.1f}% and >= {warn_abs_ms:.1f} ms.")
    lines.append(f"Result: {'WARNING (regression detected)' if regressed else 'OK (no threshold breach)'}")
    lines.append("")

    payload = {
        "base": base,
        "head": head,
        "delta_sec": delta_sec,
        "delta_ms": delta_ms,
        "delta_pct": pct,
        "warn_pct": warn_pct,
        "warn_abs_ms": warn_abs_ms,
        "regressed": regressed,
    }
    return "\n".join(lines), regressed, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare perfTest runs for PR benchmark guard.")
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--head-dir", required=True, type=Path)
    parser.add_argument("--warn-pct", type=float, default=20.0)
    parser.add_argument("--warn-abs-ms", type=float, default=10.0)
    parser.add_argument("--markdown-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    base = parse_suite(args.base_dir)
    head = parse_suite(args.head_dir)
    markdown, regressed, payload = build_summary(base, head, args.warn_pct, args.warn_abs_ms)

    args.markdown_out.write_text(markdown + "\n", encoding="utf-8")
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if regressed:
        print(
            f"::warning::PR benchmark regression detected: {payload['delta_pct']:.2f}% ({payload['delta_ms']:.2f} ms) slower than base."
        )
    else:
        print("No benchmark regression above warning thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

