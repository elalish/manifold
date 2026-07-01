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

SCHEMA_VERSION = "1.0.0"
EMBER_SUITE = "weekly_ember_phase"
PERF_SUITE = "perf_size_sweep"
GTEST_SUITE = "existing_gtests"

EMBER_CASE_PATTERN = re.compile(r"^### case\s+([0-9]+)(?:\s+\([0-9]+\s+vs\s+[0-9]+\))?")
PHASE_PATTERN = re.compile(r"^-+\s+([0-9]+)\s+ms for\s+(.*)$")
VERTS_PATTERN = re.compile(r"^[0-9]+\s+verts and\s+[0-9]+\s+tris$")
PERF_PATTERN = re.compile(r"^nTri\s*=\s*([0-9]+),\s*time\s*=\s*([0-9.eE+-]+)\s+sec$")
GTEST_OK_PATTERN = re.compile(r"^\[\s+OK\s+\]\s+([^\s]+)\s+\(([0-9]+) ms\)")
CMAKE_SUMMARY_PATTERN = re.compile(r"^--\s+([A-Z0-9_]+):\s*(.*)$")

INDEPENDENT_PHASES = [
    "Assembly",
    "Triangulation",
    "Simplification",
    "Sorting",
    "Intersect12 P->Q",
    "Intersect12 Q->P",
    "Winding03 P",
    "Winding03 Q",
]


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def summarize(values: list[float], suffix: str = "") -> dict:
    fields = {
        "samples": values,
        "mean": mean(values),
        "median": statistics.median(values),
        "stdev": stdev(values),
        "min": min(values),
        "max": max(values),
    }
    return {
        **{f"{name}{suffix}": value for name, value in fields.items()},
        "n_runs": len(values),
    }


def summarize_ms(values: list[float]) -> dict:
    return summarize(values, "_ms")

def summarize_ratio(values: list[float]) -> dict:
    return summarize(values)

def run_files(suite_dir: Path) -> list[Path]:
    files = sorted(suite_dir.glob("run*.txt"))
    if not files:
        raise RuntimeError(f"No run*.txt files found in {suite_dir}")
    return files


def load_ember_specs(source_dir: Path) -> list[dict]:
    spec_path = (
        source_dir
        / "extras"
        / "ember_tests"
        / "testfiles"
        / "ember-benchmark-cases.json"
    )
    return json.loads(spec_path.read_text(encoding="utf-8"))


def parse_ember_run(run_path: Path, run_index: int) -> dict:
    cases = []
    current = None
    accepting_phases = False

    def finish_current() -> None:
        if current is not None:
            cases.append(current)

    for raw_line in run_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        case_match = EMBER_CASE_PATTERN.match(line)
        if case_match:
            if current is not None and not accepting_phases:
                same_case = current["case_index"] == int(case_match.group(1))
                if same_case:
                    continue
            finish_current()
            current = {
                "case_index": int(case_match.group(1)),
                "phases_ms": {},
            }
            accepting_phases = True
            continue

        if current is None or not accepting_phases:
            continue

        phase_match = PHASE_PATTERN.match(line)
        if phase_match:
            phase = phase_match.group(2).strip()
            if phase in INDEPENDENT_PHASES and phase not in current["phases_ms"]:
                current["phases_ms"][phase] = float(phase_match.group(1))
            continue

        if VERTS_PATTERN.match(line.strip()):
            accepting_phases = False

    finish_current()
    if not cases:
        raise RuntimeError(f"No Ember benchmark cases found in {run_path}")

    for case in cases:
        for phase in INDEPENDENT_PHASES:
            case["phases_ms"].setdefault(phase, 0.0)

    return {"path": str(run_path), "run_index": run_index, "cases": cases}


def parse_ember_suite(suite_dir: Path, source_dir: Path) -> dict:
    runs = [parse_ember_run(path, i + 1) for i, path in enumerate(run_files(suite_dir))]
    ember_specs = load_ember_specs(source_dir)
    case_order = [case["case_index"] for case in runs[0]["cases"]]
    for run in runs[1:]:
        run_order = [case["case_index"] for case in run["cases"]]
        if run_order != case_order:
            raise RuntimeError(
                f"Case layout mismatch in {run['path']}: expected {case_order}, got {run_order}"
            )

    cases = []
    for position, case_index in enumerate(case_order):
        if case_index < 0 or case_index >= len(ember_specs):
            raise IndexError(f"Case index {case_index} is outside 0..{len(ember_specs) - 1}")
        spec = ember_specs[case_index]
        phase_samples = {phase: [] for phase in INDEPENDENT_PHASES}
        full_phase_samples = []
        intersect12_share_samples = []

        for run in runs:
            phases = run["cases"][position]["phases_ms"]
            full_phase_sum = sum(phases[phase] for phase in INDEPENDENT_PHASES)
            full_phase_samples.append(full_phase_sum)
            intersect12_sum = phases["Intersect12 P->Q"] + phases["Intersect12 Q->P"]
            intersect12_share_samples.append(
                intersect12_sum / full_phase_sum if full_phase_sum else 0.0
            )
            for phase in INDEPENDENT_PHASES:
                phase_samples[phase].append(phases[phase])

        phase_metrics = {
            phase: summarize_ms(samples) for phase, samples in phase_samples.items()
        }
        dominant_phase = max(
            INDEPENDENT_PHASES, key=lambda phase: phase_metrics[phase]["mean_ms"]
        )
        cases.append(
            {
                "case_index": case_index,
                "id_a": spec["id_a"],
                "id_b": spec["id_b"],
                "full_phase_sum_ms": summarize_ms(full_phase_samples),
                "intersect12_share": summarize_ratio(intersect12_share_samples),
                "dominant_phase": dominant_phase,
                "phases": phase_metrics,
            }
        )

    return {
        "type": "ember_phase",
        "description": "Selected Ember boolean phase timing cases",
        "phase_order": INDEPENDENT_PHASES,
        "case_order": case_order,
        "cases": cases,
        "runs": runs,
    }


def parse_perf_run(run_path: Path, run_index: int) -> dict:
    workloads = []
    for line in run_path.read_text(encoding="utf-8").splitlines():
        match = PERF_PATTERN.match(line.strip())
        if not match:
            continue
        workloads.append(
            {
                "n_tri": int(match.group(1)),
                "time_ms": float(match.group(2)) * 1000.0,
            }
        )
    if not workloads:
        raise RuntimeError(f"No perfTest rows found in {run_path}")
    return {"path": str(run_path), "run_index": run_index, "workloads": workloads}


def parse_perf_suite(suite_dir: Path) -> dict:
    runs = [parse_perf_run(path, i + 1) for i, path in enumerate(run_files(suite_dir))]
    workload_order = [item["n_tri"] for item in runs[0]["workloads"]]
    for run in runs[1:]:
        run_order = [item["n_tri"] for item in run["workloads"]]
        if run_order != workload_order:
            raise RuntimeError(
                f"perfTest layout mismatch in {run['path']}: expected {workload_order}, got {run_order}"
            )

    workloads = []
    for position, n_tri in enumerate(workload_order):
        samples = [run["workloads"][position]["time_ms"] for run in runs]
        workloads.append({"n_tri": n_tri, "time_ms": summarize_ms(samples)})

    return {
        "type": "perf_size_sweep",
        "description": "perfTest sphere boolean size sweep",
        "workload_order": workload_order,
        "workloads": workloads,
        "runs": runs,
    }


def parse_gtest_run(run_path: Path, run_index: int) -> dict:
    tests = []
    for line in run_path.read_text(encoding="utf-8").splitlines():
        match = GTEST_OK_PATTERN.match(line.strip())
        if not match:
            continue
        tests.append({"name": match.group(1), "time_ms": float(match.group(2))})
    if not tests:
        raise RuntimeError(f"No gtest timing rows found in {run_path}")
    return {"path": str(run_path), "run_index": run_index, "tests": tests}


def parse_gtest_suite(suite_dir: Path) -> dict:
    runs = [parse_gtest_run(path, i + 1) for i, path in enumerate(run_files(suite_dir))]
    test_order = [item["name"] for item in runs[0]["tests"]]
    for run in runs[1:]:
        run_order = [item["name"] for item in run["tests"]]
        if run_order != test_order:
            raise RuntimeError(
                f"gtest layout mismatch in {run['path']}: expected {test_order}, got {run_order}"
            )

    tests = []
    for position, name in enumerate(test_order):
        samples = [run["tests"][position]["time_ms"] for run in runs]
        tests.append({"name": name, "time_ms": summarize_ms(samples)})

    return {
        "type": "existing_gtests",
        "description": "Selected existing regression tests used as weekly benchmark signals",
        "test_order": test_order,
        "tests": tests,
        "runs": runs,
    }


def parse_cmake_configure_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {}

    values = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = CMAKE_SUMMARY_PATTERN.match(line.strip())
        if match:
            values[match.group(1)] = match.group(2).strip()

    return {
        "version": values.get("CMAKE_VERSION"),
        "generator": values.get("CMAKE_GENERATOR"),
        "build_type": values.get("CMAKE_BUILD_TYPE"),
        "cxx_compiler_id": values.get("CMAKE_CXX_COMPILER_ID"),
        "cxx_compiler_version": values.get("CMAKE_CXX_COMPILER_VERSION"),
    }


def cmake_compiler(cmake: dict) -> str | None:
    compiler_id = cmake.get("cxx_compiler_id")
    compiler_version = cmake.get("cxx_compiler_version")
    if compiler_id and compiler_version:
        return f"{compiler_id} {compiler_version}"
    return compiler_id or compiler_version


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
    #/proc/cpuinfo on Linux because it gives a much better CPU name
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
    timestamp = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    cmake = parse_cmake_configure_log(args.suite_dir / "cmake_configure.log")
    cpu = cpu_details()
    return {
        "schema_version": SCHEMA_VERSION,
        "commit_sha": args.commit_sha or os.getenv("GITHUB_SHA"),
        "workflow": args.workflow or os.getenv("GITHUB_WORKFLOW"),
        "runner": args.runner or os.getenv("RUNNER_NAME"),
        "os": args.os_name or os.getenv("RUNNER_OS"),
        "compiler": args.compiler or cmake_compiler(cmake),
        "cmake": cmake,
        "cpu_model": args.cpu_model or cpu["model"],
        "cpu_count": args.cpu_count or cpu["logical_count"],
        "cpu_brand": cpu["brand"],
        "cpu_model_identifier": cpu["model_identifier"],
        "cpu_arch": cpu["arch"],
        "cpu_logical_count": cpu["logical_count"],
        "cpu_physical_count": cpu["physical_count"],
        "cpu_performance_core_count": cpu["performance_core_count"],
        "cpu_efficiency_core_count": cpu["efficiency_core_count"],
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
    }


def build_ember_summary(suite: dict) -> list[str]:
    lines = []
    lines.append("#### Ember Phase Timings")
    lines.append("")
    lines.append(
        "| Case | Dominant phase | Full mean (ms) | Intersect12 share | Assembly | Triangulation | Simplification | Sorting | P->Q | Q->P | Winding P | Winding Q | Runs |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for case in sorted(
        suite["cases"],
        key=lambda entry: entry["full_phase_sum_ms"]["mean_ms"],
        reverse=True,
    ):
        phases = case["phases"]
        lines.append(
            "| {case_index} | {dominant_phase} | {full:.2f} | {share:.3f} | "
            "{assembly:.2f} | {triangulation:.2f} | {simplification:.2f} | "
            "{sorting:.2f} | {p_to_q:.2f} | {q_to_p:.2f} | {winding_p:.2f} | "
            "{winding_q:.2f} | {runs} |".format(
                case_index=case["case_index"],
                dominant_phase=case["dominant_phase"],
                full=case["full_phase_sum_ms"]["mean_ms"],
                share=case["intersect12_share"]["mean"],
                assembly=phases["Assembly"]["mean_ms"],
                triangulation=phases["Triangulation"]["mean_ms"],
                simplification=phases["Simplification"]["mean_ms"],
                sorting=phases["Sorting"]["mean_ms"],
                p_to_q=phases["Intersect12 P->Q"]["mean_ms"],
                q_to_p=phases["Intersect12 Q->P"]["mean_ms"],
                winding_p=phases["Winding03 P"]["mean_ms"],
                winding_q=phases["Winding03 Q"]["mean_ms"],
                runs=case["full_phase_sum_ms"]["n_runs"],
            )
        )

    lines.append("")
    lines.append(
        "Note: phase timings use independent phases only; `Intersections (total)` is not added to the denominator."
    )
    lines.append("")
    return lines


def build_perf_summary(suite: dict) -> list[str]:
    lines = []
    lines.append("#### perfTest Size Sweep")
    lines.append("")
    lines.append("| nTri | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for workload in suite["workloads"]:
        timing = workload["time_ms"]
        lines.append(
            "| {n_tri} | {mean:.2f} | {median:.2f} | {min_:.2f} | {max_:.2f} | {runs} |".format(
                n_tri=workload["n_tri"],
                mean=timing["mean_ms"],
                median=timing["median_ms"],
                min_=timing["min_ms"],
                max_=timing["max_ms"],
                runs=timing["n_runs"],
            )
        )
    lines.append("")
    return lines


def build_gtest_summary(suite: dict) -> list[str]:
    lines = []
    lines.append("#### Existing Regression Tests")
    lines.append("")
    lines.append("| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Runs |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for test in suite["tests"]:
        timing = test["time_ms"]
        lines.append(
            "| {name} | {mean:.2f} | {median:.2f} | {min_:.2f} | {max_:.2f} | {runs} |".format(
                name=test["name"],
                mean=timing["mean_ms"],
                median=timing["median_ms"],
                min_=timing["min_ms"],
                max_=timing["max_ms"],
                runs=timing["n_runs"],
            )
        )
    lines.append("")
    return lines


def build_summary(suites: dict, metadata: dict, repeats: int) -> str:
    lines = []
    lines.append("### Weekly Benchmarks")
    lines.append("")
    lines.append(f"Commit: `{metadata.get('commit_sha') or 'unknown'}`")
    lines.append(f"Runner: `{metadata.get('runner') or 'unknown'}`")
    lines.append(f"OS: `{metadata.get('os') or 'unknown'}`")
    lines.append(f"Compiler: `{metadata.get('compiler') or 'unknown'}`")
    lines.append(f"CPU: `{metadata.get('cpu_model') or 'unknown'}`")
    lines.append(f"CPU count: `{metadata.get('cpu_count') or 'unknown'}`")
    if metadata.get("cpu_model_identifier"):
        lines.append(f"CPU model identifier: `{metadata['cpu_model_identifier']}`")
    if metadata.get("cpu_physical_count"):
        lines.append(f"CPU physical cores: `{metadata['cpu_physical_count']}`")
    if metadata.get("cpu_performance_core_count") is not None:
        lines.append(
            f"CPU performance cores: `{metadata['cpu_performance_core_count']}`"
        )
    if metadata.get("cpu_efficiency_core_count") is not None:
        lines.append(
            f"CPU efficiency cores: `{metadata['cpu_efficiency_core_count']}`"
        )
    lines.append(f"Repeats: `{repeats}`")
    lines.append("")

    if EMBER_SUITE in suites:
        lines.extend(build_ember_summary(suites[EMBER_SUITE]))
    if PERF_SUITE in suites:
        lines.extend(build_perf_summary(suites[PERF_SUITE]))
    if GTEST_SUITE in suites:
        lines.extend(build_gtest_summary(suites[GTEST_SUITE]))
    return "\n".join(lines)


def parse_suites(root_dir: Path, source_dir: Path) -> dict:
    suites = {}
    ember_dir = root_dir / "ember_phase"
    perf_dir = root_dir / "perf_size_sweep"
    gtest_dir = root_dir / "existing_gtests"

    if ember_dir.exists():
        suites[EMBER_SUITE] = parse_ember_suite(ember_dir, source_dir)
    elif list(root_dir.glob("run*.txt")):
        suites[EMBER_SUITE] = parse_ember_suite(root_dir, source_dir)

    if perf_dir.exists():
        suites[PERF_SUITE] = parse_perf_suite(perf_dir)

    if gtest_dir.exists():
        suites[GTEST_SUITE] = parse_gtest_suite(gtest_dir)

    if not suites:
        raise RuntimeError(f"No weekly benchmark suites found in {root_dir}")
    return suites


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse weekly benchmark runs.")
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--source-dir", default=Path("."), type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--commit-sha")
    parser.add_argument("--workflow")
    parser.add_argument("--runner")
    parser.add_argument("--os-name")
    parser.add_argument("--compiler")
    parser.add_argument("--cpu-model")
    parser.add_argument("--cpu-count", type=int)
    args = parser.parse_args()

    metadata = resolve_metadata(args)
    suites = parse_suites(args.suite_dir, args.source_dir)
    suite_names = list(suites.keys())
    markdown = build_summary(suites, metadata, args.repeats)
    payload = {
        "metadata": metadata,
        "config": {
            "repeats": args.repeats,
            "suites": suite_names,
        },
        "suites": suites,
    }

    args.markdown_out.write_text(markdown + "\n", encoding="utf-8")
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
