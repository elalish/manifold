#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

THINGI10K_URL = (
    "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main/raw_meshes"
)
DEFAULT_EMBER_CASES = "16 84 667 695 260 406 551 582"
DEFAULT_GTEST_FILTER = ":".join(
    [
        "Manifold.DeepChainDoesNotOverflowNumLeaves",
        "Boolean.BatchBoolean",
        "CrossSection.BatchBoolean",
        "Polygon.Sponge4",
        "Polygon.Zebra1",
        "Polygon.Zebra3",
        "ExecutionContextFromMeshGL.CancelConcurrent",
    ]
)
#perf test is added seperately 

@dataclass(frozen=True)
class BuildContext:
    source_dir: Path
    out_dir: Path
    repeats: int
    build_dir: Path


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    target: str
    binary: str
    run: object


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


def run_command(
    args: list[str], cwd: Path | None = None, log_path: Path | None = None
) -> subprocess.CompletedProcess:
    print("+ " + " ".join(args))
    sys.stdout.flush()
    if log_path is None:
        return subprocess.run(args, check=True, cwd=cwd)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        args,
        check=False,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = result.stdout or ""
    if output:
        print(output, end="")
    log_path.write_text("+ " + " ".join(args) + "\n" + output, encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, args, output=result.stdout
        )
    return result


def configure_build(ctx: BuildContext) -> None:
    run_command(
        [
            "cmake",
            "-S",
            str(ctx.source_dir),
            "-B",
            str(ctx.build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DMANIFOLD_STRICT=ON",
            "-DMANIFOLD_PYBIND=OFF",
            "-DMANIFOLD_TEST=ON",
            "-DMANIFOLD_PAR=ON",
            "-DMANIFOLD_TIMING=ON",
            "-DASSIMP_ENABLE=ON",
        ],
        log_path=ctx.out_dir / "cmake_configure.log",
    )


def build_targets(ctx: BuildContext, suites: list[BenchmarkSuite]) -> None:
    targets = list(dict.fromkeys(suite.target for suite in suites))
    run_command(["cmake", "--build", str(ctx.build_dir), "--target", *targets])


def find_binary(ctx: BuildContext, name: str) -> Path:
    candidates = [
        ctx.build_dir / "extras" / name,
        ctx.build_dir / "test" / name,
        ctx.build_dir / "bin" / name,
        ctx.build_dir / name,
        ctx.build_dir / "extras" / f"{name}.exe",
        ctx.build_dir / "test" / f"{name}.exe",
        ctx.build_dir / "bin" / f"{name}.exe",
        ctx.build_dir / f"{name}.exe",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{name} binary not found in expected paths: {joined}")


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


def load_ember_specs(spec_path: Path, cases: list[int]) -> list[dict]:
    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    selected = []
    for case_index in cases:
        if case_index < 0 or case_index >= len(specs):
            raise IndexError(f"Case index {case_index} is outside 0..{len(specs) - 1}")
        spec = dict(specs[case_index])
        spec["case_index"] = case_index
        selected.append(spec)
    return selected


def write_ember_cases_metadata(out_dir: Path, specs: list[dict]) -> None:
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


def run_ember_case(
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


def run_ember_suite(ctx: BuildContext, binary: Path) -> None:
    out_dir = ctx.out_dir / "ember_phase"
    mesh_dir = ctx.out_dir / "raw_meshes"
    spec_file = (
        ctx.source_dir
        / "extras"
        / "ember_tests"
        / "testfiles"
        / "ember-benchmark-cases.json"
    )
    cases = parse_cases(os.getenv("WEEKLY_BENCHMARK_CASES", DEFAULT_EMBER_CASES))
    threads = int(os.getenv("WEEKLY_BENCHMARK_THREADS", "1"))
    verbose = int(os.getenv("WEEKLY_BENCHMARK_VERBOSE", "2"))

    specs = load_ember_specs(spec_file, cases)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ember_cases_metadata(out_dir, specs)

    for run_index in range(1, ctx.repeats + 1):
        run_file = out_dir / f"run{run_index}.txt"
        benchmark_csv = out_dir / "benchmark.csv"
        benchmark_csv.unlink(missing_ok=True)
        with run_file.open("w", encoding="utf-8") as out:
            for spec in specs:
                out.write(f"### case {spec['case_index']}\n")
                stdout = run_ember_case(
                    binary.resolve(),
                    spec,
                    mesh_dir.resolve(),
                    threads,
                    verbose,
                    out_dir.resolve(),
                )
                out.write(stdout)
                if not stdout.endswith("\n"):
                    out.write("\n")

        if benchmark_csv.exists():
            benchmark_csv.replace(out_dir / f"benchmark_run{run_index}.csv")

        print(f"completed Ember benchmark run {run_index}/{ctx.repeats}")
        sys.stdout.flush()


def run_perf_size_sweep_suite(ctx: BuildContext, binary: Path) -> None:
    out_dir = ctx.out_dir / "perf_size_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run_index in range(1, ctx.repeats + 1):
        result = subprocess.run(
            [str(binary)],
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        (out_dir / f"run{run_index}.txt").write_text(output, encoding="utf-8")
        print(f"completed perfTest run {run_index}/{ctx.repeats}")
        sys.stdout.flush()


def run_existing_gtests_suite(ctx: BuildContext, binary: Path) -> None:
    out_dir = ctx.out_dir / "existing_gtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    gtest_filter = os.getenv("WEEKLY_BENCHMARK_GTEST_FILTER", DEFAULT_GTEST_FILTER)
    for run_index in range(1, ctx.repeats + 1):
        result = subprocess.run(
            [str(binary), f"--gtest_filter={gtest_filter}"],
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if "[==========] Running 0 tests" in output:
            raise RuntimeError(f"gtest filter selected no tests: {gtest_filter}")
        (out_dir / f"run{run_index}.txt").write_text(output, encoding="utf-8")
        print(f"completed existing gtest run {run_index}/{ctx.repeats}")
        sys.stdout.flush()


# Add future benchmark suites here.
# emits run<N>.txt files that parse_weekly_benchmarks.py can collect.
SUITES = [
    BenchmarkSuite("ember_phase", "man_bench", "man_bench", run_ember_suite),
    BenchmarkSuite(
        "perf_size_sweep", "perfTest", "perfTest", run_perf_size_sweep_suite
    ),
    BenchmarkSuite(
        "existing_gtests", "manifold_test", "manifold_test", run_existing_gtests_suite
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run weekly benchmark suites.")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("repeats", type=int)
    args = parser.parse_args()

    ctx = BuildContext(
        source_dir=args.source_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        repeats=args.repeats,
        build_dir=(args.out_dir / "build").resolve(),
    )

    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    configure_build(ctx)
    build_targets(ctx, SUITES)
    for suite in SUITES:
        print(f"::group::Run weekly benchmark suite: {suite.name}")
        binary = find_binary(ctx, suite.binary)
        suite.run(ctx, binary)
        print("::endgroup::")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
