#!/usr/bin/env python3
import argparse
import os
import platform
import re
import subprocess
from pathlib import Path

NUM_SIZES = 8
NTRI_PATTERN = re.compile(r"^nTri = ([0-9]+),")

CMAKE_CONFIGURE_ARGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DMANIFOLD_STRICT=ON",
    "-DMANIFOLD_DOWNLOADS=OFF",
    "-DMANIFOLD_PYBIND=OFF",
    "-DMANIFOLD_TEST=ON",
    "-DMANIFOLD_PAR=OFF",
]

BINARY_CANDIDATES = (Path("extras/perfTest"), Path("bin/perfTest"))


def prepare_worktrees(base_sha: str, head_sha: str) -> None:
    subprocess.run(["git", "worktree", "add", "wt-base", base_sha], check=True)
    subprocess.run(["git", "worktree", "add", "wt-head", head_sha], check=True)


def build_perf_test(src_dir: Path, build_dir: Path, osx_architectures: str) -> Path:
    cmake_args = ["cmake", "-S", str(src_dir), "-B", str(build_dir), *CMAKE_CONFIGURE_ARGS]
    if platform.system() == "Darwin":
        cmake_args.append(f"-DCMAKE_OSX_ARCHITECTURES={osx_architectures}")
    subprocess.run(cmake_args, check=True)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "perfTest"], check=True
    )

    for candidate in BINARY_CANDIDATES:
        binary = build_dir / candidate
        if os.access(binary, os.X_OK):
            return binary
    raise RuntimeError("perfTest binary not found in expected paths.")


def parse_peak_rss(output: str) -> tuple[float, int]:
    # macOS `/usr/bin/time -l`= bytes; Linux `-v` = kbytes.
    if platform.system() == "Darwin":
        for line in output.splitlines():
            if "maximum resident set size" in line:
                peak_rss_bytes = int(line.split()[0])
                return peak_rss_bytes / 1024 / 1024, peak_rss_bytes
        return 0.0, 0

    for line in output.splitlines():
        if "Maximum resident set size" in line:
            peak_rss_kb = float(line.split(":", 1)[1].strip())
            return peak_rss_kb / 1024, round(peak_rss_kb * 1024)
    return 0.0, 0


def run_measured_perf_size(binary: Path, size_index: int) -> tuple[str, int]:
    time_flag = "-l" if platform.system() == "Darwin" else "-v"
    result = subprocess.run(
        ["/usr/bin/time", time_flag, str(binary), "--size-index", str(size_index)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = result.stdout
    peak_rss_mb, peak_rss_bytes = parse_peak_rss(output)

    ntri = "unknown"
    for line in output.splitlines():
        match = NTRI_PATTERN.match(line)
        if match:
            ntri = match.group(1)
            break

    block = (
        f"{output.rstrip(chr(10))}\n"
        f"PEAK_RSS nTri={ntri} size_index={size_index} "
        f"peak_rss_mb={peak_rss_mb:.2f} peak_rss_bytes={peak_rss_bytes}"
    )
    return block, result.returncode


def run_measured_suite(
    src_dir: Path, out_dir: Path, repeats: int, osx_architectures: str
) -> None:
    binary = build_perf_test(src_dir, out_dir / "build", osx_architectures)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, repeats + 1):
        run_file = out_dir / f"run{i}.txt"
        with run_file.open("w", encoding="utf-8") as handle:
            for size_index in range(NUM_SIZES):
                block, status = run_measured_perf_size(binary, size_index)
                handle.write(f"### perfTest size_index={size_index}\n{block}\n\n")
                handle.flush()
                if status != 0:
                    raise RuntimeError(
                        f"perfTest exited with status {status} "
                        f"(run {i}, size_index={size_index}, dir={out_dir})"
                    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and benchmark perfTest for PR base/head worktrees."
    )
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--osx-architectures", default="arm64")
    args = parser.parse_args()

    prepare_worktrees(args.base_sha, args.head_sha)

    for variant, worktree in (("base", "wt-base"), ("head", "wt-head")):
        run_measured_suite(
            Path(worktree),
            Path("bench") / variant,
            args.repeats,
            args.osx_architectures,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
