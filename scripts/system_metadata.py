#!/usr/bin/env python3
import argparse
import datetime
import os
import platform
import re
import subprocess
from pathlib import Path

CMAKE_SUMMARY_PATTERN = re.compile(r"^--\s+([A-Z0-9_]+):\s*(.*)$")


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


def detect_compiler() -> str | None:
    # best compiler fingerprint for metadata
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
    # /proc/cpuinfo gives a better CPU name than platform.processor()
    # on GitHub-hosted Ubuntu runners.
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
    # resolve metadata from args first, then GitHub env vars
    timestamp = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    suite_dir = getattr(args, "suite_dir", None)
    cmake = parse_cmake_configure_log(suite_dir / "cmake_configure.log") if suite_dir else {}
    cpu = cpu_details()
    cpu_count = getattr(args, "cpu_count", None) or cpu["logical_count"]
    return {
        "commit_sha": getattr(args, "commit_sha", None) or os.getenv("GITHUB_SHA"),
        "workflow": getattr(args, "workflow", None) or os.getenv("GITHUB_WORKFLOW"),
        "runner": getattr(args, "runner", None) or os.getenv("RUNNER_NAME"),
        "os": getattr(args, "os_name", None) or os.getenv("RUNNER_OS"),
        "compiler": getattr(args, "compiler", None)
        or cmake_compiler(cmake)
        or detect_compiler(),
        "cmake": cmake,
        "cpu_model": getattr(args, "cpu_model", None) or cpu["model"],
        "cpu_count": cpu_count,
        "cpu_brand": cpu["brand"],
        "cpu_model_identifier": cpu["model_identifier"],
        "cpu_arch": cpu["arch"],
        "cpu_logical_count": cpu["logical_count"],
        "cpu_physical_count": cpu["physical_count"],
        "cpu_performance_core_count": cpu["performance_core_count"],
        "cpu_efficiency_core_count": cpu["efficiency_core_count"],
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
    }
