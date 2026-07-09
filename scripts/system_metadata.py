#!/usr/bin/env python3
import argparse
import datetime
import os
import platform
import subprocess
from pathlib import Path


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
