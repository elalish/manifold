#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)


def parse_timestamp(raw: str | None) -> datetime.datetime:
    if not raw:
        return utc_now()
    try:
        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
            datetime.timezone.utc
        )
    except ValueError:
        return utc_now()


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def github_run_url(run_id: str) -> str | None:
    server = os.getenv("GITHUB_SERVER_URL")
    repo = os.getenv("GITHUB_REPOSITORY")
    if not server or not repo or not os.getenv("GITHUB_RUN_ID"):
        return None
    return f"{server}/{repo}/actions/runs/{run_id}"


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        return {"schema_version": 1, "runs": []}
    return json.loads(index_path.read_text(encoding="utf-8-sig"))


def optional_field(payload: dict, key: str, value: str | int | None) -> None:
    if value not in (None, ""):
        payload[key] = value


def sanitizer_summary(args: argparse.Namespace) -> dict | None:
    summary = {}
    optional_field(summary, "subset", args.sanitizer_subset)
    optional_field(summary, "build_result", args.sanitizer_build_result)
    optional_field(summary, "test_result", args.sanitizer_test_result)
    optional_field(summary, "runner", args.sanitizer_runner)
    optional_field(summary, "os", args.sanitizer_os)
    optional_field(summary, "compiler", args.sanitizer_compiler)
    return summary or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish weekly benchmark result files.")
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--release-tag")
    parser.add_argument("--sanitizer-subset")
    parser.add_argument("--sanitizer-build-result")
    parser.add_argument("--sanitizer-test-result")
    parser.add_argument("--sanitizer-runner")
    parser.add_argument("--sanitizer-os")
    parser.add_argument("--sanitizer-compiler")
    args = parser.parse_args()

    result_path = args.suite_dir / "result.json"
    summary_path = args.suite_dir / "summary.md"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing weekly benchmark result: {result_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing weekly benchmark summary: {summary_path}")

    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    metadata = result.get("metadata", {})
    timestamp = parse_timestamp(metadata.get("timestamp"))
    timestamp_iso = timestamp.isoformat().replace("+00:00", "Z")
    run_id = os.getenv("GITHUB_RUN_ID") or timestamp.strftime("%Y%m%dT%H%M%SZ")

    dated_rel = (
        Path("weekly")
        / timestamp.strftime("%Y")
        / timestamp.strftime("%m")
        / timestamp.strftime("%d")
        / run_id
    )
    dated_dir = args.data_dir / dated_rel
    dated_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "run_id": run_id,
        "timestamp": timestamp_iso,
        "date": timestamp.strftime("%Y-%m-%d"),
        "commit_sha": metadata.get("commit_sha"),
        "workflow": metadata.get("workflow"),
        "runner": metadata.get("runner"),
        "os": metadata.get("os"),
        "compiler": metadata.get("compiler"),
        "cpu_model": metadata.get("cpu_model"),
        "cpu_count": metadata.get("cpu_count"),
        "result_path": str(dated_rel / "result.json").replace("\\", "/"),
        "summary_path": str(dated_rel / "summary.md").replace("\\", "/"),
        "github_run_url": github_run_url(run_id),
    }
    if args.release_tag:
        entry["release_tag"] = args.release_tag
    sanitizer = sanitizer_summary(args)
    if sanitizer:
        entry["sanitizer"] = sanitizer
    entry["trigger"] = "release" if args.release_tag else "weekly"

    result_dest = dated_dir / "result.json"
    summary_dest = dated_dir / "summary.md"
    stored_result = dict(result)
    stored_result["storage"] = entry
    json_dump(result_dest, stored_result)
    shutil.copy2(summary_path, summary_dest)

    json_dump(args.data_dir / "weekly" / "latest.json", stored_result)

    index_path = args.data_dir / "weekly" / "index.json"
    index = load_index(index_path)
    runs = [run for run in index.get("runs", []) if str(run.get("run_id")) != run_id]
    runs.append(entry)
    runs.sort(key=lambda run: run.get("timestamp", ""))
    index = {
        "schema_version": 1,
        "generated_at": utc_now().isoformat().replace("+00:00", "Z"),
        "latest_run_id": run_id,
        "runs": runs,
    }
    json_dump(index_path, index)

    print(f"Published weekly benchmark result to {result_dest}")
    print(f"Updated {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
