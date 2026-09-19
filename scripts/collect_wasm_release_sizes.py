#!/usr/bin/env python3
"""Record the published size of manifold.wasm for every release on npm.

The weekly job measures the wasm it builds, which gives a dense line but only
from the day it starts running. This fills in the other axis: every version
ever published, going back to 1.0.0, read from jsDelivr's metadata API.

The whole series is rebuilt on each run rather than appended to. It is ~30
requests, none of the answers change once a version is published, and a
backfill is then the same code path as an update.
"""
import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
PACKAGE = "manifold-3d"
WASM_PATH = "/manifold.wasm"
REGISTRY_URL = f"https://registry.npmjs.org/{PACKAGE}"
# structure=flat returns one list of files with sizes, rather than a tree.
JSDELIVR_URL = (
    "https://data.jsdelivr.com/v1/packages/npm/{package}@{version}?structure=flat"
)
TIMEOUT = 30


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def wasm_size(version: str) -> int | None:
    files = fetch_json(JSDELIVR_URL.format(package=PACKAGE, version=version))["files"]
    return next((f["size"] for f in files if f["name"] == WASM_PATH), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="JSON file to write")
    args = parser.parse_args()

    registry = fetch_json(REGISTRY_URL)
    published = registry["time"]
    # Ordered by publish date rather than semver: the chart's x-axis is
    # chronological, and ISO dates sort lexically. Prereleases carry - or +.
    versions = sorted(
        (v for v in registry["versions"] if not any(c in v for c in "-+")),
        key=published.get,
    )

    releases = []
    skipped = []
    for version in versions:
        size = wasm_size(version)
        if size is None:
            # 2.3.0 is published but ships no wasm. Recording it as zero would
            # put a false cliff in the chart.
            skipped.append(version)
            continue
        releases.append(
            {
                "version": version,
                "date": published[version][:10],
                "size_bytes": size,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "releases": releases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(releases)} releases to {args.output}")
    if skipped:
        print(f"No wasm published for: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
