#!/usr/bin/env python3
"""Record the gzipped size of manifold.wasm for every release on npm.

The weekly job measures the wasm it builds, which gives a dense line but only
from the day it starts running. This fills in the other axis: every version
ever published, going back to 1.0.0.

Sizes are gzipped because that is what a browser waits for, and it is the
number web libraries usually quote. Compression happens here rather than being
read from a CDN header, so the series does not move when a CDN changes its
settings. It uses Python's gzip for the same reason the weekly job does: GNU
gzip and zlib disagree by ~0.4% at the same level, which would otherwise put a
permanent offset between the two series.

The whole series is rebuilt on each run rather than appended to. None of the
answers change once a version is published, so a backfill is then the same
code path as an update.
"""
import argparse
import gzip
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2
PACKAGE = "manifold-3d"
REGISTRY_URL = f"https://registry.npmjs.org/{PACKAGE}"
WASM_URL = "https://cdn.jsdelivr.net/npm/{package}@{version}/manifold.wasm"
GZIP_LEVEL = 9
TIMEOUT = 60


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read()


def wasm_gzip_bytes(version: str) -> int | None:
    try:
        wasm = fetch(WASM_URL.format(package=PACKAGE, version=version))
    except urllib.error.HTTPError as error:
        # 2.3.0 is published but ships no wasm.
        if error.code == 404:
            return None
        raise
    return len(gzip.compress(wasm, GZIP_LEVEL))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="JSON file to write")
    args = parser.parse_args()

    registry = json.loads(fetch(REGISTRY_URL))
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
        size = wasm_gzip_bytes(version)
        if size is None:
            skipped.append(version)
            continue
        releases.append(
            {
                "version": version,
                "date": published[version][:10],
                "gzip_bytes": size,
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
