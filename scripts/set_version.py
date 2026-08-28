#!/usr/bin/env python3
"""Set the release version across every file that carries it.

The version lives in five places in five different formats, and keeping them
in sync by hand is how master ended up on 3.5.1 while v3.5.2 was already
tagged. Run this instead:

    python3 scripts/set_version.py 3.6.0

Prints the files it changed. Exits non-zero if any file was already at the
requested version but others were not, since that means a previous bump was
only partly applied.
"""
import argparse
import re
import sys
from pathlib import Path

VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def replace_once(path: Path, pattern: re.Pattern, replacement: str) -> bool:
    """Rewrite the single match of pattern in path. True if the file changed.

    newline="" on both ends so line endings survive untouched; without it
    Python rewrites every line of the file and the diff becomes unreadable.
    """
    with open(path, encoding="utf-8", newline="") as f:
        original = f.read()
    updated, count = pattern.subn(replacement, original, count=1)
    if count != 1:
        raise SystemExit(
            f"{path}: expected exactly one match for {pattern.pattern!r}, "
            f"found {count}. The file's format has changed; update this script."
        )
    if updated == original:
        return False
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(updated)
    return True


def set_version(root: Path, major: str, minor: str, patch: str) -> list[Path]:
    dotted = f"{major}.{minor}.{patch}"
    edits = [
        (
            Path("bindings/wasm/package.json"),
            re.compile(r'("version":\s*")\d+\.\d+\.\d+(")'),
            rf"\g<1>{dotted}\g<2>",
        ),
        (
            Path("pyproject.toml"),
            re.compile(r'(?m)^(version\s*=\s*")\d+\.\d+\.\d+(")'),
            rf"\g<1>{dotted}\g<2>",
        ),
        (
            Path("flake.nix"),
            re.compile(r'(manifold-version\s*=\s*")\d+\.\d+\.\d+(")'),
            rf"\g<1>{dotted}\g<2>",
        ),
        (
            Path("CMakeLists.txt"),
            re.compile(
                r"(set\(MANIFOLD_VERSION_MAJOR )\d+(\)\s*\n"
                r"set\(MANIFOLD_VERSION_MINOR )\d+(\)\s*\n"
                r"set\(MANIFOLD_VERSION_PATCH )\d+(\))"
            ),
            rf"\g<1>{major}\g<2>{minor}\g<3>{patch}\g<4>",
        ),
        # scripts/test-cmake.sh
        # This asserts the version just built is available to consumers, so it
        # tracks the release. The separate find_package() minimum above it is
        # deliberately older and is left alone.
        (
            Path("scripts/test-cmake.sh"),
            re.compile(r"(MANIFOLD_VERSION_NUMBER\()\d+,\s*\d+,\s*\d+(\))"),
            rf"\g<1>{major}, {minor}, {patch}\g<2>",
        ),
    ]

    changed = []
    for relative, pattern, replacement in edits:
        path = root / relative
        if not path.exists():
            raise SystemExit(f"{relative}: not found. Has it moved?")
        if replace_once(path, pattern, replacement):
            changed.append(relative)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="release version, e.g. 3.6.0")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (defaults to this script's parent directory)",
    )
    args = parser.parse_args()

    match = VERSION_PATTERN.match(args.version)
    if not match:
        raise SystemExit(
            f"'{args.version}' is not a MAJOR.MINOR.PATCH version. "
            "Strip any leading 'v'."
        )

    changed = set_version(args.root, *match.groups())
    if not changed:
        print(f"Already at {args.version}; nothing to do.")
        return 0

    print(f"Set version to {args.version} in:")
    for path in changed:
        print(f"  {path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
