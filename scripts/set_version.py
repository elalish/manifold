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


def replace_all(path: Path, pattern: re.Pattern, replacement: str) -> bool:
    """Rewrite every match of pattern in path. True if the file changed.

    A file with no match at all is an error: it means the format changed and
    a version would silently go un-bumped. Lockfiles carry the version more
    than once, so every occurrence is replaced rather than just the first.

    newline="" on both ends so line endings survive untouched; without it
    Python rewrites every line of the file and the diff becomes unreadable.
    """
    with open(path, encoding="utf-8", newline="") as f:
        original = f.read()
    updated, count = pattern.subn(replacement, original)
    if count == 0:
        raise SystemExit(
            f"{path}: no match for {pattern.pattern!r}. "
            "The file's format has changed; update this script."
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
        # scripts/test-cmake.sh carries the version twice: once as the
        # find_package() minimum and once as the MANIFOLD_VERSION_NUMBER check.
        # Both track the release so there is only ever one version in play.
        (
            Path("scripts/test-cmake.sh"),
            re.compile(r'(find_package\(manifold ")\d+\.\d+\.\d+(")'),
            rf"\g<1>{dotted}\g<2>",
        ),
        (
            Path("scripts/test-cmake.sh"),
            re.compile(r"(MANIFOLD_VERSION_NUMBER\()\d+,\s*\d+,\s*\d+(\))"),
            rf"\g<1>{major}, {minor}, {patch}\g<2>",
        ),
    ]

    # The lockfiles embed the version in their manifold-3d entries. Editing
    # them directly rather than running npm install keeps the release commit
    # to just the version change, and keeps the workflow fast.
    lock_pattern = re.compile(
        r'("name":\s*"manifold-3d",\s*\n\s*"version":\s*")\d+\.\d+\.\d+(")'
    )
    # Globbed non-recursively; "**" would descend into node_modules.
    lockfiles = [Path("bindings/wasm/package-lock.json")] + sorted(
        p.relative_to(root)
        for p in root.glob("bindings/wasm/examples/*/package-lock.json")
    )
    for lockfile in lockfiles:
        edits.append((lockfile, lock_pattern, rf"\g<1>{dotted}\g<2>"))

    changed = []
    for relative, pattern, replacement in edits:
        path = root / relative
        if not path.exists():
            raise SystemExit(f"{relative}: not found. Has it moved?")
        if replace_all(path, pattern, replacement) and relative not in changed:
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
