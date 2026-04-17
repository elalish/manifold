#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-meshes}"
FILES="${DETERMINISM_COMPARE_FILES:-}"
DIFF_LINES="${DETERMINISM_COMPARE_DIFF_LINES:-200}"
NORMALIZE_SCRIPT="${DETERMINISM_NORMALIZE_SCRIPT:-./scripts/normalize_objs.sh}"

if [ -z "$FILES" ]; then
  echo "DETERMINISM_COMPARE_FILES is required."
  exit 2
fi

for os in linux mac windows; do
  if [ ! -d "$ROOT_DIR/$os" ]; then
    echo "Missing directory: $ROOT_DIR/$os"
    exit 1
  fi
done

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

failed=0

for f in $FILES; do
  echo "--- $f ---"
  file_missing=0
  for os in linux mac windows; do
    if [ ! -f "$ROOT_DIR/$os/$f" ]; then
      echo "Missing file: $ROOT_DIR/$os/$f"
      failed=1
      file_missing=1
    fi
  done
  if [ "$file_missing" -ne 0 ]; then
    continue
  fi

  for os in linux mac windows; do
    bash "$NORMALIZE_SCRIPT" "$ROOT_DIR/$os/$f" "$tmpdir/$os-$f"
  done

  linux_hash="$(sha256sum "$tmpdir/linux-$f" | awk '{print $1}')"
  mac_hash="$(sha256sum "$tmpdir/mac-$f" | awk '{print $1}')"
  win_hash="$(sha256sum "$tmpdir/windows-$f" | awk '{print $1}')"
  echo "linux:   $linux_hash"
  echo "mac:     $mac_hash"
  echo "windows: $win_hash"

  if [ "$linux_hash" != "$mac_hash" ] || [ "$linux_hash" != "$win_hash" ]; then
    echo "MISMATCH in $f"
    echo "linux vs mac diff:"
    diff "$tmpdir/linux-$f" "$tmpdir/mac-$f" | sed -n "1,${DIFF_LINES}p" || true
    echo "linux vs windows diff:"
    diff "$tmpdir/linux-$f" "$tmpdir/windows-$f" | sed -n "1,${DIFF_LINES}p" || true
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "::error::Cross-platform determinism check failed."
  exit 1
fi

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### Cross-platform determinism check"
    echo ""
    echo "No cross-platform mismatches detected."
  } >> "$GITHUB_STEP_SUMMARY"
fi
