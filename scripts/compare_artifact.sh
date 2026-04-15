#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-meshes}"
FILES="${DETERMINISM_COMPARE_FILES:-}"
NON_BLOCKING="${DETERMINISM_COMPARE_NON_BLOCKING:-0}"
LABEL="${DETERMINISM_COMPARE_LABEL:-Cross-platform determinism}"
DIFF_LINES="${DETERMINISM_COMPARE_DIFF_LINES:-200}"
NORMALIZE_SCRIPT="${DETERMINISM_NORMALIZE_SCRIPT:-./scripts/normalize_objs.sh}"

if [ -z "$FILES" ]; then
  echo "DETERMINISM_COMPARE_FILES is required."
  exit 2
fi

for os in linux mac windows; do
  if [ ! -d "$ROOT_DIR/$os" ]; then
    echo "Missing directory: $ROOT_DIR/$os"
    if [ "$NON_BLOCKING" = "1" ]; then
      echo "::warning::$LABEL: missing artifacts under $ROOT_DIR."
      exit 0
    fi
    exit 1
  fi
done

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

failed=0
mismatches=""

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
    mismatches="$mismatches $f(missing)"
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
    mismatches="$mismatches $f"
  fi
done

if [ "$failed" -ne 0 ]; then
  if [ "$NON_BLOCKING" = "1" ]; then
    echo "::warning::$LABEL found cross-platform mismatches:$mismatches"
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
      {
        echo "### $LABEL"
        echo ""
        echo "Mismatches:$mismatches"
      } >> "$GITHUB_STEP_SUMMARY"
    fi
    exit 0
  fi
  echo "::error::$LABEL failed."
  exit 1
fi

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### $LABEL"
    echo ""
    echo "No cross-platform mismatches detected."
  } >> "$GITHUB_STEP_SUMMARY"
fi
