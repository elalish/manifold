#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-boolean-dumps}"
LIMIT_LINES=120

warn() {
  echo "::warning::$1"
}

if [ ! -d "$ROOT_DIR/linux" ] || [ ! -d "$ROOT_DIR/mac" ] || [ ! -d "$ROOT_DIR/windows" ]; then
  warn "Boolean dump directories missing under '$ROOT_DIR'; skipping dump comparison."
  exit 0
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

for os in linux mac windows; do
  (
    cd "$ROOT_DIR/$os"
    find . -type f -print | LC_ALL=C sort
  ) > "$tmpdir/$os.files"
done

failed=0
mismatches=()

if ! diff -u "$tmpdir/linux.files" "$tmpdir/mac.files" > "$tmpdir/filelist_linux_mac.diff"; then
  echo "Dump file list mismatch: linux vs mac"
  sed -n "1,${LIMIT_LINES}p" "$tmpdir/filelist_linux_mac.diff"
  failed=1
fi

if ! diff -u "$tmpdir/linux.files" "$tmpdir/windows.files" > "$tmpdir/filelist_linux_windows.diff"; then
  echo "Dump file list mismatch: linux vs windows"
  sed -n "1,${LIMIT_LINES}p" "$tmpdir/filelist_linux_windows.diff"
  failed=1
fi

while IFS= read -r rel; do
  [ -z "$rel" ] && continue

  missing=0
  for os in linux mac windows; do
    if [ ! -f "$ROOT_DIR/$os/$rel" ]; then
      echo "Missing dump file: $ROOT_DIR/$os/$rel"
      missing=1
      failed=1
    fi
  done
  [ "$missing" -ne 0 ] && continue

  for os in linux mac windows; do
    tr -d '\r' < "$ROOT_DIR/$os/$rel" > "$tmpdir/$os.norm"
  done

  linux_hash="$(sha256sum "$tmpdir/linux.norm" | awk '{print $1}')"
  mac_hash="$(sha256sum "$tmpdir/mac.norm" | awk '{print $1}')"
  win_hash="$(sha256sum "$tmpdir/windows.norm" | awk '{print $1}')"

  if [ "$linux_hash" != "$mac_hash" ] || [ "$linux_hash" != "$win_hash" ]; then
    mismatches+=("$rel")
    echo "--- dump mismatch: $rel ---"
    echo "linux:   $linux_hash"
    echo "mac:     $mac_hash"
    echo "windows: $win_hash"
    echo "linux vs mac diff:"
    diff "$tmpdir/linux.norm" "$tmpdir/mac.norm" | sed -n "1,${LIMIT_LINES}p" || true
    echo "linux vs windows diff:"
    diff "$tmpdir/linux.norm" "$tmpdir/windows.norm" | sed -n "1,${LIMIT_LINES}p" || true
    failed=1
  fi
done < "$tmpdir/linux.files"

if [ "$failed" -ne 0 ]; then
  warn "Boolean debug dump comparison found cross-platform mismatches."
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
      echo "### Boolean debug dump comparison"
      echo ""
      echo "Cross-platform mismatches detected in intermediate dump files."
      echo ""
      echo "Mismatched files:"
      if [ "${#mismatches[@]}" -eq 0 ]; then
        echo "- (none listed; see job log for file-list differences)"
      else
        for rel in "${mismatches[@]}"; do
          echo "- ${rel#./}"
        done
      fi
    } >> "$GITHUB_STEP_SUMMARY"
  fi
else
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
      echo "### Boolean debug dump comparison"
      echo ""
      echo "No cross-platform mismatches detected in intermediate dump files."
    } >> "$GITHUB_STEP_SUMMARY"
  fi
fi

# Discovery helper: always non-blocking.
exit 0
