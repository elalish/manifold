#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <base-sha> <head-sha>"
  exit 2
fi

BASE_SHA="$1"
HEAD_SHA="$2"

git worktree add wt-base "$BASE_SHA"
git worktree add wt-head "$HEAD_SHA"
