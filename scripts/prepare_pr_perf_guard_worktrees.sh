#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <base-sha> <head-sha>"
  exit 2
fi

BASE_SHA="$1"
HEAD_SHA="$2"

if [ -z "$BASE_SHA" ]; then
  echo "Error: BASE_SHA is empty"
  exit 1
fi

if [ -z "$HEAD_SHA" ]; then
  echo "Error: HEAD_SHA is empty"
  exit 1
fi

git worktree add wt-base "$BASE_SHA"
git worktree add wt-head "$HEAD_SHA"
