#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input-obj> <output-obj>"
  exit 2
fi

input="$1"
output="$2"

tr -d '\r' < "$input" > "$output"
