#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OUT=${1:-"$ROOT/tools/kernels_258v/kv8_primitive.spv"}
if command -v ocloc >/dev/null 2>&1; then
  ocloc compile -device lnl -file "$ROOT/tools/kernels_258v/kv8_primitive.cl" -output "$OUT"
  cp "${OUT}_lnl.spv" "$OUT"
else
  clang -target spirv64 -x cl -cl-std=CL2.0 -O2 \
    -c "$ROOT/tools/kernels_258v/kv8_primitive.cl" -o "$OUT"
fi
printf 'wrote %s\n' "$OUT"
