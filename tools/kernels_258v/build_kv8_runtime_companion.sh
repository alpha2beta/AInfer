#!/usr/bin/env bash
set -euo pipefail

# Build the standalone KV8 module expected beside the main runtime SPIR-V:
# all_kernels.spv.kv8. It intentionally contains only the four trunk KV8
# entry points; do not link it with all_kernels.spv (duplicate entry points
# caused a device-loss failure during the first integration attempt).
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OUT=${1:-"$ROOT/tools/kernels_258v/all_kernels.spv.kv8"}
if command -v ocloc >/dev/null 2>&1; then
  ocloc compile -device lnl -file "$ROOT/tools/kernels_258v/kv8_primitive.cl" -output "$OUT"
  cp "${OUT}_lnl.spv" "$OUT"
else
  clang -target spirv64 -x cl -cl-std=CL2.0 -O2 \
    -c "$ROOT/tools/kernels_258v/kv8_primitive.cl" -o "$OUT"
fi
printf 'wrote %s\n' "$OUT"
