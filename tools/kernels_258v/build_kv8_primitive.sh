#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OUT=${1:-"$ROOT/tools/kernels_258v/kv8_primitive.spv"}
clang -target spirv64 -x cl -cl-std=CL2.0 -O2 \
  -c "$ROOT/tools/kernels_258v/kv8_primitive.cl" -o "$OUT"
printf 'wrote %s\n' "$OUT"
