#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OUT=${1:-"$ROOT/tools/kernels_258v/kv8_primitive.spv"}
TMP=${TMPDIR:-/tmp}/ainfer-kv8-primitive
mkdir -p "$TMP"

"$ROOT/tools/kernels_258v/build_kv8_primitive.sh" "$OUT"
g++ -O2 -std=c++17 "$ROOT/tools/kernels_258v/test_kv8_primitive.cpp" \
  -I"$ROOT/tools/l0probe/include" -L"$ROOT/tools/toolchain/sysroot/usr/lib" \
  -lze_loader -o "$TMP/test_kv8_primitive"
g++ -O2 -std=c++17 "$ROOT/tools/kernels_258v/test_kv8_batch_primitive.cpp" \
  -I"$ROOT/tools/l0probe/include" -L"$ROOT/tools/toolchain/sysroot/usr/lib" \
  -lze_loader -o "$TMP/test_kv8_batch_primitive"

export LD_LIBRARY_PATH="$ROOT/tools/toolchain/sysroot/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$TMP/test_kv8_primitive" "$OUT" "$TMP/report_kv8_primitive.json"
"$TMP/test_kv8_batch_primitive" "$OUT" "$TMP/report_kv8_batch.json"
python3 - "$TMP/report_kv8_primitive.json" "$TMP/report_kv8_batch.json" <<'PY'
import json, sys
reports = [json.load(open(p)) for p in sys.argv[1:]]
if any(r.get("status") != "PASSED" for r in reports):
    raise SystemExit(1)
print(json.dumps({"status": "PASSED", "reports": reports}, indent=2))
PY
