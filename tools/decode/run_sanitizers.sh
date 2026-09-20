#!/usr/bin/env bash
# T9.3: AddressSanitizer + UndefinedBehaviorSanitizer gate for AInfer host code.
# Builds host-side targets with -fsanitize=address,undefined and runs:
#   1. test_arena_spans  (T9.1 span/arena units)
#   2. test_exec_guards  (T9.2 step-guard units)
#   3. l0load full 712-tensor load (T3.5 loader validation)
#   4. negatives.py 12-case rejection suite (T3.6, exercises OOB paths)
#   5. test_runtime_258v full M4 suite (T5.1-T5.6 incl. model load)
# Exit 0 iff every stage builds, passes, and reports zero sanitizer findings.
# Device kernels are out of scope for sanitizers (host code only).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYSROOT="$REPO/tools/toolchain/sysroot/usr/lib"
INCL="$REPO/tools/l0probe/include"
OUTDIR="${1:-/tmp/ainfer_asan}"
mkdir -p "$OUTDIR"
SANFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -std=c++17"
export LD_LIBRARY_PATH="$SYSROOT"
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
PASS=0
FAIL=0
stage() { echo "=== T9.3 [$1] $2 ==="; }
ok()   { echo "T9.3 PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "T9.3 FAIL: $1"; FAIL=$((FAIL+1)); }

CXX="${CXX:-g++}"

stage build "unit tests"
$CXX $SANFLAGS "$REPO/tools/decode/test_arena_spans.cpp" -I"$INCL" -o "$OUTDIR/test_arena_spans" \
  && $CXX $SANFLAGS "$REPO/tools/decode/test_exec_guards.cpp" -I"$INCL" -o "$OUTDIR/test_exec_guards" \
  && ok "unit test builds" || bad "unit test builds"

stage run "test_arena_spans"
"$OUTDIR/test_arena_spans" >/dev/null 2>"$OUTDIR/asan_arena.log" && ok "arena spans" || bad "arena spans (see $OUTDIR/asan_arena.log)"

stage run "test_exec_guards"
"$OUTDIR/test_exec_guards" >/dev/null 2>"$OUTDIR/asan_guards.log" && ok "exec guards" || bad "exec guards (see $OUTDIR/asan_guards.log)"

stage build "l0load + runtime"
$CXX $SANFLAGS "$REPO/tools/l0load/loader.cpp" -I"$INCL" -I"$REPO/tools/toolchain/sysroot/usr/include" \
  -L"$SYSROOT" -lze_loader -o "$OUTDIR/l0load" \
  && $CXX $SANFLAGS "$REPO/tools/decode/runtime_258v.cpp" "$REPO/tools/decode/test_runtime_258v.cpp" \
  -I"$INCL" -L"$SYSROOT" -lze_loader -o "$OUTDIR/test_runtime" \
  && ok "l0load + runtime builds" || bad "l0load + runtime builds"

if [ -x "$OUTDIR/l0load" ]; then
  stage run "l0load full container"
  "$OUTDIR/l0load" "$REPO/models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer" \
    "$OUTDIR/report_l0load_asan.json" >/dev/null 2>"$OUTDIR/asan_l0load.log" \
    && ok "l0load 712/712" || bad "l0load (see $OUTDIR/asan_l0load.log)"

  stage run "negatives.py x12"
  python3 "$REPO/tools/l0load/negatives.py" "$OUTDIR/l0load" "$OUTDIR/report_l0neg_asan.json" \
    >/dev/null 2>"$OUTDIR/asan_l0neg.log" \
    && ok "negatives 12/12" || bad "negatives (see $OUTDIR/asan_l0neg.log)"
fi

if [ -x "$OUTDIR/test_runtime" ]; then
  stage run "M4 suite"
  "$OUTDIR/test_runtime" >/dev/null 2>"$OUTDIR/asan_m4.log" \
    && ok "M4 7/7" || bad "M4 suite (see $OUTDIR/asan_m4.log)"
fi

echo "=== T9.3 summary: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
