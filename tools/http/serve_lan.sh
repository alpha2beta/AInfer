#!/usr/bin/env bash
# AInfer 258V LAN server launcher.
#
# Starts the resident inference daemon (tools/http/server_258v.py) bound for
# the LAN so OpenCode (or any OpenAI-compatible client) on ANOTHER pc can
# call it for real test cases:
#
#   ./tools/http/serve_lan.sh [--port 8080] [--max-ctx 4096] [--timeout 600]
#                             [--bind 0.0.0.0] [--kv8 | --bf16]
#                             [--no-speculative] [--no-rebuild]
#
# Remote usage (from the other PC, HOST = this machine's LAN IP printed below):
#   curl http://HOST:PORT/healthz
#   curl -N http://HOST:PORT/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model":"tiel-coder-35b","messages":[{"role":"user","content":"Reverse a string in Python."}],"max_tokens":64}'
#
# Notes:
# - Model stays resident in-process (~19 GB); first start takes ~1 min to load.
# - Single-flight execution: one generation at a time, queue-size waiting slots.
# - max_tokens is capped at 2048 per request by the server.
# - Decode is slow at long context (~2 tok/s at 32K); set --timeout generously.
# - KV policy: unset env -> auto-KV8 at max-ctx >= 16384; --kv8 forces INT8 KV
#   (halves KV traffic); --bf16 forces BF16 KV.
set -euo pipefail

HOST="0.0.0.0"
PORT="8080"
MAX_CTX="4096"
TIMEOUT="600"
QUEUE="16"
SPEC="--speculative"
REBUILD="1"
KV_MODE="auto"

usage() {
  sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'
  echo "Options: --port P --bind H --max-ctx N --timeout S --queue-size Q"
  echo "         --kv8 --bf16 --no-speculative --no-rebuild -h/--help"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --bind) HOST="$2"; shift 2;;
    --max-ctx) MAX_CTX="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --queue-size) QUEUE="$2"; shift 2;;
    --kv8) KV_MODE="kv8"; shift;;
    --bf16) KV_MODE="bf16"; shift;;
    --no-speculative) SPEC="--no-speculative"; shift;;
    --no-rebuild) REBUILD="0"; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown flag: $1" >&2; usage >&2; exit 2;;
  esac
done

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
PY="$HOME/.venvs/ainfer/bin/python"
MODEL="$ROOT/models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer"
SPV="$ROOT/tools/kernels_258v/all_kernels.spv"
SO="$ROOT/tools/decode/libainfer_258v.so"
SYSROOT_LIB="$ROOT/tools/toolchain/sysroot/usr/lib"

fail() { echo "serve_lan: ERROR: $1" >&2; exit 1; }
note() { echo "serve_lan: $1"; }

# ---- preflight ----
[ -x "$PY" ] || fail "project venv python not found at $PY"
[ -f "$MODEL" ] || fail "model container missing: $MODEL"
[ -f "$SPV" ] || fail "SPIR-V bundle missing: $SPV"
[ -f "$SYSROOT_LIB/libze_loader.so.1" ] || fail "Level Zero loader missing in sysroot"
[ -f "$ROOT/tools/kernels_258v/all_kernels.spv.attn" ] \
  && note "decode-attention override present (auto-loaded, ~1.2x long-context decode)" \
  || note "WARNING: all_kernels.spv.attn absent — bundled (slower) decode attention will be used"
if [ "$KV_MODE" = "kv8" ] || [ "$MAX_CTX" -ge 16384 ]; then
  [ -f "$SPV.kv8" ] || fail "KV8 companion missing ($SPV.kv8); rebuild via tools/kernels_258v/build_kv8_runtime_companion.sh"
fi
if ! lspci -nn 2>/dev/null | grep -qi "8086.*[Vv]GA\|Intel.*[Gg]raphics"; then
  ls /dev/dri/renderD* >/dev/null 2>&1 || note "WARNING: no Intel GPU detected via lspci//dev/dri — init will fail if no Arc 140V"
fi
if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  fail "port $PORT already in use (ss -ltn). Pick another with --port."
fi

# ---- rebuild resident .so if sources are newer (unless --no-rebuild) ----
if [ "$REBUILD" = "1" ]; then
  if [ "$SO" -ot "$ROOT/tools/decode/runtime_258v.cpp" ] || \
     [ "$SO" -ot "$ROOT/tools/decode/c_api_258v.cpp" ] || \
     [ "$SO" -ot "$ROOT/tools/decode/runtime_258v.h" ] || \
     [ ! -f "$SO" ]; then
    note "rebuilding libainfer_258v.so from current source..."
    g++ -O2 -shared -fPIC -std=c++17 \
      "$ROOT/tools/decode/runtime_258v.cpp" "$ROOT/tools/decode/c_api_258v.cpp" \
      -I"$ROOT/tools/l0probe/include" -I"$ROOT/tools/decode" \
      -L"$SYSROOT_LIB" -lze_loader -o "$SO"
    note "rebuilt $SO"
  else
    note "libainfer_258v.so is up to date"
  fi
fi

# ---- KV policy env ----
case "$KV_MODE" in
  kv8) export AINFER_KV8=1; note "KV mode: forced INT8 (AINFER_KV8=1)";;
  bf16) export AINFER_KV8=0; note "KV mode: forced BF16 (AINFER_KV8=0)";;
  auto)
    unset AINFER_KV8 || true
    if [ "$MAX_CTX" -ge 16384 ]; then note "KV mode: auto-INT8 (max-ctx >= 16384)";
    else note "KV mode: auto-BF16 (max-ctx < 16384)"; fi;;
esac

# ---- LAN address for the other PC ----
LAN_IP=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1 || true)
[ -z "$LAN_IP" ] && LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$LAN_IP" ] && LAN_IP="<this-host-ip>"

echo "================================================================="
echo " AInfer 258V resident server — LAN launch"
echo "  Local:  http://127.0.0.1:$PORT/healthz"
echo "  Remote (other PC): http://$LAN_IP:$PORT"
echo "  Max-ctx: $MAX_CTX | Timeout: ${TIMEOUT}s | Queue: $QUEUE | Speculative: $SPEC"
echo "-----------------------------------------------------------------"
echo " If the other PC cannot connect, open the port on this machine, e.g.:"
echo "   sudo ufw allow $PORT/tcp   # or: sudo firewall-cmd --add-port=$PORT/tcp --permanent && sudo firewall-cmd --reload"
echo " Remote smoke test (other PC):"
echo "   curl http://$LAN_IP:$PORT/healthz"
echo "   curl http://$LAN_IP:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\":\"tiel-coder-35b\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":16}'"
echo "================================================================="

export LD_LIBRARY_PATH="$SYSROOT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PY" "$ROOT/tools/http/server_258v.py" \
  --host "$HOST" --port "$PORT" \
  --model "$MODEL" --spv "$SPV" \
  --max-ctx "$MAX_CTX" --timeout "$TIMEOUT" --queue-size "$QUEUE" "$SPEC"
