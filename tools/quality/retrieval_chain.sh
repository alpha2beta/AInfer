#!/bin/bash
# T8.5 retrieval chain: merged single-binary prefill->decode per case.
# Usage: retrieval_chain.sh [--case ID]  (run from repo root, background)
# Chunks + question prepared by mk_retrieval.py. FAILS FAST per case
# (records FAIL and continues to next case; chain aborts only on prefill crash).
set -u
cd /mnt/usb/AInfer
DEC=./build-b60/tools/decode/decode_l0
MODEL=models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
ONLY=""
if [ "${1:-}" = "--case" ]; then ONLY="$2"; fi
for d in /mnt/usb/retr/retr-*; do
  id=$(basename "$d")
  if [ -n "$ONLY" ] && [ "$id" != "$ONLY" ]; then continue; fi
  if [ -f "$d/done" ]; then echo "$id: skip (done)"; continue; fi
  ctx=$(~/.venvs/ainfer/bin/python -c "import json;print(json.load(open('$d/manifest.json'))['ctx'])")
  nch=$((ctx / 256))
  echo "===== $id ctx=$ctx nch=$nch $(date) ====="
  if ! AINFER_MAXCTX=$((ctx + 64)) CHUNK_M=256 CHUNK64MC_HOST_IN="$d/ch" \
    $DEC $MODEL 1 1 build-b60/tools/cmdlist "$d/gen.json" \
    --ids-file="$d/q.txt" --max-new=24 --prefill-chunks=$nch \
    > "$d/dec.log" 2>&1; then
    echo "$id: PREFILL/DECODE FAILED"; echo "FAIL" > "$d/done"; continue
  fi
  if ~/.venvs/ainfer/bin/python tools/quality/check_retrieval.py "$id"; then
    echo "PASS" > "$d/done"
  else
    echo "FAIL" > "$d/done"
  fi
done
echo "RETRIEVAL CHAIN DONE $(date)"
