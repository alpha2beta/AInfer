#!/bin/bash
# 64K needle eval chain (variants 0/1/3/4): prefill -> import-decode -> check.
# Usage: needle_chain.sh (run from repo root, background with nohup)
# Dumps go to /mnt/usb (tmpfs too small for 4.4 GB/variant). FAILS FAST:
# any stage failure stops the chain (a silent skip-through once voided
# variant 0/1 results).
# Each variant: ~80 min prefill + ~3 min decode.
set -u
cd /mnt/usb/AInfer
BIN=./build-b60/tools/cmdlist/chunk64mc_replay
DEC=./build-b60/tools/decode/decode_l0
SPV="build-b60/tools/cmdlist/norm.spv build-b60/tools/cmdlist/chunkgemm.spv build-b60/tools/cmdlist/chunkssmconv.spv build-b60/tools/cmdlist/chunkssmrecur.spv build-b60/tools/cmdlist/silumul.spv build-b60/tools/cmdlist/splitrepeat.spv build-b60/tools/cmdlist/l2normqk.spv build-b60/tools/cmdlist/betag.spv build-b60/tools/cmdlist/rmsinv.spv build-b60/tools/cmdlist/normgated.spv build-b60/tools/cmdlist/resaddf.spv build-b60/tools/cmdlist/cvtf32f16.spv build-b60/tools/cmdlist/splitqk.spv build-b60/tools/cmdlist/batchnorm.spv build-b60/tools/cmdlist/chunkrope.spv build-b60/tools/cmdlist/chunkkvappend.spv build-b60/tools/cmdlist/chunkqkgemm.spv build-b60/tools/cmdlist/chunksoftmaxrow.spv build-b60/tools/cmdlist/chunkwvgemm.spv build-b60/tools/cmdlist/gatemul.spv"
MODEL=models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
for v in 0 1 3 4; do
  echo "===== variant $v $(date) ====="
  rm -rf /mnt/usb/kvneedle$v; mkdir -p /mnt/usb/kvneedle$v
  # shellcheck disable=SC2086
  CHUNK_M=256 CHUNK_N=254 CHUNK_STREAM=1 CHUNK64MC_HOST_IN=/tmp/needle$v/ch \
    CHUNK_DUMP_CACHES=/mnt/usb/kvneedle$v \
    $BIN $MODEL $SPV /tmp/needle_pf$v.json > /tmp/needle_pf$v.log 2>&1 \
    || { echo "PREFILL FAILED v$v"; exit 1; }
  echo "prefill done v$v $(tail -1 /tmp/needle_pf$v.log | head -c 100)"
  ~/.venvs/ainfer/bin/python tools/t74/mk_needle_ids.py "$v" || exit 1
  AINFER_STEPLOG=1 $DEC $MODEL 65043 1 build-b60/tools/cmdlist \
    /tmp/needle_gen$v.json --ids-file=/tmp/ids_needle$v.txt --max-new=64 \
    --import-caches=/mnt/usb/kvneedle$v > /tmp/needle_dec$v.log 2>&1 \
    || { echo "DECODE FAILED v$v"; exit 1; }
  echo "decode done v$v"
  ~/.venvs/ainfer/bin/python tools/t74/needle_check.py "$v" /tmp/needle_gen$v.json \
    || { echo "CHECK FAILED v$v"; exit 1; }
  rm -rf /mnt/usb/kvneedle$v
done
echo "CHAIN DONE $(date)"
