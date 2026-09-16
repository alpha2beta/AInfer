"""T7.4 64-layer orchestration reference check: float INT4-dequant chain over
the same 32-token chunk the device ran (input read from /tmp/chunk64_in.bin,
written by chunk64real_replay), vs device output (/tmp/chunk64_out.bin).

Reuses fwd_cpu.manual_block + layer_tensors_int4 (complete per-layer dicts,
streamed one shard at a time). Rope at positions 0..31, causal mask for full
layers, initial SSM state None (single chunk, P=0). Writes
tools/t74/report_chunk64.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "forward"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import fwd_cpu as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding)

M = 32
H = 5120


def main():
    t0 = time.time()
    xin = np.fromfile("/tmp/chunk64_in.bin", dtype=np.float32)
    assert xin.size == M * H, xin.size
    x = torch.from_numpy(xin).reshape(1, M, H)
    rope = Qwen3_5TextRotaryEmbedding(F.cfg)
    pos = torch.arange(M).unsqueeze(0)
    cos, sin = rope(torch.empty(1, M, 5120), pos)
    mask = torch.triu(torch.full((M, M), float("-inf")), 1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    assert F.LT.count("full_attention") == 16
    for L in range(64):
        W = F.layer_tensors_int4(L)
        with torch.no_grad():
            x = F.manual_block(W, L, x, cos, sin,
                               mask if F.LT[L] == "full_attention" else None)
        del W
        if (L + 1) % 16 == 0:
            print(f"  ref layer {L + 1}/64 ({time.time() - t0:.0f}s)",
                  flush=True)
    ref = x.reshape(-1).numpy()
    got = np.fromfile("/tmp/chunk64_out.bin", dtype=np.float32)
    assert got.size == ref.size, (got.size, ref.size)
    refmax = float(np.abs(ref).max())
    worst = float(np.abs(got - ref).max() / (refmax or 1))
    mean = float(np.abs(got - ref).mean() / (refmax or 1))
    ok = bool(np.isfinite(got).all() and worst <= 1e-2)
    print(f"chunk64 ref worst-rel {worst:.2e} mean-rel {mean:.2e} "
          f"{'CHUNK64-OK' if ok else 'MISMATCH'}", flush=True)
    rep = {"device": "B60",
           "chunk64": "64-layer chunked prefill vs float INT4-dequant chain",
           "M": M, "P": 0, "worst_rel": worst, "mean_rel": mean,
           "ref_tol": 1e-2, "refmax": refmax,
           "seconds": round(time.time() - t0, 1),
           "chunk64_ok": ok}
    json.dump(rep, open(os.path.join(REPO, "tools", "t74",
                                     "report_chunk64.json"), "w"), indent=1)
    print("WROTE report_chunk64.json", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
