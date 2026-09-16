"""T7.4 N-chunk 64-layer reference check: float INT4-dequant chain over the
full TC-token sequence (chunk inputs concatenated), vs device last-chunk
output (/tmp/chunk64mc_out_b.bin). Plus the handoff slot-placement check:
layer-3 rope'd K snapshot (/tmp/chunk64mc_kn3.bin, float M*KVW) RNE-quantized
to BF16 must equal the layer-3 cache slots [base..base+M) bytewise
(/tmp/chunk64mc_kc3.bin) — proving chunk appends land in decode_l0's slot
addressing ((t*4+kv)*256). Geometry via CHUNK_M/CHUNK_N env (default 32/2).
Writes tools/t74/report_chunk64mc.json (M=32/N=2) or the path in
CHUNK64MC_REPORT (scaling runs).
"""
import json
import os
import struct
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

M = int(os.getenv("CHUNK_M", "32"))
NCH = int(os.getenv("CHUNK_N", "2"))
H, TC = 5120, NCH * M
BASE = (NCH - 1) * M
REPORT = os.getenv("CHUNK64MC_REPORT",
                   os.path.join(REPO, "tools", "t74",
                                "report_chunk64mc.json"))


def f32_to_bf16_rne(x):
    u = struct.unpack('<I', struct.pack('<f', float(x)))[0]
    return (u + 0x7FFF + ((u >> 16) & 1)) >> 16


DPRE = os.getenv("CHUNK_DUMP_PREFIX", "/tmp/chunk64mc_in")


def main():
    t0 = time.time()
    parts = []
    for ch in range(NCH):
        for cand in (f"{DPRE}_{ch}.bin",
                     f"/tmp/chunk64mc_in_{'ab'[ch]}.bin" if ch < 2 else None):
            if cand and os.path.exists(cand):
                a = np.fromfile(cand, dtype=np.float32)
                assert a.size == M * H, (cand, a.size)
                parts.append(a)
                break
        else:
            raise SystemExit(f"no input dump for chunk {ch}")
    x = torch.from_numpy(np.concatenate(parts).reshape(1, TC, H))
    rope = Qwen3_5TextRotaryEmbedding(F.cfg)
    pos = torch.arange(TC).unsqueeze(0)
    cos, sin = rope(torch.empty(1, TC, 5120), pos)
    mask = torch.triu(torch.full((TC, TC), float("-inf")), 1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    for L in range(64):
        W = F.layer_tensors_int4(L)
        with torch.no_grad():
            x = F.manual_block(W, L, x, cos, sin,
                               mask if F.LT[L] == "full_attention" else None)
        del W
        if (L + 1) % 16 == 0:
            print(f"  ref layer {L + 1}/64 ({time.time() - t0:.0f}s)",
                  flush=True)
    ref = x.reshape(TC, H).numpy()[BASE:, :].reshape(-1)
    got = np.fromfile("/tmp/chunk64mc_out_b.bin", dtype=np.float32)
    assert got.size == ref.size, (got.size, ref.size)
    refmax = float(np.abs(ref).max())
    worst = float(np.abs(got - ref).max() / (refmax or 1))
    mean = float(np.abs(got - ref).mean() / (refmax or 1))
    ok = bool(np.isfinite(got).all() and worst <= 2e-2)
    print(f"chunk64mc ref worst-rel {worst:.2e} mean-rel {mean:.2e} "
          f"{'CHUNK64MC-OK' if ok else 'MISMATCH'}", flush=True)
    # Handoff slot check: layer-3 Kn snapshot -> BF16 vs cache slots
    # [BASE..BASE+M).
    kn = np.fromfile("/tmp/chunk64mc_kn3.bin", dtype=np.float32)
    kc = np.fromfile("/tmp/chunk64mc_kc3.bin", dtype=np.uint16)
    assert kn.size == M * 1024 and kc.size == 4 * TC * 256
    slot_ok, first_bad = True, -1
    for m in range(M):
        for i in range(1024):
            hh, d = i // 256, i % 256
            want = f32_to_bf16_rne(kn[m * 1024 + i])
            gotb = int(kc[((BASE + m) * 4 + hh) * 256 + d])
            if want != gotb:
                slot_ok, first_bad = False, (m, hh, d, want, gotb)
                break
        if not slot_ok:
            break
    print(f"slot-placement {'SLOT-OK ([%d..%d) bitwise)' % (BASE, BASE + M) if slot_ok else 'MISMATCH ' + str(first_bad)}",
          flush=True)
    rep = {"device": "B60",
           "chunk64mc": f"64-layer {NCH}-chunk prefill vs T={TC} float chain + slot check",
           "M": M, "NCH": NCH, "chunks": f"0..{TC - 1}",
           "worst_rel": worst, "mean_rel": mean, "ref_tol": 2e-2,
           "refmax": refmax, "slot_placement_bitwise": bool(slot_ok),
           "seconds": round(time.time() - t0, 1),
           "chunk64mc_ok": bool(ok and slot_ok)}
    json.dump(rep, open(REPORT, "w"), indent=1)
    print(f"WROTE {REPORT}", flush=True)
    return 0 if (ok and slot_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
