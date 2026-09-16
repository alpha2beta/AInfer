"""T6.3 INT8-KV diagnostic source: capture real K/V per full layer (CPU BF16).

Runs the manual full-attention branch over a short prefill (12 ids) and
saves post-RoPE K (4x256) and raw V (4x256) for all 16 full layers, plus
per-channel absmax (K) and per-token absmax (V) stats that size the INT8
quant (K per-channel / V per-token scales).
Writes /tmp/kvcap.npz. Background job (~10 min).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "forward"))
import fwd_cpu as F2
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb)

IDS = [248045, 846, 198, 3710, 369, 279, 248046, 198, 271, 1206, 11290, 220]


def main():
    torch.set_num_threads(12)
    t_all = time.time()
    rope = Qwen3_5TextRotaryEmbedding(F2.cfg)
    T = len(IDS)
    pos = torch.arange(T).unsqueeze(0)
    cos, sin = rope(torch.empty(1, T, 5120), pos)
    mask = torch.triu(torch.full((T, T), float("-inf")), 1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    emb = F2.load_bf16(["model.language_model.embed_tokens.weight"])[
        "model.language_model.embed_tokens.weight"]
    x = emb[torch.tensor(IDS)].unsqueeze(0).to(torch.float32)
    del emb
    Ks, Vs = {}, {}
    with torch.no_grad():
        for L in range(64):
            W = F2.layer_tensors_bf16(L)
            P = f"model.language_model.layers.{L}."
            if F2.LT[L] == "full_attention":
                h = F2.rms1w(x, W[P + "input_layernorm.weight"].squeeze())
                qq = h @ W[P + "self_attn.q_proj.weight"].T
                kk = F2.rms1w(
                    (h @ W[P + "self_attn.k_proj.weight"].T).view(1, T, 4, 256),
                    W[P + "self_attn.k_norm.weight"].squeeze()).transpose(1, 2)
                vv = (h @ W[P + "self_attn.v_proj.weight"].T).view(
                    1, T, 4, 256).transpose(1, 2)
                from transformers.models.qwen3_5.modeling_qwen3_5 import (
                    apply_rotary_pos_emb as arp)
                _, kkr = arp(torch.zeros_like(kk), kk, cos, sin)
                Ks[L] = kkr.float().numpy()
                Vs[L] = vv.float().numpy()
                print(f"L{L} captured ({time.time()-t_all:.0f}s)", flush=True)
            x = F2.manual_block(W, L, x, cos, sin,
                                mask if F2.LT[L] == "full_attention" else None)
            del W
    out = {"ids": np.array(IDS)}
    for L in Ks:
        out[f"K{L}"] = Ks[L]
        out[f"V{L}"] = Vs[L]
    np.savez("/tmp/kvcap.npz", **out)
    # stats: K per-channel absmax (over t,kv), V per-token absmax
    for L in sorted(Ks):
        k = np.abs(Ks[L][0]).max(axis=(0, 1))  # (256,) per channel? (kv,T,d)
        k = np.abs(Ks[L][0])  # (4, T, 256)
        kch = k.max(axis=(0, 1))
        v = np.abs(Vs[L][0])
        vtok = v.max(axis=(0, 2))
        print(f"L{L}: Kch max {kch.max():.2f} p99 {np.quantile(kch,0.99):.2f} | "
              f"Vtok max {vtok.max():.2f} p99 {np.quantile(vtok,0.99):.2f}",
              flush=True)
    print(f"WROTE /tmp/kvcap.npz ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
