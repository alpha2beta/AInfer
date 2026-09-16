"""Prefill arbitration reference: T=64 INT4-dequant chain over the real
embedded 70-prompt ids (first 64), BF16 lm_head top-1 per position.
Compared against device chunked-prefill + BF16 lm_head top-1
(/tmp/arb_top1.json) — the delta isolates fp16-chunk-path noise.
Writes /tmp/arb_ref.json.
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
from prefill_arb import IDS70
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding)

M, TC = 32, 64


def main():
    t0 = time.time()
    emb = F.load_bf16(["model.language_model.embed_tokens.weight"])[
        "model.language_model.embed_tokens.weight"]
    x = emb[torch.tensor(IDS70[:TC])].unsqueeze(0).to(torch.float32)
    del emb
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
            print(f"  arb layer {L + 1}/64 ({time.time() - t0:.0f}s)",
                  flush=True)
    lm = F.load_bf16(["lm_head.weight"])["lm_head.weight"]
    with torch.no_grad():
        logits = (x[0].to(torch.float32) @ lm.T.to(torch.float32)).numpy()
    top1 = logits.argmax(axis=1).tolist()
    top5v = np.sort(logits, axis=1)[:, ::-1][:, :2]
    margins = (top5v[:, 0] - top5v[:, 1]).tolist()
    json.dump({"top1": top1, "top1_margin": [round(float(m), 4) for m in margins],
               "seconds": round(time.time() - t0, 1)},
              open("/tmp/arb_ref.json", "w"))
    print("WROTE arb_ref.json", flush=True)


if __name__ == "__main__":
    main()
