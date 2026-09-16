"""Embed one needle-corpus variant into per-chunk hidden files for chunk64mc.
Usage: needle_embed.py <variant> <out_prefix>  (M=256 fixed)
Writes <out_prefix>_<ch>.bin (256x5120 fp32) + prints code for the record.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "forward"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import fwd_cpu as F

M = 256


def main():
    vi, pre = int(sys.argv[1]), sys.argv[2]
    corp = json.load(open(os.path.join(REPO, "tools", "t74",
                                       "needle_corpus.json")))
    v = corp["variants"][vi]
    ids = v["ids"]
    assert len(ids) == 65024, len(ids)
    emb = F.load_bf16(["model.language_model.embed_tokens.weight"])[
        "model.language_model.embed_tokens.weight"]
    nch = len(ids) // M
    assert len(ids) % M == 0
    with torch.no_grad():
        for ch in range(nch):
            h = emb[torch.tensor(ids[ch * M:(ch + 1) * M])].numpy().astype(
                np.float32)
            h.tofile(f"{pre}_{ch}.bin")
            if (ch + 1) % 64 == 0:
                print(f"  chunk {ch + 1}/{nch}", flush=True)
    print(f"variant {vi} code {v['code']} span {v['needle_span']} "
          f"{nch} chunks -> {pre}_<ch>.bin", flush=True)


if __name__ == "__main__":
    main()
