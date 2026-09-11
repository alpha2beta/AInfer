"""T1.5 box-native quality pilot: streamed-CPU BF16 greedy on one corpus prompt.

Reuses tools/forward/fwd_cpu.py (manual 64-layer wiring, one shard mapped at
a time — runs on the 30 GB box). Prefills the full plain-text prompt ids,
then decodes NEW greedy steps (one full forward per step, ~40-60 s each).
Keeps embed/lm_head/norm resident (~10 GB fp32); streams 18 layer shards.
Writes /tmp/pilot_bf16.json: {prompt_ids, generated, top1_per_step}.
Usage: pilot_greedy.py [prompt_index] [new_tokens]
"""
import json
import os
import sys
import time

import torch

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "forward"))
import fwd_cpu as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding)

CORPUS = json.load(open(os.path.join(REPO, "reference", "corpus_t15.json")))


def main():
    pi = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    new = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    prompt = CORPUS["prompts"][pi]
    ids = list(CORPUS["prompt_ids"][pi])
    print("prompt:", prompt, flush=True)
    print("ids:", ids, flush=True)
    torch.set_num_threads(12)
    t_all = time.time()
    # Resident: embed + final norm + lm_head (fp32, ~10 GB total).
    emb = F.load_bf16(["model.language_model.embed_tokens.weight"])[
        "model.language_model.embed_tokens.weight"]
    normw = F.load_bf16(["model.language_model.norm.weight"])[
        "model.language_model.norm.weight"].squeeze()
    lm = F.load_bf16(["lm_head.weight"])["lm_head.weight"]
    rope = Qwen3_5TextRotaryEmbedding(F.cfg)
    gen, tops = [], []
    with torch.no_grad():
        for step in range(new):
            t = len(ids)
            pos = torch.arange(t).unsqueeze(0)
            cos, sin = rope(torch.empty(1, t, 5120), pos)
            mask = torch.triu(torch.full((t, t), float("-inf")), 1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            x = emb[torch.tensor(ids)].unsqueeze(0).to(torch.float32)
            for L in range(64):
                W = F.layer_tensors_bf16(L)
                x = F.manual_block(W, L, x, cos, sin,
                                   mask if F.LT[L] == "full_attention" else None)
                del W
            x = F.rms1w(x, normw)
            logits = x @ lm.T
            assert torch.isfinite(logits).all()
            top5 = torch.topk(logits[0, -1], 5)
            nxt = int(top5.indices[0])
            gen.append(nxt)
            tops.append({"top5": top5.indices.tolist(),
                         "top5v": [round(float(v), 3) for v in top5.values]})
            print(f"step {step} pos {t} -> {nxt} "
                  f"top5={top5.indices.tolist()} ({time.time()-t_all:.0f}s)",
                  flush=True)
            ids.append(nxt)
            if nxt == 248046:  # EOS
                break
    out = {"prompt": prompt, "prompt_ids": CORPUS["prompt_ids"][pi],
           "generated": gen, "top_per_step": tops,
           "seconds": round(time.time() - t_all, 1)}
    json.dump(out, open("/tmp/pilot_bf16.json", "w"), indent=1)
    print("generated:", gen, flush=True)
    print("WROTE /tmp/pilot_bf16.json", flush=True)


if __name__ == "__main__":
    main()
