#!/usr/bin/env python3
"""T8.6 BF16 teacher-forced replay along the INT4 greedy trajectory.

For one case: ids = prompt_ids + int4_generated; streamed one-shard-at-a-time
BF16 full_forward (reuses tools/forward/fwd_cpu, box-native). Records per
generated position {pos, int4_tok, bf_top[10], bf_topv[10]} for margin/
overlap/first-divergence analysis. ~10 min/case (USB streaming-bound).
Usage: bf16_replay.py [--case ID]  (writes runs/bf16/<id>.json)
"""
import json
import os
import sys
import time

import torch

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "forward"))
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import fwd_cpu as F
import tok as tokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

HERE = os.path.dirname(os.path.abspath(__file__))


def bf16_load(tag):
    if tag == "embed":
        return F.load_bf16(["model.language_model.embed_tokens.weight"])[
            "model.language_model.embed_tokens.weight"]
    if tag == "norm":
        return F.load_bf16(["model.language_model.norm.weight"])[
            "model.language_model.norm.weight"].squeeze()
    if tag == "lmhead":
        return F.load_bf16(["lm_head.weight"])["lm_head.weight"]
    return F.layer_tensors_bf16(tag)


def main():
    only = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else None
    corp = {c["id"]: c for c in json.load(
        open(os.path.join(HERE, "corpus_t85.json")))["cases"]}
    tk = tokenizer.load()
    rope = Qwen3_5TextRotaryEmbedding(F.cfg)
    for cid, case in corp.items():
        if only and cid != only:
            continue
        if case["category"] == "retrieval":
            continue
        out = os.path.join(HERE, "runs", "bf16", cid + ".json")
        if os.path.exists(out):
            continue
        rep = os.path.join(HERE, "runs", "int4", cid + ".json")
        if not os.path.exists(rep):
            print(f"{cid}: no int4 report, skip", flush=True)
            continue
        gen = json.load(open(rep)).get("generated", [])
        if not gen:
            continue
        rendered = tokenizer.render_chat(
            [{"role": "user", "content": case["prompt"]}],
            add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer.encode(tk, rendered)
        ids = prompt_ids + [t for t in gen if t not in (248044, 248046)]
        t0 = time.time()
        pos = torch.arange(len(ids)).unsqueeze(0)
        cos, sin = rope(torch.empty(1, len(ids), 5120), pos)
        mask = torch.triu(torch.full((len(ids), len(ids)), float("-inf")), 1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            lb = F.full_forward(ids, cos, sin, mask, bf16_load, f"{cid}-bf16")[0]
        assert torch.isfinite(lb).all()
        recs = []
        for j, tok_id in enumerate(gen):
            # lb[i] predicts token i+1; gen[j] sits at seq index
            # len(prompt)+j, so its predictor is lb[len(prompt)+j-1].
            i = len(prompt_ids) + j - 1
            if i < 0 or i >= len(lb):
                break
            top = torch.topk(lb[i], 10)
            recs.append({"pos": j, "int4_tok": tok_id,
                         "bf_top": top.indices.tolist(),
                         "bf_topv": [round(float(v), 4) for v in top.values]})
        os.makedirs(os.path.join(HERE, "runs", "bf16"), exist_ok=True)
        json.dump({"id": cid, "n_prompt": len(prompt_ids), "positions": recs},
                  open(out, "w"))
        print(f"{cid}: {len(recs)} positions ({time.time()-t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
