"""T1.4 logits batch: short-prompt BF16 logits (+ INT4 top-1 agreement) for all
6 staged corpus prompts (reference/greedy_prompts_t14.json).

Reuses fwd_cpu.full_forward (streamed one-shard-at-a-time, box-native).
Logits only — no multi-step CPU greedy (35 s/token x tokens x prompts is
hours; greedy agreement already rides on the recorded loop via T1.5).
Writes reference/fwd_T14_batch.json + per-prompt last-position logits .pt.
"""
import json
import os
import sys
import time

import torch

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "forward"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import fwd_cpu as F
import binfer as BINF
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

REF = os.path.join(REPO, "reference")
BASE = os.path.join(REPO, "models", "Qwen3.8-27B")


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


def int4_load(tag):
    if tag == "embed":
        return torch.from_numpy(BINF.load_tensor_from_binfer(
            BINF.OUT_FILE, "model.language_model.embed_tokens.weight"))
    if tag == "norm":
        return torch.from_numpy(BINF.load_tensor_from_binfer(
            BINF.OUT_FILE, "model.language_model.norm.weight")).squeeze()
    if tag == "lmhead":
        return torch.from_numpy(BINF.load_tensor_from_binfer(
            BINF.OUT_FILE, "lm_head.weight"))
    return F.layer_tensors_int4(tag)


def main():
    staged = json.load(open(os.path.join(REF, "greedy_prompts_t14.json")))
    prompts, pidsets = staged["prompts"], staged["prompt_ids"]
    assert len(prompts) == 6 and len(pidsets) == 6
    rope = Qwen3_5TextRotaryEmbedding(F.cfg)
    out = {"task": "T1.4-logits-batch", "prompts": []}
    for pi, (text, ids) in enumerate(zip(prompts, pidsets)):
        t0 = time.time()
        pos = torch.arange(len(ids)).unsqueeze(0)
        cos, sin = rope(torch.empty(1, len(ids), 5120), pos)
        mask = torch.triu(torch.full((len(ids), len(ids)), float("-inf")), 1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            lb = F.full_forward(ids, cos, sin, mask, bf16_load, f"P{pi}-bf16")[0]
            lq = F.full_forward(ids, cos, sin, mask, int4_load, f"P{pi}-int4")[0]
        assert torch.isfinite(lb).all() and torch.isfinite(lq).all()
        agree = [bool(lb[i].argmax() == lq[i].argmax()) for i in range(len(ids))]
        top5 = torch.topk(lb[-1], 5)
        torch.save(lb[-1], os.path.join(REF, f"fwd_T14_p{pi}_last.pt"))
        out["prompts"].append({
            "prompt": text, "ids": ids,
            "seconds": round(time.time() - t0, 1),
            "top5_last": top5.indices.tolist(),
            "top5v_last": [round(float(v), 3) for v in top5.values],
            "logit_std": round(float(lb.std()), 4),
            "int4_top1_agree": agree,
        })
        print(f"P{pi} done ({time.time() - t0:.0f}s) agree={agree}", flush=True)
        json.dump(out, open(os.path.join(REF, "fwd_T14_batch.json"), "w"), indent=1)
    print("WROTE fwd_T14_batch.json", flush=True)


if __name__ == "__main__":
    main()
