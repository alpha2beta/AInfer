"""T4.2-S3: full-question BF16 forward vs llama.cpp greedy first token.

Same raw message both sides. Ours: full prompt ids (T4.3 template) through the
manual 64-layer BF16 path; reference: llama-cli SYCL temp-0 single turn.
Metric: first-token text agreement (exact) + top-5 listing for the record.
Appends S3 into reference/fwd_T42.json.
"""
import json
import os
import subprocess
import sys
import time

import torch

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "forward"))
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
import tok as tokenizer
from fwd_cpu import (cfg, manual_block, rms1w, layer_tensors_bf16,
                     LT, BASE, REF)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from safetensors import safe_open

torch.set_num_threads(12)
tk = tokenizer.load()
msgs = [{"role": "user", "content": "What is 84 * 3 / 2?"}]
ids = tokenizer.prompt_ids(tk, msgs)
print("full prompt ids:", len(ids), ids, flush=True)

rope = Qwen3_5TextRotaryEmbedding(cfg)
pos = torch.arange(len(ids)).unsqueeze(0)
cos, sin = rope(torch.empty(1, len(ids), 5120), pos)
mask = torch.triu(torch.full((len(ids), len(ids)), float("-inf")), 1)
mask = mask.unsqueeze(0).unsqueeze(0)

wmap = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]


def load1(name):
    with safe_open(os.path.join(BASE, wmap[name]), framework="pt") as f:
        return f.get_tensor(name).to(torch.float32)


t0 = time.time()
x = load1("model.language_model.embed_tokens.weight")[torch.tensor(ids)].unsqueeze(0)
for L in range(64):
    P = f"model.language_model.layers.{L}."
    names = [P + "input_layernorm.weight", P + "post_attention_layernorm.weight",
             P + "mlp.gate_proj.weight", P + "mlp.up_proj.weight", P + "mlp.down_proj.weight"]
    if LT[L] == "full_attention":
        names += [P + f"self_attn.{p}.weight" for p in
                  ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
    else:
        names += [P + f"linear_attn.{p}.weight" for p in
                  ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                   "conv1d", "out_proj", "norm")]
        names += [P + "linear_attn.dt_bias", P + "linear_attn.A_log"]
    by_shard = {}
    for n in names:
        by_shard.setdefault(wmap[n], []).append(n)
    W = {}
    for sh, ns in by_shard.items():
        with safe_open(os.path.join(BASE, sh), framework="pt") as f:
            for n in ns:
                W[n] = f.get_tensor(n).to(torch.float32)
    with torch.no_grad():
        x = manual_block(W, L, x, cos, sin,
                         mask if LT[L] == "full_attention" else None)
    del W
    if (L + 1) % 16 == 0:
        print(f"  layer {L + 1}/64 ({time.time() - t0:.0f}s)", flush=True)
nw = load1("model.language_model.norm.weight").squeeze()
x = rms1w(x, nw)
del nw
lm = load1("lm_head.weight")
with torch.no_grad():
    logits = x @ lm.T
del lm
top5 = torch.topk(logits[0, -1], 5)
top1_text = tokenizer.decode(tk, [int(top5.indices[0])])
print("ours top5:", top5.indices.tolist(), "top1 text:", repr(top1_text), flush=True)

# llama reference, same raw message, greedy
cmd = ("/home/yanchun/llama.arc/build/bin/llama-cli -m "
       "/mnt/usb/Test/Dirk-Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 --no-warmup "
       "--no-display-prompt --simple-io --single-turn -n 64 --temp 0 -s 0")
p = subprocess.Popen(["bash", "-c",
                      "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec " + cmd],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True)
out, _ = p.communicate(input="What is 84 * 3 / 2? Reply with just the number.\n",
                       timeout=1500)
# strip thinking block + chrome (same rules as CLI)
lines, body, in_think = out.splitlines(), [], False
for ln in lines:
    s = ln.strip()
    if s == "[Start thinking]":
        in_think = True
        continue
    if s == "[End thinking]":
        in_think = False
        continue
    if in_think or not s or s.startswith(">") or s.startswith("[ Prompt:"):
        continue
    if s in ("Exiting...",) or "available commands:" in s or s.startswith("/"):
        continue
    if s.startswith(("build      :", "model      :", "ftype      :",
                     "modalities :", "Loading model")):
        continue
    if s and set(s) <= {"\u2584", "\u2588", "\u2580", " "}:
        continue
    body.append(ln)
llama_text = "\n".join(body).strip()
agree = llama_text.lstrip().startswith(top1_text.strip()) if top1_text.strip() else False
print("llama first:", repr(llama_text[:80]), "agree:", agree, flush=True)

rep = json.load(open(os.path.join(REF, "fwd_T42.json")))
rep["stages"]["S3"] = {"prompt_tokens": len(ids),
                       "seconds": round(time.time() - t0, 1),
                       "ours_top5": top5.indices.tolist(),
                       "ours_top1_text": top1_text,
                       "llama_first80": llama_text[:80],
                       "first_token_agree": bool(agree)}
json.dump(rep, open(os.path.join(REF, "fwd_T42.json"), "w"), indent=1)
print("WROTE fwd_T42.json +S3", flush=True)
