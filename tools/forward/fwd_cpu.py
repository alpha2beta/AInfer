"""T4.2 CPU full-forward proof: manual 64-layer wiring, streamed weights.

Paths (fp32 CPU, short prefill, causal):
  BF16: SafeTensors streamed one shard at a time (never >1 shard mapped).
  INT4: dequantized from .binfer via tools/binfer.py loader.
Stages:
  S0 layers 0-3 manual vs HF Qwen3_5DecoderLayer (one full interval).
  S1 full 64-layer BF16 forward -> logits, top-k, sanity.
  S2 full 64-layer INT4-dequant forward -> top-1 agreement vs BF16.
Writes reference/fwd_T42.json. Prompt ids from our T4.3 tokenizer.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import tok as tokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb,
    causal_conv1d_fn, torch_chunk_gated_delta_rule)
from safetensors import safe_open
import binfer as BINF

BASE = os.path.join(REPO, "models", "Qwen3.8-27B")
REF = os.path.join(REPO, "reference")
cfg = Qwen3_5TextConfig.from_dict(json.load(open(f"{BASE}/config.json"))["text_config"])
wmap = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
torch.set_num_threads(12)
LT = cfg.layer_types
NT = 4  # short-prompt prefill tokens


def rms1w(x, w, eps=1e-6):
    return x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps) * (1 + w)


def load_bf16(names):
    """names: list of full tensor names. Returns {short: fp32}. One shard at a time."""
    by_shard = {}
    for n in names:
        by_shard.setdefault(wmap[n], []).append(n)
    out = {}
    for sh, ns in by_shard.items():
        with safe_open(os.path.join(BASE, sh), framework="pt") as f:
            for n in ns:
                out[n] = f.get_tensor(n).to(torch.float32)
    return out


def layer_tensors_bf16(L):
    P = f"model.language_model.layers.{L}."
    if LT[L] == "full_attention":
        names = [P + s for s in ("input_layernorm.weight",
                 "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                 "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                 "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                 "post_attention_layernorm.weight",
                 "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")]
    else:
        names = [P + s for s in ("input_layernorm.weight",
                 "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                 "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
                 "linear_attn.conv1d.weight", "linear_attn.dt_bias",
                 "linear_attn.A_log", "linear_attn.norm.weight",
                 "linear_attn.out_proj.weight",
                 "post_attention_layernorm.weight",
                 "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")]
    return load_bf16(names)


def layer_tensors_int4(L):
    """Same weights dequantized from .binfer (layout-0 INT4 + BF16 scales)."""
    P = f"model.language_model.layers.{L}."
    out = {}
    for short in ("input_layernorm.weight", "self_attn.q_proj.weight",
                  "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                  "self_attn.o_proj.weight", "self_attn.q_norm.weight",
                  "self_attn.k_norm.weight", "post_attention_layernorm.weight",
                  "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                  "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
                  "linear_attn.conv1d.weight", "linear_attn.dt_bias",
                  "linear_attn.A_log", "linear_attn.norm.weight",
                  "linear_attn.out_proj.weight",
                  "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
        name = P + short
        try:
            out[name] = torch.from_numpy(BINF.load_tensor_from_binfer(BINF.OUT_FILE, name))
        except KeyError:
            pass
    return out


def manual_block(W, L, x, cos, sin, mask):
    """W: {full name: tensor}. Manual low-level wiring. Returns block output."""
    P = f"model.language_model.layers.{L}."
    B, T, _ = x.shape
    h = rms1w(x, W[P + "input_layernorm.weight"].squeeze())
    if LT[L] == "full_attention":
        nq = 24
        qkv = h @ W[P + "self_attn.q_proj.weight"].T
        qq, gate = torch.chunk(qkv.view(B, T, nq, 512), 2, dim=-1)
        qq = rms1w(qq.reshape(B, T, nq, 256), W[P + "self_attn.q_norm.weight"].squeeze()).transpose(1, 2)
        kk = rms1w((h @ W[P + "self_attn.k_proj.weight"].T).view(B, T, 4, 256),
                   W[P + "self_attn.k_norm.weight"].squeeze()).transpose(1, 2)
        vv = (h @ W[P + "self_attn.v_proj.weight"].T).view(B, T, 4, 256).transpose(1, 2)
        qq, kk = apply_rotary_pos_emb(qq, kk, cos, sin)
        kk = kk.repeat_interleave(6, dim=1)
        vv = vv.repeat_interleave(6, dim=1)
        sc = qq @ kk.transpose(-1, -2) / 16.0 + mask
        wts = torch.softmax(sc.float(), dim=-1).type_as(qq)
        ao = (wts @ vv).transpose(1, 2).reshape(B, T, -1)
        ao = ao * torch.sigmoid(gate.reshape(B, T, -1))
        mix = ao @ W[P + "self_attn.o_proj.weight"].T
    else:
        kd, vd = 2048, 6144
        nk, nhv, hkd, hvd = 16, 48, 128, 128
        mixed = (h @ W[P + "linear_attn.in_proj_qkv.weight"].T).transpose(1, 2)
        z = (h @ W[P + "linear_attn.in_proj_z.weight"].T).reshape(B, T, -1, hvd)
        b = h @ W[P + "linear_attn.in_proj_b.weight"].T
        a_ = h @ W[P + "linear_attn.in_proj_a.weight"].T
        mixed = causal_conv1d_fn(mixed, W[P + "linear_attn.conv1d.weight"].squeeze(1),
                                 None, activation="silu").transpose(1, 2)
        q, k, v = torch.split(mixed, [kd, kd, vd], dim=-1)
        q = q.reshape(B, T, -1, hkd)
        k = k.reshape(B, T, -1, hkd)
        v = v.reshape(B, T, -1, hvd)
        beta = b.sigmoid()
        g = -W[P + "linear_attn.A_log"].float().exp() * F.softplus(
            a_.float() + W[P + "linear_attn.dt_bias"])
        q = q.repeat_interleave(3, dim=2)
        k = k.repeat_interleave(3, dim=2)
        core, _ = torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True)
        core = core.reshape(-1, hvd).float()
        zz = z.reshape(-1, hvd).float()
        var = core.pow(2).mean(-1, keepdim=True)
        core = W[P + "linear_attn.norm.weight"] * core * torch.rsqrt(var + 1e-6)
        core = core * F.silu(zz)
        mix = core.reshape(B, T, -1) @ W[P + "linear_attn.out_proj.weight"].T
    h2 = x + mix
    h3 = rms1w(h2, W[P + "post_attention_layernorm.weight"].squeeze())
    mlp = (F.silu(h3 @ W[P + "mlp.gate_proj.weight"].T) *
           (h3 @ W[P + "mlp.up_proj.weight"].T)) @ W[P + "mlp.down_proj.weight"].T
    return h2 + mlp


def full_forward(ids, cos, sin, mask, load_weights, tag):
    """Run all 64 layers; load_weights(L) -> {name: tensor}. Returns logits."""
    t0 = time.time()
    x = load_weights("embed")[torch.tensor(ids)].unsqueeze(0).to(torch.float32)
    for L in range(64):
        W = load_weights(L)
        with torch.no_grad():
            x = manual_block(W, L, x, cos, sin,
                             mask if LT[L] == "full_attention" else None)
        del W
        if (L + 1) % 16 == 0:
            print(f"  {tag} layer {L + 1}/64 ({time.time() - t0:.0f}s)", flush=True)
    nw = load_weights("norm")
    x = rms1w(x, nw)
    del nw
    lm = load_weights("lmhead")
    with torch.no_grad():
        logits = x @ lm.T.to(torch.float32)
    del lm
    return logits


def main():
    import sys as _sys
    only = set(_sys.argv[1:]) or {"S0", "S1", "S2", "S3"}
    t_all = time.time()
    rep = {}
    if os.path.exists(os.path.join(REF, "fwd_T42.json")):
        try:
            rep = json.load(open(os.path.join(REF, "fwd_T42.json")))
            rep = {"stages": rep.get("stages", {})}
        except Exception:
            rep = {"stages": {}}
    else:
        rep = {"stages": {}}
    # prompt ids from our T4.3 tokenizer
    tk = tokenizer.load()
    msgs = [{"role": "user", "content": "What is 84 * 3 / 2?"}]
    ids = tokenizer.prompt_ids(tk, msgs)[:NT]
    full_ids = tokenizer.prompt_ids(tk, msgs)
    print("prompt ids:", ids, f"(full {len(full_ids)})", flush=True)
    rope = Qwen3_5TextRotaryEmbedding(cfg)
    pos = torch.arange(len(ids)).unsqueeze(0)
    cos, sin = rope(torch.empty(1, len(ids), 5120), pos)
    mask = torch.triu(torch.full((len(ids), len(ids)), float("-inf")), 1)
    mask = mask.unsqueeze(0).unsqueeze(0)

    # S0: layers 0-3 manual vs HF decoders
    print("== S0: interval proof ==", flush=True)
    s0 = {}
    for L in range(4):
        P = f"model.language_model.layers.{L}."
        W = layer_tensors_bf16(L)
        dec = Qwen3_5DecoderLayer(cfg, L)
        dec.input_layernorm.weight.data = W[P + "input_layernorm.weight"].squeeze()
        dec.post_attention_layernorm.weight.data = W[P + "post_attention_layernorm.weight"].squeeze()
        for pr in ("gate_proj", "up_proj", "down_proj"):
            getattr(dec.mlp, pr).weight.data = W[P + f"mlp.{pr}.weight"]
        if LT[L] == "full_attention":
            a = dec.self_attn
            for pr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(a, pr).weight.data = W[P + f"self_attn.{pr}.weight"]
            a.q_norm.weight.data = W[P + "self_attn.q_norm.weight"].squeeze()
            a.k_norm.weight.data = W[P + "self_attn.k_norm.weight"].squeeze()
        else:
            m = dec.linear_attn
            for pr in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                       "conv1d", "out_proj"):
                getattr(m, pr).weight.data = W[P + f"linear_attn.{pr}.weight"]
            m.dt_bias.data = W[P + "linear_attn.dt_bias"].squeeze()
            m.A_log.data = W[P + "linear_attn.A_log"].squeeze()
            m.norm.weight.data = W[P + "linear_attn.norm.weight"].squeeze()
        dec.eval()
        torch.manual_seed(7)
        x = torch.randn(1, len(ids), cfg.hidden_size)
        with torch.no_grad():
            m = mask if LT[L] == "full_attention" else None
            ref = dec(x, (cos, sin), m)
            got = manual_block(W, L, x, cos, sin, m)
        d = (got - ref).abs()
        s0[f"L{L}-{LT[L]}"] = {"max": float(d.max()), "mean": float(d.mean())}
        print(f"L{L} {LT[L]}: maxdiff={d.max():.2e}", flush=True)
        del W, dec
    rep["stages"]["S0"] = s0

    # S1: full BF16 forward
    print("== S1: full BF16 forward ==", flush=True)
    t0 = time.time()
    emb = load_bf16(["model.language_model.embed_tokens.weight"])[
        "model.language_model.embed_tokens.weight"]
    x = emb[torch.tensor(ids)].unsqueeze(0)
    del emb
    for L in range(64):
        W = layer_tensors_bf16(L)
        with torch.no_grad():
            x = manual_block(W, L, x, cos, sin,
                             mask if LT[L] == "full_attention" else None)
        del W
        if (L + 1) % 16 == 0:
            print(f"  layer {L + 1}/64 ({time.time() - t0:.0f}s)", flush=True)
    normw = load_bf16(["model.language_model.norm.weight"])[
        "model.language_model.norm.weight"].squeeze()
    x = rms1w(x, normw)
    del normw
    lm = load_bf16(["lm_head.weight"])["lm_head.weight"]
    with torch.no_grad():
        logits = x @ lm.T
    del lm
    assert torch.isfinite(logits).all()
    top5 = torch.topk(logits[0, -1], 5)
    rep["stages"]["S1"] = {"seconds": round(time.time() - t0, 1),
                           "top5": top5.indices.tolist(),
                           "top5_logit": [round(float(v), 3) for v in top5.values],
                           "logit_std": round(float(logits.std()), 4),
                           "logits_sha": None}
    print("S1 top5:", top5.indices.tolist(), flush=True)
    torch.save(logits, os.path.join(REF, "fwd_T42_bf16_logits.pt"))
    rep["stages"]["S1"]["note"] = "logits saved to fwd_T42_bf16_logits.pt"

    # S2: full INT4-dequant forward
    print("== S2: full INT4 forward ==", flush=True)
    t0 = time.time()
    emb = torch.from_numpy(BINF.load_tensor_from_binfer(
        BINF.OUT_FILE, "model.language_model.embed_tokens.weight"))
    x = emb[torch.tensor(ids)].unsqueeze(0).to(torch.float32)
    del emb
    for L in range(64):
        W = layer_tensors_int4(L)
        with torch.no_grad():
            x = manual_block(W, L, x, cos, sin,
                             mask if LT[L] == "full_attention" else None)
        del W
        if (L + 1) % 16 == 0:
            print(f"  layer {L + 1}/64 ({time.time() - t0:.0f}s)", flush=True)
    normw = torch.from_numpy(BINF.load_tensor_from_binfer(
        BINF.OUT_FILE, "model.language_model.norm.weight")).squeeze().to(torch.float32)
    x = rms1w(x, normw)
    del normw
    lm = torch.from_numpy(BINF.load_tensor_from_binfer(BINF.OUT_FILE, "lm_head.weight"))
    with torch.no_grad():
        logits_q = x @ lm.T.to(torch.float32)
    del lm
    bf = torch.load(os.path.join(REF, "fwd_T42_bf16_logits.pt"))[0]
    logits_q = logits_q[0]
    agree = [(bf[i].argmax() == logits_q[i].argmax()) for i in range(len(ids))]
    rep["stages"]["S2"] = {"seconds": round(time.time() - t0, 1),
                           "top1_agree": [bool(a) for a in agree],
                           "top1_q": [int(logits_q[i].argmax()) for i in range(len(ids))],
                           "top1_bf16": [int(bf[i].argmax()) for i in range(len(ids))]}
    print("S2 top1 agree:", [bool(a) for a in agree], flush=True)
    rep["seconds_total"] = round(time.time() - t_all, 1)
    json.dump(rep, open(os.path.join(REF, "fwd_T42.json"), "w"), indent=1)
    print("WROTE fwd_T42.json", flush=True)


if __name__ == "__main__":
    main()
