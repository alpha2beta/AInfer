"""T7.2 MTP acceptance measurement (background CPU job).

Protocol (speculative-decoding truth): for each corpus prompt, run the TRUE
trunk prefill (BF16 streamed, manual wiring) capturing logits AND final hidden
states; run the TRUE MTP-1 prefill (fc fusion + 1 full decoder layer, causal,
same RoPE) over shifted inputs; draft[t] = argmax(MTP out at t) is the MTP
candidate for position t+1... precisely: MTP(embed[ids[t]], h[t]) predicts
t+1... verified against trunk greedy argmax(logits[t]) at the same position.
Acceptance alpha = fraction of positions where draft == trunk greedy.

Dataflow (from vLLM qwen3_next_mtp; HF has no MTP forward):
  e = rms1w(embed(ids), pre_fc_norm_embedding)
  h = rms1w(hidden, pre_fc_norm_hidden)
  x = cat([e, h]) @ fc.T ; 1x full decoder layer (own causal context) ;
  mtp.norm ; shared lm_head.
Prompts truncated to 8 tokens (6+ acceptance positions each -> ~40 samples).
Writes tools/t72/report_accept.json.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "forward"))
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
from fwd_cpu import (LT, NT, cfg, load_bf16, manual_block, rms1w,
                     layer_tensors_bf16)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding)

BASE = os.path.join(REPO, "models", "Qwen3.8-27B")
torch.set_num_threads(12)

MTP = "mtp."
ML = "mtp.layers.0."


def mtp_tensors():
    W = load_bf16([
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.language_model.norm.weight",
        MTP + "fc.weight",
        MTP + "norm.weight",
        MTP + "pre_fc_norm_embedding.weight",
        MTP + "pre_fc_norm_hidden.weight",
        ML + "input_layernorm.weight",
        ML + "self_attn.q_proj.weight",
        ML + "self_attn.k_proj.weight",
        ML + "self_attn.v_proj.weight",
        ML + "self_attn.o_proj.weight",
        ML + "self_attn.q_norm.weight",
        ML + "self_attn.k_norm.weight",
        ML + "post_attention_layernorm.weight",
        ML + "mlp.gate_proj.weight",
        ML + "mlp.up_proj.weight",
        ML + "mlp.down_proj.weight",
    ])
    return W


def mtp_layer(W, x, cos, sin, mask):
    """One full decoder layer over (B, T, 5120); returns output. Mirrors the
    full_attention branch of manual_block with MTP weights."""
    B, T, _ = x.shape
    nq = 24
    h = rms1w(x, W[ML + "input_layernorm.weight"].squeeze())
    qkv = h @ W[ML + "self_attn.q_proj.weight"].T
    qq, gate = torch.chunk(qkv.view(B, T, nq, 512), 2, dim=-1)
    qq = rms1w(qq.reshape(B, T, nq, 256),
               W[ML + "self_attn.q_norm.weight"].squeeze()).transpose(1, 2)
    kk = rms1w((h @ W[ML + "self_attn.k_proj.weight"].T).view(B, T, 4, 256),
               W[ML + "self_attn.k_norm.weight"].squeeze()).transpose(1, 2)
    vv = (h @ W[ML + "self_attn.v_proj.weight"].T).view(B, T, 4, 256).transpose(1, 2)
    # rope halves live in modeling_qwen3_5.apply_rotary_pos_emb
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_rotary_pos_emb)
    qq, kk = apply_rotary_pos_emb(qq, kk, cos, sin)
    kk = kk.repeat_interleave(6, dim=1)
    vv = vv.repeat_interleave(6, dim=1)
    sc = qq @ kk.transpose(-1, -2) / 16.0 + mask
    wts = torch.softmax(sc.float(), dim=-1).type_as(qq)
    ao = (wts @ vv).transpose(1, 2).reshape(B, T, -1)
    ao = ao * torch.sigmoid(gate.reshape(B, T, -1))
    mix = ao @ W[ML + "self_attn.o_proj.weight"].T
    h2 = x + mix
    h3 = rms1w(h2, W[ML + "post_attention_layernorm.weight"].squeeze())
    mlp = (F.silu(h3 @ W[ML + "mlp.gate_proj.weight"].T) *
           (h3 @ W[ML + "mlp.up_proj.weight"].T)) @ W[ML + "mlp.down_proj.weight"].T
    return h2 + mlp


def trunk_prefill(ids, cos, sin, mask):
    """Full 64-layer BF16 trunk forward. Returns (logits, hidden pre-norm)."""
    t0 = time.time()
    emb = load_bf16(["model.language_model.embed_tokens.weight"])[
        "model.language_model.embed_tokens.weight"]
    x = emb[torch.tensor(ids)].unsqueeze(0).to(torch.float32)
    del emb
    for L in range(64):
        W = layer_tensors_bf16(L)
        with torch.no_grad():
            x = manual_block(W, L, x, cos, sin,
                             mask if LT[L] == "full_attention" else None)
        del W
        if (L + 1) % 16 == 0:
            print(f"  trunk layer {L + 1}/64 ({time.time() - t0:.0f}s)",
                  flush=True)
    hidden = x.clone()
    normw = load_bf16(["model.language_model.norm.weight"])[
        "model.language_model.norm.weight"].squeeze()
    x = rms1w(x, normw)
    del normw
    lm = load_bf16(["lm_head.weight"])["lm_head.weight"]
    with torch.no_grad():
        logits = x @ lm.T.to(torch.float32)
    del lm
    return logits, hidden


def main():
    out_path = os.path.join(REPO, "tools", "t72", "report_accept.json")
    corpus = json.load(open(os.path.join(REPO, "reference", "corpus_t15.json")))
    rope = Qwen3_5TextRotaryEmbedding(cfg)
    MW = mtp_tensors()
    print("MTP weights loaded", flush=True)
    samples = []
    for pi, pids in enumerate(corpus["prompt_ids"]):
        ids = pids[:8]
        T = len(ids)
        pos = torch.arange(T).unsqueeze(0)
        cos, sin = rope(torch.empty(1, T, 5120), pos)
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        print(f"== prompt {pi}: {T} tokens ==", flush=True)
        with torch.no_grad():
            logits, hidden = trunk_prefill(ids, cos, sin, mask)
        # MTP prefill over shifted inputs: (embed[t], h[t]) -> draft for t+1.
        # Valid for t in 0..T-2 (verify needs trunk logits at t+1... precisely
        # draft[t] predicts position t+1; verify = argmax(logits[t])).
        # NOTE: MTP(embed[ids[t]], h[t]) with causal context predicts t+1.
        emb_t = MW["model.language_model.embed_tokens.weight"][
            torch.tensor(ids)].unsqueeze(0)
        e = rms1w(emb_t, MW[MTP + "pre_fc_norm_embedding.weight"].squeeze())
        hh = rms1w(hidden, MW[MTP + "pre_fc_norm_hidden.weight"].squeeze())
        x = torch.cat([e, hh], dim=-1) @ MW[MTP + "fc.weight"].T
        with torch.no_grad():
            y = mtp_layer(MW, x, cos, sin, mask)
            y = rms1w(y, MW[MTP + "norm.weight"].squeeze())
            dl = y @ MW["lm_head.weight"].T.to(torch.float32)
        for t in range(T - 1):
            draft = int(dl[0, t].argmax())
            truth = int(logits[0, t].argmax())
            samples.append({"prompt": pi, "pos": t + 1, "draft": draft,
                            "truth": truth, "agree": draft == truth})
        n_ag = sum(1 for s in samples if s["prompt"] == pi and s["agree"])
        print(f"prompt {pi}: agree {n_ag}/{T - 1}", flush=True)
        del logits, hidden
    alpha = sum(1 for s in samples if s["agree"]) / max(1, len(samples))
    json.dump({"protocol": "MTP(embed[t],h[t])->t+1 vs trunk argmax",
               "samples": len(samples), "accepted": sum(1 for s in samples
                                                        if s["agree"]),
               "alpha": alpha, "detail": samples},
              open(out_path, "w"), indent=1)
    print(f"ALPHA={alpha:.3f} n={len(samples)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
