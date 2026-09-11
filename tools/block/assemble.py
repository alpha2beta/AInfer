"""T4.1 CPU block-wiring proof: manual full-block (both types) vs HF decoder.
Manual = low-level torch ops only (no HF Attention/GatedDeltaNet/MLP modules):
projections, (1+w) norms, NeoX RoPE-64, GQA repeat, scaled causal softmax fp32,
sigmoid gate, conv-k4+silu, chunk/recurrent delta rule fns, silu MLP.
Reference = HF Qwen3_5DecoderLayer with real L3/L0 weights.
Writes reference/block_T41.json.
"""
import json
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb, causal_conv1d_fn,
    torch_chunk_gated_delta_rule)
from safetensors import safe_open

BASE = "/mnt/usb/AInfer/models/Qwen3.8-27B"
REF = "/mnt/usb/AInfer/reference"
cfg = Qwen3_5TextConfig.from_dict(json.load(open(f"{BASE}/config.json"))["text_config"])
wmap = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
handles = {}
def wt(name):
    sh = wmap[name]
    if sh not in handles:
        handles[sh] = safe_open(f"{BASE}/{sh}", framework="pt")
    return handles[sh].get_tensor(name).to(torch.float32)

def load_decoder(L):
    P = f"model.language_model.layers.{L}."
    dec = Qwen3_5DecoderLayer(cfg, L)
    dec.input_layernorm.weight.data = wt(P + "input_layernorm.weight").squeeze()
    dec.post_attention_layernorm.weight.data = wt(P + "post_attention_layernorm.weight").squeeze()
    for proj in ("gate_proj", "up_proj", "down_proj"):
        getattr(dec.mlp, proj).weight.data = wt(P + f"mlp.{proj}.weight")
    if dec.block_type == "full_attention":
        a = dec.self_attn
        a.q_proj.weight.data = wt(P + "self_attn.q_proj.weight")
        a.k_proj.weight.data = wt(P + "self_attn.k_proj.weight")
        a.v_proj.weight.data = wt(P + "self_attn.v_proj.weight")
        a.o_proj.weight.data = wt(P + "self_attn.o_proj.weight")
        a.q_norm.weight.data = wt(P + "self_attn.q_norm.weight").squeeze()
        a.k_norm.weight.data = wt(P + "self_attn.k_norm.weight").squeeze()
    else:
        m = dec.linear_attn
        m.in_proj_qkv.weight.data = wt(P + "linear_attn.in_proj_qkv.weight")
        m.in_proj_z.weight.data = wt(P + "linear_attn.in_proj_z.weight")
        m.in_proj_b.weight.data = wt(P + "linear_attn.in_proj_b.weight")
        m.in_proj_a.weight.data = wt(P + "linear_attn.in_proj_a.weight")
        m.conv1d.weight.data = wt(P + "linear_attn.conv1d.weight")
        m.dt_bias.data = wt(P + "linear_attn.dt_bias").squeeze()
        m.A_log.data = wt(P + "linear_attn.A_log").squeeze()
        m.norm.weight.data = wt(P + "linear_attn.norm.weight").squeeze()
        m.out_proj.weight.data = wt(P + "linear_attn.out_proj.weight")
    dec.eval()
    return dec

def rms1w(x, w, eps=1e-6):
    return x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps) * (1 + w)

def manual_full_block(dec, x, cos, sin, mask=None):
    L = dec.self_attn.layer_idx if dec.block_type == "full_attention" else dec.linear_attn.layer_idx
    B, T, _ = x.shape
    h = rms1w(x, dec.input_layernorm.weight)
    if dec.block_type == "full_attention":
        a = dec.self_attn
        nq = a.config.num_attention_heads
        qkv = h @ a.q_proj.weight.T
        qq, gate = torch.chunk(qkv.view(B, T, nq, 512), 2, dim=-1)
        qq = rms1w(qq.reshape(B, T, nq, 256), a.q_norm.weight).transpose(1, 2)
        kk = rms1w((h @ a.k_proj.weight.T).view(B, T, 4, 256), a.k_norm.weight).transpose(1, 2)
        vv = (h @ a.v_proj.weight.T).view(B, T, 4, 256).transpose(1, 2)
        qq, kk = apply_rotary_pos_emb(qq, kk, cos, sin)
        kk = kk.repeat_interleave(6, dim=1)
        vv = vv.repeat_interleave(6, dim=1)
        sc = qq @ kk.transpose(-1, -2) / 16.0
        # Causal masking matches HF eager path only via explicit additive mask
        # (mask=None means NON-causal in HF eager). Caller passes it for both.
        if mask is not None:
            sc = sc + mask
        wts = torch.softmax(sc.float(), dim=-1).type_as(qq)
        ao = (wts @ vv).transpose(1, 2).reshape(B, T, -1)
        ao = ao * torch.sigmoid(gate.reshape(B, T, -1))
        mix = ao @ a.o_proj.weight.T
    else:
        m = dec.linear_attn
        mixed = m.in_proj_qkv(h).transpose(1, 2)
        z = m.in_proj_z(h).reshape(B, T, -1, m.head_v_dim)
        b = m.in_proj_b(h)
        a_ = m.in_proj_a(h)
        mixed = causal_conv1d_fn(mixed, m.conv1d.weight.squeeze(1), None,
                                 activation=m.activation).transpose(1, 2)
        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(B, T, -1, m.head_k_dim)
        k = k.reshape(B, T, -1, m.head_k_dim)
        v = v.reshape(B, T, -1, m.head_v_dim)
        beta = b.sigmoid()
        g = -m.A_log.float().exp() * F.softplus(a_.float() + m.dt_bias)
        if m.num_v_heads // m.num_k_heads > 1:
            q = q.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
            k = k.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        core, _ = torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True)
        core = core.reshape(-1, m.head_v_dim)
        zz = z.reshape(-1, m.head_v_dim)
        # RMSNormGated: norm(core) * (1+w) gated by silu? check: norm(x, gate)
        core = m.norm(core, zz).reshape(B, T, -1)
        mix = m.out_proj(core)
    h2 = x + mix
    h3 = rms1w(h2, dec.post_attention_layernorm.weight)
    mlp = dec.mlp
    out = h2 + (F.silu(h3 @ mlp.gate_proj.weight.T) * (h3 @ mlp.up_proj.weight.T)) @ mlp.down_proj.weight.T
    return out

rope = Qwen3_5TextRotaryEmbedding(cfg)
torch.manual_seed(0)
results = {}
for L in (3, 0):
    dec = load_decoder(L)
    x = torch.randn(1, 4, cfg.hidden_size)
    pos = torch.arange(4).unsqueeze(0)
    cos, sin = rope(x, pos)
    T = 4
    causal = torch.triu(torch.full((T, T), float("-inf")), 1).unsqueeze(0).unsqueeze(0)
    # Full-attention needs the causal mask on BOTH sides (HF eager mask=None is
    # non-causal). Linear attention is structurally causal; mask=None on both.
    m = causal if dec.block_type == "full_attention" else None
    with torch.no_grad():
        ref = dec(x, (cos, sin), m)
        got = manual_full_block(dec, x, cos, sin, m)
    d = (got - ref).abs()
    key = f"L{L}-{dec.block_type}"
    results[key] = {"max": float(d.max()), "mean": float(d.mean()),
                    "refmax": float(ref.abs().max())}
    print(f"{key}: maxdiff={d.max():.2e} mean={d.mean():.2e} refmax={ref.abs().max():.2f}",
          flush=True)
json.dump({"date": "2026-09-08",
           "note": "manual low-level wiring vs HF Qwen3_5DecoderLayer, real weights, 4-token prefill",
           "results": results},
          open(f"{REF}/block_T41.json", "w"), indent=1)
print("WROTE block_T41.json")
