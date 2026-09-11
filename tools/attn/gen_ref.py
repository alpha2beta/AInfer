"""Real HF single-block references for a full-attention layer (T1.4/T4.1 targets).
Loads ONLY layer-3 (full_attention) weights + norms + MLP into HF modules on CPU.
Also recomputes layer-0 MLP with the correct (1+w) RMSNorm.
Writes reference/attn_block_L3.json and fixes reference/mlp_layer0.json.
"""
import json, os
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention, Qwen3_5MLP, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding)
from safetensors import safe_open

BASE = "/mnt/usb/AInfer/models/Qwen3.8-27B"
REF = "/mnt/usb/AInfer/reference"
cfg = Qwen3_5TextConfig.from_dict(json.load(open(f"{BASE}/config.json"))["text_config"])
print("layers:", cfg.num_hidden_layers, "heads:", cfg.num_attention_heads,
      "kv:", cfg.num_key_value_heads, "headdim:", cfg.head_dim)

# shard map
wmap = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
handles = {}
def wt(name):
    sh = wmap[name]
    if sh not in handles:
        handles[sh] = safe_open(f"{BASE}/{sh}", framework="pt")
    return handles[sh].get_tensor(name).to(torch.float32)

torch.manual_seed(0)
L = 3
P = f"model.language_model.layers.{L}."
attn = Qwen3_5Attention(cfg, L)
attn.q_proj.weight.data = wt(P + "self_attn.q_proj.weight")
attn.k_proj.weight.data = wt(P + "self_attn.k_proj.weight")
attn.v_proj.weight.data = wt(P + "self_attn.v_proj.weight")
attn.o_proj.weight.data = wt(P + "self_attn.o_proj.weight")
attn.q_norm.weight.data = wt(P + "self_attn.q_norm.weight").squeeze()
attn.k_norm.weight.data = wt(P + "self_attn.k_norm.weight").squeeze()
ln1w = wt(P + "input_layernorm.weight").squeeze()
ln2w = wt(P + "post_attention_layernorm.weight").squeeze()
mlp = Qwen3_5MLP(cfg, cfg.intermediate_size)
mlp.gate_proj.weight.data = wt(P + "mlp.gate_proj.weight")
mlp.up_proj.weight.data = wt(P + "mlp.up_proj.weight")
mlp.down_proj.weight.data = wt(P + "mlp.down_proj.weight")
for m in (attn, mlp):
    m.eval()

rope = Qwen3_5TextRotaryEmbedding(cfg)
x = torch.randn(1, 4, cfg.hidden_size)  # batch 1, 4 prompt tokens
pos = torch.arange(4).unsqueeze(0)
cos, sin = rope(x, pos)
print("cos/sin:", tuple(cos.shape), cos.dtype)

# full layer forward (prefill-style, 4 tokens)
h = x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1 + ln1w)
ao, _ = attn(h, (cos, sin), None)
h2 = x + ao
h3 = h2 * torch.rsqrt((h2.float() ** 2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1 + ln2w)
out = h2 + mlp(h3.float()).type_as(x)

# decode step: single new token at position 4 with KV cache length 4
from transformers import DynamicCache
cache = DynamicCache()
_, _ = attn(h, (cos, sin), None, past_key_values=cache)
xt = torch.randn(1, 1, cfg.hidden_size)
post = pos[:, -1:] + 1
cos1, sin1 = rope(xt, post)
ht = xt * torch.rsqrt((xt.float() ** 2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1 + ln1w)
# capture decode-step internals for kernel verification (BEFORE cache update,
# so K/V hold exactly the 4 prefill positions; q/gate are the new token's)
with torch.no_grad():
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_rotary_pos_emb as _rope)
    htp = attn.q_proj(ht)
    qq, gg = torch.chunk(htp.view(1, 1, 24, 512), 2, dim=-1)
    qq = attn.q_norm(qq).transpose(1, 2)          # [1, 24, 1, 256]
    kk = attn.k_norm(attn.k_proj(ht).view(1, 1, 4, 256)).transpose(1, 2)
    qq, _ = _rope(qq, qq.clone(), cos1, sin1)
    kk, _ = _rope(kk, kk.clone(), cos1, sin1)
    vv = attn.v_proj(ht).view(1, 1, 4, 256).transpose(1, 2)
    K4 = cache.layers[3].keys.clone()          # [1, 4, 4, 256] prefill K
    V4 = cache.layers[3].values.clone()        # [1, 4, 4, 256] prefill V
    K5 = torch.cat([K4, kk], dim=2)            # append new K -> 5
    V5 = torch.cat([V4, vv], dim=2)
ao1, wts = attn(ht, (cos1, sin1), None, past_key_values=cache)
print("attn weights shape:", tuple(wts.shape))  # [1, 24, 1, 5]

rec = {
    "layer": L, "note": "HF CPU fp32, batch 1, 4-token prefill + 1 decode step",
    "x_in": x.tolist(),
    "cos0": cos[0, 0, :8].tolist(), "sin0": sin[0, 0, :8].tolist(),
    "block_out": out.tolist(),
    "decode_token_out": (xt + ao1).tolist(),
    "decode_xt": xt.tolist(),
    "decode_attn_weights_head0": wts[0, 0, 0].tolist(),
    "q_norm_w0": attn.q_norm.weight.data[:4].tolist(),
    "decode_q": qq.squeeze(2).tolist(),      # [24, 256] post-norm+rope
    "decode_gate": gg.squeeze(0).squeeze(0).tolist(),  # [24, 256]
    "decode_K5": K5.squeeze(0).tolist(),     # [4 KV, 5, 256]
    "decode_V5": V5.squeeze(0).tolist(),     # [4 KV, 5, 256]
}
json.dump(rec, open(f"{REF}/attn_block_L3.json", "w"), indent=1)
print("WROTE attn_block_L3.json")

# redo layer-0 MLP with correct (1+w) norm
P0 = "model.language_model.layers.0."
ln0 = wt(P0 + "input_layernorm.weight").squeeze()
g = wt(P0 + "mlp.gate_proj.weight"); u = wt(P0 + "mlp.up_proj.weight")
d = wt(P0 + "mlp.down_proj.weight")
torch.manual_seed(0)
xx = torch.randn(2, cfg.hidden_size)
nn = xx * torch.rsqrt((xx ** 2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1 + ln0)
yy = xx + ((torch.nn.functional.silu(nn @ g.T) * (nn @ u.T)) @ d.T)
json.dump({"in": xx.tolist(), "out": yy.tolist(),
           "note": "layer0 RMSNorm(1+w)+MLP+residual, CORRECTED (old file used w not 1+w)"},
          open(f"{REF}/mlp_layer0.json", "w"), indent=1)
print("FIXED mlp_layer0.json")
