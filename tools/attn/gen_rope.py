"""RoPE application fixture: HF-applied Q/K vectors for exact kernel check."""
import json
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb)

BASE = "/mnt/usb/AInfer/models/Qwen3.8-27B"
REF = "/mnt/usb/AInfer/reference"
cfg = Qwen3_5TextConfig.from_dict(json.load(open(f"{BASE}/config.json"))["text_config"])
rope = Qwen3_5TextRotaryEmbedding(cfg)
torch.manual_seed(1)
q = torch.randn(1, 2, 4, 256)  # 2 q-heads, seq 4
k = torch.randn(1, 1, 4, 256)  # 1 kv-head
pos = torch.arange(4).unsqueeze(0)
cos, sin = rope(torch.empty(1, 4, 5120), pos)
qe, ke = apply_rotary_pos_emb(q, k, cos, sin)
json.dump({"q": q.tolist(), "k": k.tolist(),
           "cos": cos.tolist(), "sin": sin.tolist(),
           "q_applied": qe.tolist(), "k_applied": ke.tolist(),
           "note": "NeoX half-rotation, leading 64 dims; rest passthrough"},
          open(f"{REF}/rope_applied.json", "w"))
print("WROTE rope_applied.json", cos.shape)
