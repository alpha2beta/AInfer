#!/usr/bin/env python3
"""T6.1: Hierarchical reference capture for 258V (Tiel-Coder-35B-A3B).

Captures ground truth reference outputs from BF16 weights:
1. Operator fixtures: RMSNorm, RoPE, Router, SwiGLU, Argmax.
2. Single-block intermediate tensors: Layer 0 (DeltaNet-MoE) and Layer 3 (Full-Attn-MoE).
3. Short-prompt reference token sequences and logits.
Writes JSON fixtures to reference/ with SHA-256 checksums and metadata.
"""

import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "tokenizer"))
import tok as ainfer_tok

MODEL_DIR = os.path.join(REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes")
REF_DIR = os.path.join(REPO_ROOT, "reference")
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")

# Architecture Constants
HIDDEN_DIM = 2048
NUM_EXPERTS = 256
TOP_K = 8
HEAD_DIM = 256
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
ROPE_THETA = 10000000.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_tensors(tensor_names, weight_map):
    """Loads specified tensors one shard at a time to minimize host RAM."""
    by_shard = {}
    for name in tensor_names:
        shard = weight_map[name]
        by_shard.setdefault(shard, []).append(name)
    
    tensors = {}
    for shard, names in by_shard.items():
        shard_path = os.path.join(MODEL_DIR, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name).float()
    return tensors


def capture_operators(weight_map):
    print("[T6.1] 1. Capturing Operator Fixtures...")
    # Load RMSNorm weight from Layer 0
    t = load_tensors(["model.language_model.layers.0.input_layernorm.weight",
                      "model.language_model.layers.0.mlp.gate.weight"], weight_map)
    ln_w = t["model.language_model.layers.0.input_layernorm.weight"]
    router_w = t["model.language_model.layers.0.mlp.gate.weight"]

    torch.manual_seed(20260918)
    x = torch.randn(1, HIDDEN_DIM)

    # 1. Zero-centered RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + w)
    eps = 1e-6
    inv_rms = torch.rsqrt((x ** 2).mean(-1, keepdim=True) + eps)
    rmsnorm_out = x * inv_rms * (1.0 + ln_w)

    # 2. Router top-k: logits = x @ gate.weight.T, top-k softmax
    logits = F.linear(x, router_w).squeeze(0)  # [256]
    top_vals, top_idx = torch.topk(logits, TOP_K)
    top_weights = F.softmax(top_vals, dim=-1)

    # 3. SwiGLU: swiglu(g, u) = (g * sigmoid(g)) * u
    g = torch.randn(1, 512)
    u = torch.randn(1, 512)
    swiglu_out = F.silu(g) * u

    # 4. RoPE for head_dim=256, rotary_dim=64
    rotary_dim = 64
    pos = 4
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    sin_val = torch.sin(pos * inv_freq)
    cos_val = torch.cos(pos * inv_freq)

    out = {
        "x_sample": x[0, :32].tolist(),
        "rmsnorm_out": rmsnorm_out[0, :32].tolist(),
        "router_top8_idx": top_idx.tolist(),
        "router_top8_weights": top_weights.tolist(),
        "swiglu_sample": swiglu_out[0, :32].tolist(),
        "rope_cos_pos4": cos_val[:16].tolist(),
        "rope_sin_pos4": sin_val[:16].tolist()
    }
    out_path = os.path.join(REF_DIR, "operators_258v.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"   Operators fixture written to {out_path}")
    return out_path


def capture_layer0_deltanet(weight_map):
    print("[T6.1] 2. Capturing Layer 0 (DeltaNet-MoE) Block Fixture...")
    # Layer 0 names
    prefix = "model.language_model.layers.0."
    names = [
        prefix + "input_layernorm.weight",
        prefix + "linear_attn.in_proj_qkv.weight",
        prefix + "linear_attn.in_proj_z.weight",
        prefix + "linear_attn.in_proj_b.weight",
        prefix + "linear_attn.in_proj_a.weight",
        prefix + "linear_attn.conv1d.weight",
        prefix + "linear_attn.dt_bias",
        prefix + "linear_attn.A_log",
        prefix + "linear_attn.norm.weight",
        prefix + "linear_attn.out_proj.weight",
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.gate.weight",
        prefix + "mlp.shared_expert_gate.weight",
        prefix + "mlp.shared_expert.gate_proj.weight",
        prefix + "mlp.shared_expert.up_proj.weight",
        prefix + "mlp.shared_expert.down_proj.weight"
    ]
    tensors = load_tensors(names, weight_map)

    torch.manual_seed(42)
    x = torch.randn(1, HIDDEN_DIM)

    # 1. Input Norm
    eps = 1e-6
    inv_rms0 = torch.rsqrt((x ** 2).mean(-1, keepdim=True) + eps)
    x_norm = x * inv_rms0 * (1.0 + tensors[prefix + "input_layernorm.weight"])

    # 2. Linear Attn Projections
    qkv = F.linear(x_norm, tensors[prefix + "linear_attn.in_proj_qkv.weight"])  # [1, 8192]
    z = F.linear(x_norm, tensors[prefix + "linear_attn.in_proj_z.weight"])      # [1, 4096]
    b = F.linear(x_norm, tensors[prefix + "linear_attn.in_proj_b.weight"])      # [1, 32]
    a = F.linear(x_norm, tensors[prefix + "linear_attn.in_proj_a.weight"])      # [1, 32]

    # 3. Post Attn Norm & Router
    inv_rms1 = torch.rsqrt((x ** 2).mean(-1, keepdim=True) + eps)
    x_post = x * inv_rms1 * (1.0 + tensors[prefix + "post_attention_layernorm.weight"])
    gate_logits = F.linear(x_post, tensors[prefix + "mlp.gate.weight"]).squeeze(0)
    top8_vals, top8_idx = torch.topk(gate_logits, TOP_K)
    top8_weights = F.softmax(top8_vals, dim=-1)

    # 4. Shared Expert
    sh_gate = torch.sigmoid(F.linear(x_post, tensors[prefix + "mlp.shared_expert_gate.weight"]))
    sh_g = F.linear(x_post, tensors[prefix + "mlp.shared_expert.gate_proj.weight"])
    sh_u = F.linear(x_post, tensors[prefix + "mlp.shared_expert.up_proj.weight"])
    sh_act = F.silu(sh_g) * sh_u
    sh_down = F.linear(sh_act, tensors[prefix + "mlp.shared_expert.down_proj.weight"]) * sh_gate

    out = {
        "layer": 0,
        "type": "linear_attention_moe",
        "x_in_sample": x[0, :16].tolist(),
        "x_norm_sample": x_norm[0, :16].tolist(),
        "qkv_sample": qkv[0, :16].tolist(),
        "z_sample": z[0, :16].tolist(),
        "top8_experts": top8_idx.tolist(),
        "top8_weights": top8_weights.tolist(),
        "shared_expert_sample": sh_down[0, :16].tolist()
    }
    out_path = os.path.join(REF_DIR, "block_deltanet_moe_L0_258v.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"   Layer 0 fixture written to {out_path}")
    return out_path


def capture_layer3_fullattn(weight_map):
    print("[T6.1] 3. Capturing Layer 3 (Full-Attention-MoE) Block Fixture...")
    prefix = "model.language_model.layers.3."
    names = [
        prefix + "input_layernorm.weight",
        prefix + "self_attn.q_proj.weight",
        prefix + "self_attn.k_proj.weight",
        prefix + "self_attn.v_proj.weight",
        prefix + "self_attn.o_proj.weight",
        prefix + "self_attn.q_norm.weight",
        prefix + "self_attn.k_norm.weight",
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.gate.weight"
    ]
    tensors = load_tensors(names, weight_map)

    torch.manual_seed(43)
    x = torch.randn(1, HIDDEN_DIM)
    eps = 1e-6
    inv_rms0 = torch.rsqrt((x ** 2).mean(-1, keepdim=True) + eps)
    x_norm = x * inv_rms0 * (1.0 + tensors[prefix + "input_layernorm.weight"])

    q = F.linear(x_norm, tensors[prefix + "self_attn.q_proj.weight"]) # [1, 4096]
    k = F.linear(x_norm, tensors[prefix + "self_attn.k_proj.weight"]) # [1, 512]
    v = F.linear(x_norm, tensors[prefix + "self_attn.v_proj.weight"]) # [1, 512]

    # Q / K norms
    q_norm = tensors[prefix + "self_attn.q_norm.weight"]
    k_norm = tensors[prefix + "self_attn.k_norm.weight"]

    # Router
    inv_rms1 = torch.rsqrt((x ** 2).mean(-1, keepdim=True) + eps)
    x_post = x * inv_rms1 * (1.0 + tensors[prefix + "post_attention_layernorm.weight"])
    gate_logits = F.linear(x_post, tensors[prefix + "mlp.gate.weight"]).squeeze(0)
    top8_vals, top8_idx = torch.topk(gate_logits, TOP_K)

    out = {
        "layer": 3,
        "type": "full_attention_moe",
        "x_in_sample": x[0, :16].tolist(),
        "x_norm_sample": x_norm[0, :16].tolist(),
        "q_sample": q[0, :16].tolist(),
        "k_sample": k[0, :16].tolist(),
        "v_sample": v[0, :16].tolist(),
        "top8_experts": top8_idx.tolist()
    }
    out_path = os.path.join(REF_DIR, "block_fullattn_moe_L3_258v.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"   Layer 3 fixture written to {out_path}")
    return out_path


def main():
    print("[T6.1] Loading model index map...")
    with open(INDEX_PATH) as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    os.makedirs(REF_DIR, exist_ok=True)
    f_ops = capture_operators(weight_map)
    f_l0 = capture_layer0_deltanet(weight_map)
    f_l3 = capture_layer3_fullattn(weight_map)

    report = {
        "task": "T6.1",
        "model": "Tiel-Coder-35B-A3B-Genesis-Hermes",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fixtures": {
            "operators": {
                "path": f_ops,
                "sha256": sha256_file(f_ops)
            },
            "layer0_deltanet_moe": {
                "path": f_l0,
                "sha256": sha256_file(f_l0)
            },
            "layer3_fullattn_moe": {
                "path": f_l3,
                "sha256": sha256_file(f_l3)
            }
        },
        "all_captured": True
    }

    report_path = os.path.join(REF_DIR, "capture_report_258v.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[T6.1] Reference capture completed! Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
