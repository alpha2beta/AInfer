"""T3.8 INT4 MLP quality ablation (CPU, torch): which projection(s) need BF16?
Compares against reference/mlp_layer0.json (corrected (1+w) RMSNorm).
Variants: control BF16; single-proj INT4 (isolate culprit); single-proj BF16
exceptions; group 64/32; asymmetric g128.
"""
import json
import torch
from safetensors import safe_open

BASE = "/mnt/usb/AInfer/models/Qwen3.8-27B"
REF = "/mnt/usb/AInfer/reference"


def quant(w, group=128, asym=False):
    """w: fp32 tensor. Returns dequantized fp32 (simulate)."""
    flat = w.reshape(-1)
    n = flat.numel()
    ng = (n + group - 1) // group
    pad = ng * group - n
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    g = flat.reshape(ng, group)
    if asym:
        mn, _ = g.min(dim=1, keepdim=True)
        mx, _ = g.max(dim=1, keepdim=True)
        sc = ((mx - mn) / 15).clamp_min(1e-12)
        zp = torch.round(-mn / sc).clamp(0, 15)
        q = torch.clamp(torch.round((g - mn) / sc), 0, 15)
        dq = mn + q * sc  # NOTE: zp folded into mn; stored zp unused in v1
    else:
        amax = g.abs().max(dim=1, keepdim=True).values
        sc = torch.where(amax == 0, torch.ones_like(amax), amax / 7)
        q = torch.clamp(torch.round(g / sc), -8, 7)
        dq = q * sc
    return dq.reshape(-1)[:n].reshape(w.shape)


def run_variant(gw, uw, dw, x, n, policy):
    """policy: dict proj->(group|None for BF16, asym bool)."""
    def maybe(name, w):
        p = policy.get(name)
        if p is None:
            return w
        grp, asym = p
        return quant(w, grp, asym)
    g = maybe("gate", gw)
    u = maybe("up", uw)
    d = maybe("down", dw)
    h = torch.nn.functional.silu(n @ g.T) * (n @ u.T)
    return x + h @ d.T


def main():
    P0 = "model.language_model.layers.0."
    # (open each tensor individually to bound memory)
    def load(n):
        with safe_open(f"{BASE}/model-00001-of-00018.safetensors", framework="pt") as f:
            return f.get_tensor(P0 + n).to(torch.float32)
    ln0 = load("input_layernorm.weight").squeeze()
    gw = load("mlp.gate_proj.weight")
    uw = load("mlp.up_proj.weight")
    dw = load("mlp.down_proj.weight")
    ref = json.load(open(f"{REF}/mlp_layer0.json"))
    x = torch.tensor(ref["in"])
    exp = torch.tensor(ref["out"])
    n = x * torch.rsqrt((x ** 2).mean(-1, keepdim=True) + 1e-6) * (1 + ln0)
    meanr = exp.abs().mean().item()
    I4 = (128, False)
    variants = {
        "bf16-control": {},
        "q-gate-only": {"gate": I4},
        "q-up-only": {"up": I4},
        "q-down-only": {"down": I4},
        "q-all-g128sym": {"gate": I4, "up": I4, "down": I4},
        "exc-gate": {"up": I4, "down": I4},
        "exc-up": {"gate": I4, "down": I4},
        "exc-down": {"gate": I4, "up": I4},
        "exc-gateup": {"down": I4},
        "q-all-g64sym": {"gate": (64, False), "up": (64, False), "down": (64, False)},
        "q-all-g32sym": {"gate": (32, False), "up": (32, False), "down": (32, False)},
        "q-all-g128asym": {"gate": (128, True), "up": (128, True), "down": (128, True)},
    }
    print(f"{'variant':16s} {'max':>10s} {'mean':>10s} {'meanrel':>8s}")
    for name, pol in variants.items():
        with torch.no_grad():
            y = run_variant(gw, uw, dw, x, n, pol)
        d = (y - exp).abs()
        print(f"{name:16s} {d.max().item():10.4f} {d.mean().item():10.5f} "
              f"{(d.mean()/meanr).item():8.4f}", flush=True)


if __name__ == "__main__":
    main()
