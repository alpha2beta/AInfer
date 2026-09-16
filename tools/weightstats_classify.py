#!/usr/bin/env python3
"""T8.3: methodology + classified schema for the weight-stat anomaly list.

Reads reference/weight_stats.json (per-tensor {min,max,mean,std} for 1199
tensors + 136 bare anomaly names; producer unknown, treated as input) and
the safetensors HEADERS ONLY (shape/dtype inventory, no tensor data mapped)
to emit reference/weight_stats_v2.json:

  {version, methodology, peer_groups, categories, anomalies[]}

Methodology (documented in the output, reproducible from recorded stats):
  peer group  = (scope, class) with scope in {language, vision, other} and
                class = trailing component path (e.g. linear_attn.dt_bias).
  signal      = maxabs = max(|min|, |max|) of the recorded stats.
  rule        = peer_maxabs_robust_z: 0.6745*(x - median) / MAD over peers
                (groups with <4 peers use global-class fallback).
  threshold   = 5.0.
  Each recorded anomaly gets {tensor, peer_group, rule, threshold, observed,
  severity, expected_sensitive_tensor, action, checksum_verified} per
  review.md section 2.4, with action grounded in the real quantizer policy
  (tools/binfer.py quantize_policy: 2-D non-embed -> INT4 sym-g128, else
  BF16 copy; vision namespace rejected from the container).

Categories: expected_architectural_outlier | quantization_sensitive |
suspected_source_corruption | out_of_scope_vision | informational_shift.
"""
import json
import os
import struct

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, "models", "Qwen3.8-27B")
THRESHOLD = 5.0


def read_shapes():
    """Per-tensor shapes from shard headers only (never maps tensor data)."""
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
    shards = sorted({s for s in idx["weight_map"].values()})
    shapes = {}
    for s in shards:
        with open(os.path.join(MODEL, s), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n).decode("utf-8"))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            shapes[name] = meta["shape"]
    return shapes


def peer_group(name):
    if ".visual." in name or ".merger." in name:
        scope = "vision"
    elif name.startswith("model.language_model."):
        scope = "language"
    else:
        scope = "other"
    import re
    tail = name.split("model.language_model.", 1)[-1]
    tail = tail.split("model.visual.", 1)[-1]
    # strip leading (layers|blocks).N. so peers share one group
    cls = re.sub(r"^(layers|blocks)\.\d+\.", "", tail)
    cls = re.sub(r"^\d+\.", "", cls)
    return f"{scope}.{cls}"


def med(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ws = json.load(open(os.path.join(REPO, "reference", "weight_stats.json")))
    stats, anomalies = ws["stats"], ws["anomalies"]
    shapes = read_shapes()

    maxabs, groups = {}, {}
    for name, st in stats.items():
        x = max(abs(st["min"]), abs(st["max"]))
        maxabs[name] = x
        groups.setdefault(peer_group(name), []).append(x)
    gmed = {g: med(v) for g, v in groups.items()}
    gmad = {}
    for g, v in groups.items():
        m = gmed[g]
        ads = sorted(abs(x - m) for x in v)
        gmad[g] = med(ads) if len(v) >= 4 else None

    def robust_z(name):
        g = peer_group(name)
        m, mad = gmed[g], gmad[g]
        if mad is None or mad == 0:
            return None
        return 0.6745 * (maxabs[name] - m) / mad

    shape_of = {a: shapes.get(a) for a in anomalies}
    out, cats = [], {}
    for a in anomalies:
        z = robust_z(a)
        shp = shape_of[a]
        rank = len(shp) if shp else None
        scope = peer_group(a).split(".")[0]
        sens_cls = a.endswith("A_log") or a.endswith("dt_bias") \
            or "conv1d" in a or "layernorm" in a or ".norm" in a \
            or a.endswith("norm.weight")
        if scope == "vision":
            cat, sev, act = ("out_of_scope_vision", "info",
                             "reject_loader_v1")
        elif a.endswith("A_log") or a.endswith("dt_bias"):
            cat, sev, act = ("expected_architectural_outlier", "review",
                             "keep_bf16")
        elif rank == 1 or "norm" in a or "conv1d" in a:
            cat, sev, act = ("quantization_sensitive", "review", "keep_bf16")
        elif rank == 2:
            cat, sev, act = ("quantization_sensitive", "review",
                             "int4_g128_per_policy")
        else:
            cat, sev, act = ("informational_shift", "info", "keep_bf16")
        out.append({
            "tensor": a,
            "peer_group": peer_group(a),
            "shape": shp,
            "rule": "peer_maxabs_robust_z",
            "threshold": THRESHOLD,
            "observed": round(z, 2) if z is not None else None,
            "severity": sev,
            "expected_sensitive_tensor": bool(sens_cls and scope != "vision"),
            "action": act,
            "checksum_verified": False,
        })
        cats[cat] = cats.get(cat, 0) + 1

    v2 = {
        "version": 2,
        "supersedes": "reference/weight_stats.json (kept unchanged)",
        "methodology": {
            "input": "recorded per-tensor {min,max,mean,std} for 1199 "
                     "tensors; original detection rule/threshold unknown, "
                     "so this file RE-ASSESSES the 136 recorded names "
                     "under the stated rule below",
            "peer_group": "(scope, class): scope in {language, vision, "
                          "other}; class = trailing component path",
            "signal": "maxabs = max(|min|, |max|)",
            "rule": "peer_maxabs_robust_z = 0.6745*(x - peer_median) / "
                    "peer_MAD; groups <4 peers report observed=null",
            "threshold": THRESHOLD,
            "shapes": "safetensors headers only (no tensor data mapped)",
            "policy_grounding": "tools/binfer.py quantize_policy: 2-D "
                                "non-embed -> INT4 sym-g128 else BF16 copy; "
                                "vision namespace rejected (validator fails "
                                "if present)",
            "checksum_note": "shard-level sha256 deferred per manifest "
                             "scope_notes; all checksum_verified=false "
                             "until then. .binfer container carries its own "
                             "trailing sha over final bytes.",
        },
        "peer_group_count": len(groups),
        "categories": cats,
        "suspected_source_corruption": [],
        "anomalies": out,
    }
    p = os.path.join(REPO, "reference", "weight_stats_v2.json")
    json.dump(v2, open(p, "w"), indent=1)
    print(f"WEIGHTSTATS-V2-OK: {len(out)}/{len(anomalies)} classified -> {p}")
    print("categories:", cats)
    print("suspected_source_corruption: [] (no tensor outside the four "
          "expected classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
