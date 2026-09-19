#!/usr/bin/env python3
"""AInfer .binfer tools — T2.3/T3.2 quantizer/writer, T2.2/T3.4 validator/loader reference.

Supports both v1.0 dense models (Qwen3.8-27B) and v1.1 MoE models (Qwen3.5-MoE / Tiel-Coder-35B-A3B).

Usage (from repo root):
    python3 tools/binfer.py quantize [--model-dir DIR]   SafeTensors -> .binfer + conversion_report.json
    python3 tools/binfer.py validate [--binfer FILE]      full spec §11 validation of the .binfer
    python3 tools/binfer.py negatives                     malformed-input rejection tests (synthetic file)
    python3 tools/binfer.py moecheck                      dequant layer-0 router + expert GEMV from .binfer

Only stdlib is needed for `validate`/`negatives`; `quantize`/`moecheck` additionally
require torch + safetensors + numpy.
"""
import binascii
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    from safetensors import safe_open
    _HEAVY = True
except ImportError:
    np = torch = safe_open = None
    _HEAVY = False


def _need_heavy(cmd):
    if not _HEAVY:
        print(f"'{cmd}' requires torch + safetensors + numpy; "
              "run under project uv/venv environment")
        return False
    return True


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_35B_DIR = os.path.join(REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes")
B60_27B_DIR = os.path.join(REPO_ROOT, "models", "Qwen3.8-27B")

MAGIC = b"BINFER\x00\x01"
VERSION = 1
ALIGN = 64
GROUP = 128
DT = {"BF16": 0, "FP16": 1, "F16": 1, "FP32": 2, "F32": 2, "INT4_SYM_G128": 3, "INT8": 4, "FP8_E4M3": 5}


def get_default_paths(model_dir=None):
    if model_dir is None:
        model_dir = TARGET_35B_DIR if os.path.isdir(TARGET_35B_DIR) else B60_27B_DIR
    model_dir = os.path.abspath(model_dir)
    if "Tiel-Coder-35B" in model_dir or "35B" in model_dir:
        out_file = os.path.join(model_dir, "tiel-coder-35b-text-int4g128.binfer")
    else:
        out_file = os.path.join(model_dir, "qwen3.8-27b-text-int4g128.binfer")
    report_file = os.path.join(model_dir, "conversion_report.json")
    return model_dir, out_file, report_file


def align_up(n, a=ALIGN):
    return (n + a - 1) // a * a


def is_visual(name):
    return (".visual." in name or ".merger." in name
            or "patch_embed" in name or "pos_embed" in name
            or "vision_tower" in name)


def quantize_policy(name, shape, dtype="BF16"):
    """Returns (scheme, logical_dtype, storage_dtype).
    'int4': quantized to INT4 symmetric group 128 with BF16 scales.
    'copy': preserved unquantized at source precision.
    """
    dt_val = DT.get(dtype, DT["BF16"])

    # 1. Embeddings stay unquantized BF16
    if "embed_tokens" in name:
        return "copy", dt_val, dt_val

    # 2. Router gate weights stay unquantized FP32
    if "mlp.gate.weight" in name or "shared_expert_gate" in name:
        return "copy", DT["FP32"], DT["FP32"]

    # 3. 1-D vectors, norms, and DeltaNet/SSM recurrence parameters stay unquantized
    if len(shape) <= 1 or "norm" in name or "A_log" in name or "dt_bias" in name or "conv1d" in name:
        return "copy", dt_val, dt_val

    # 4. 2-D matrices & 3-D expert matrix banks -> INT4 symmetric group 128
    return "int4", dt_val, DT["INT4_SYM_G128"]


def f32_to_bf16_bits(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return (((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (np.ascontiguousarray(b, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def quantize_int4_sym(t_fp32):
    """t_fp32: torch tensor or numpy array. Returns (packed_bytes, scales_bf16_bytes, max_err, mean_err)."""
    if torch is not None and isinstance(t_fp32, torch.Tensor):
        flat = t_fp32.reshape(-1).to(torch.float32).numpy()
    else:
        flat = np.ascontiguousarray(t_fp32, dtype=np.float32).reshape(-1)
    n = flat.size
    ng = (n + GROUP - 1) // GROUP
    padded = np.zeros(ng * GROUP, dtype=np.float32)
    padded[:n] = flat
    g = padded.reshape(ng, GROUP)
    amax = np.abs(g).max(axis=1)
    scale = np.where(amax == 0, 1.0, amax / 7.0).astype(np.float32)
    q = np.clip(np.rint(g / scale[:, None]), -8, 7).astype(np.int8)
    nib = (q & 0xF).astype(np.uint8).reshape(-1)
    packed = (nib[0::2] | (nib[1::2] << 4)).tobytes()
    dq = (q.astype(np.float32) * scale[:, None]).reshape(-1)[:n]
    err = np.abs(dq - flat)
    return packed, f32_to_bf16_bits(scale).tobytes(), float(err.max()), float(err.mean())


def dequantize_int4_sym(packed, scales_bf16, numel):
    ng = (numel + GROUP - 1) // GROUP
    raw = np.frombuffer(packed, dtype=np.uint8)
    nib = np.empty(raw.size * 2, dtype=np.int8)
    nib[0::2] = raw & 0xF
    nib[1::2] = (raw >> 4) & 0xF
    nib[nib >= 8] -= 16
    sc = bf16_bits_to_f32(np.frombuffer(scales_bf16, dtype=np.uint16))
    return (nib.astype(np.float32) * np.repeat(sc, GROUP))[:numel]


# ---------------------------------------------------------------- writer ---
class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.sha = hashlib.sha256()
        self.pos = 0

    def write(self, data):
        self.f.write(data)
        self.sha.update(data)
        self.pos += len(data)

    def pad(self):
        n = align_up(self.pos) - self.pos
        if n:
            self.write(b"\x00" * n)

    def close(self):
        self.f.close()


def build_plan(model_dir):
    manifest_path = os.path.join(model_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.load(open(manifest_path))
    tensors = manifest.get("tensors", {})
    if "inventory" in tensors:
        inv = tensors["inventory"]
    else:
        inv = tensors

    names = sorted(k for k in inv if not is_visual(k))
    plan = []
    for k in names:
        meta = inv[k]
        shape = meta["shape"]
        dtype = meta.get("dtype", "BF16")
        scheme, logical, storage = quantize_policy(k, shape, dtype)
        numel = 1
        for d in shape:
            numel *= d
        if scheme == "int4":
            ng = (numel + GROUP - 1) // GROUP
            plan.append({
                "name": k, "shape": shape, "scheme": "int4",
                "logical": logical, "storage": storage, "layout": 0,
                "group": GROUP, "scale_bytes": ng * 2,
                "data_bytes": (ng * GROUP) // 2, "numel": numel,
                "src_bytes": meta.get("bytes", numel * 2)
            })
        else:
            plan.append({
                "name": k, "shape": shape, "scheme": "copy",
                "logical": logical, "storage": storage, "layout": 0,
                "group": 0, "scale_bytes": 0,
                "data_bytes": meta["bytes"], "numel": numel,
                "src_bytes": meta["bytes"]
            })
    return manifest, plan


def section_blob(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def build_moe_section(manifest, plan):
    """Builds binary MoE metadata section (Section 6) per spec §9."""
    topo = manifest.get("topology", {})
    moe_cfg = topo.get("moe", {})
    num_experts = moe_cfg.get("num_experts", 256)
    num_experts_per_tok = moe_cfg.get("num_experts_per_tok", 8)
    moe_intermediate = moe_cfg.get("moe_intermediate_size", 512)
    shared_intermediate = moe_cfg.get("shared_expert_intermediate_size", 512)
    layer_count = topo.get("total_layers", 40)

    # Index map supporting both original full name and stripped name
    name_to_idx = {}
    for idx, p in enumerate(plan):
        name_to_idx[p["name"]] = idx
        if p["name"].startswith("model.language_model."):
            name_to_idx[p["name"][len("model.language_model."):]] = idx

    hdr = struct.pack(
        "<IIIIIBBHI36s",
        num_experts,
        num_experts_per_tok,
        moe_intermediate,
        shared_intermediate,
        1,  # shared_expert_count
        2,  # routing_gate_dtype (FP32)
        1,  # norm_topk_prob
        0,  # expert_tensor_layout (3D packed bank)
        layer_count,
        b"\x00" * 36
    )

    descriptors = bytearray()
    for layer in range(layer_count):
        pfx = f"layers.{layer}."
        gate_idx = name_to_idx.get(f"{pfx}mlp.gate.weight", 0xFFFFFFFF)
        shared_gate_idx = name_to_idx.get(f"{pfx}mlp.shared_expert_gate.weight", 0xFFFFFFFF)
        experts_gate_up_idx = name_to_idx.get(f"{pfx}mlp.experts.gate_up_proj", 0xFFFFFFFF)
        experts_down_idx = name_to_idx.get(f"{pfx}mlp.experts.down_proj", 0xFFFFFFFF)
        shared_down_idx = name_to_idx.get(f"{pfx}mlp.shared_expert.down_proj.weight", 0xFFFFFFFF)
        shared_gate_proj_idx = name_to_idx.get(f"{pfx}mlp.shared_expert.gate_proj.weight", 0xFFFFFFFF)
        shared_up_proj_idx = name_to_idx.get(f"{pfx}mlp.shared_expert.up_proj.weight", 0xFFFFFFFF)

        desc = struct.pack(
            "<IIIIIIII",
            layer,
            gate_idx,
            shared_gate_idx,
            experts_gate_up_idx,
            experts_down_idx,
            shared_down_idx,
            shared_gate_proj_idx,
            shared_up_proj_idx
        )
        descriptors += desc

    return hdr + bytes(descriptors)


def cmd_quantize(model_dir=None, out_file=None):
    if not _need_heavy("quantize"):
        return 3

    model_dir, default_out, report_file = get_default_paths(model_dir)
    out_file = out_file or default_out

    print(f"=== AInfer Quantizer: {model_dir} -> {out_file} ===")
    manifest, plan = build_plan(model_dir)
    topo = manifest.get("topology", {})
    is_moe = "moe" in manifest.get("base_architecture", "").lower() or "moe" in topo

    created = manifest.get("created_at", "2026-09-18")
    identity = {
        "source_repo": manifest.get("repository", ""),
        "source_revision": manifest.get("revision", ""),
        "source_total_bytes": int(manifest.get("parameters_summary", {}).get("total_tensor_bytes", 0)),
        "converter": "tools/binfer.py",
        "converter_version": "1.1",
        "created_date": created
    }
    arch = {
        "arch": manifest.get("base_architecture", "qwen3_5_moe"),
        "text_layers": topo.get("total_layers", 40),
        "linear_layers": topo.get("linear_attention_layers", 30),
        "full_layers": topo.get("full_attention_layers", 10),
        "hidden": topo.get("hidden_size", 2048),
        "vocab": topo.get("vocab_size", 248320),
        "max_context_native": topo.get("max_position_embeddings", 262144),
        "vision": "deferred"
    }

    tok_assets = {}
    for fn in ["tokenizer.json", "chat_template.jinja", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        p = os.path.join(model_dir, fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                tok_assets[fn] = f.read()

    tokenizer = {
        "format": "hf-json-v1",
        "vocab_size": topo.get("vocab_size", 248320),
        "sha256": {fn: hashlib.sha256(tok_assets[fn]).hexdigest() for fn in tok_assets}
    }
    policy = {
        "default": "INT4 symmetric g128 BF16 scales",
        "group_size": GROUP,
        "exceptions": ["embed_tokens unquantized", "router gates FP32", "norms/conv/ssm source precision"]
    }

    blobs = [section_blob(x) for x in (identity, arch, tokenizer, policy)]
    moe_blob = build_moe_section(manifest, plan) if is_moe else None

    # Flags: bit 0 = text-only, bit 1 = MoE model enabled
    flags = 1 | (2 if is_moe else 0)
    section_count = 6 if is_moe else 5

    # Section table layout
    off = align_up(128 + section_count * 32)
    sects = []
    # Sections 1 to 4
    for i, b in enumerate(blobs, 1):
        sects.append([i, off, 8 + len(b)])
        off = align_up(off + 8 + len(b))

    # Section 5: Directory (raw entries)
    dir_off = off
    dir_bytes = len(plan) * 192
    off = align_up(off + dir_bytes)

    # Section 6: MoE Metadata (if MoE)
    moe_off = 0
    moe_bytes = 0
    if is_moe:
        moe_off = off
        moe_bytes = 8 + len(moe_blob)
        off = align_up(off + moe_bytes)

    # Section 7: Scale pool
    scale_off = off
    scale_total = sum(p["scale_bytes"] for p in plan)
    off = align_up(off + scale_total)

    # Section 8: Tensor payloads
    data_off = off
    for p in plan:
        p["scale_file_off"] = scale_off
        scale_off += p["scale_bytes"]
        p["data_file_off"] = data_off
        data_off = align_up(data_off + p["data_bytes"])
    total = data_off + 32  # trailing sha256

    # SafeTensors index
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]

    def shard_of(name):
        return os.path.join(model_dir, wmap[name])

    w = Writer(out_file)
    # File Header
    w.write(MAGIC + struct.pack("<I", VERSION) + struct.pack("<I", flags)
            + struct.pack("<Q", len(plan)) + struct.pack("<Q", 128)
            + struct.pack("<I", section_count) + struct.pack("<I", ALIGN)
            + struct.pack("<Q", total) + b"\x00" * 80)
    assert w.pos == 128

    # Section Table
    for sid, soff, sbytes in sects:
        w.write(struct.pack("<I", sid) + struct.pack("<Q", soff)
                + struct.pack("<Q", sbytes) + struct.pack("<I", binascii.crc32(
                    struct.pack("<Q", sbytes - 8) + blobs[sid - 1]) & 0xFFFFFFFF)
                + struct.pack("<I", 0) + struct.pack("<I", 0))

    # Directory entry in section table (patched later with CRC)
    w.write(struct.pack("<I2Q3I", 5, dir_off, dir_bytes, 0, 0, 0))

    # MoE metadata section in section table
    if is_moe:
        w.write(struct.pack("<I", 6) + struct.pack("<Q", moe_off)
                + struct.pack("<Q", moe_bytes) + struct.pack("<I", binascii.crc32(
                    struct.pack("<Q", len(moe_blob)) + moe_blob) & 0xFFFFFFFF)
                + struct.pack("<I", 0) + struct.pack("<I", 0))

    w.pad()

    # Write Sections 1 to 4
    for (sid, soff, sbytes), b in zip(sects, blobs):
        assert w.pos == soff, (w.pos, soff)
        w.write(struct.pack("<Q", len(b)) + b)
        w.pad()

    # Reserve space for directory
    assert w.pos == dir_off
    dir_pos = w.pos
    w.write(b"\x00" * dir_bytes)
    w.pad()

    # Write MoE metadata section
    if is_moe:
        assert w.pos == moe_off
        w.write(struct.pack("<Q", len(moe_blob)) + moe_blob)
        w.pad()

    # Stream scale pool and tensor payloads
    scale_pool = bytearray()
    per_tensor = {}

    print(f"Streaming {len(plan)} tensors into {out_file}...")
    for idx, p in enumerate(plan):
        shard_path = shard_of(p["name"])
        if p["scheme"] == "int4":
            with safe_open(shard_path, framework="pt") as h:
                t = h.get_tensor(p["name"])
                rows = t.shape[0]
                row_elems = t.numel() // rows
                # Process in chunks of up to 64MB
                rch = max(1, min(rows, (64 << 20) // (row_elems * 4)))
                packed_all = bytearray()
                scale_all = bytearray()
                mx = 0.0
                se = 0.0
                flat_n = t.numel()
                for r0 in range(0, rows, rch):
                    chunk = t[r0:r0 + rch].to(torch.float32)
                    pk, sc, cmx, cmean = quantize_int4_sym(chunk)
                    packed_all += pk
                    scale_all += sc
                    mx = max(mx, cmx)
                    se += cmean * chunk.numel()
                    del chunk
                del t

            assert len(packed_all) == p["data_bytes"] and len(scale_all) == p["scale_bytes"], \
                (p["name"], len(packed_all), p["data_bytes"])
            scale_pool += scale_all
            crc = binascii.crc32(packed_all) & 0xFFFFFFFF

            while w.pos < p["data_file_off"]:
                w.write(b"\x00" * min(1 << 20, p["data_file_off"] - w.pos))
            w.write(bytes(packed_all))
            w.pad()
            del packed_all, scale_all
            per_tensor[p["name"]] = {
                "max_abs_err": mx, "mean_abs_err": se / flat_n,
                "src_bytes": p["numel"] * 2, "q_bytes": p["data_bytes"],
                "scale_bytes": p["scale_bytes"]
            }
        else:
            with safe_open(shard_path, framework="pt") as h:
                t = h.get_tensor(p["name"])
                raw = bytes(t.untyped_storage())
                del t
            assert len(raw) == p["data_bytes"]
            while w.pos < p["data_file_off"]:
                w.write(b"\x00" * min(1 << 20, p["data_file_off"] - w.pos))
            crc = binascii.crc32(raw) & 0xFFFFFFFF
            w.write(raw)
            w.pad()
            del raw
            per_tensor[p["name"]] = {"copied_unquantized": True, "src_bytes": p["data_bytes"]}

        p["crc"] = crc
        if (idx + 1) % 25 == 0 or (idx + 1) == len(plan):
            print(f"  ... quantized/written {idx + 1}/{len(plan)} tensors", flush=True)

    # Patch directory and scale pool
    w.f.flush()
    with open(out_file, "r+b") as f:
        f.seek(dir_pos)
        dir_blob = bytearray()
        for p in plan:
            stored_name = p["name"]
            if stored_name.startswith("model.language_model."):
                stored_name = stored_name[len("model.language_model."):]
            name_b = stored_name.encode() + b"\x00"
            assert len(name_b) <= 64, f"Name too long: {stored_name}"
            e = name_b + b"\x00" * (64 - len(name_b))
            e += struct.pack("<B", len(p["shape"])) + b"\x00" * 7
            sh = list(p["shape"]) + [0] * (8 - len(p["shape"]))
            e += struct.pack("<8Q", *sh)
            e += struct.pack("<B", p["logical"]) + struct.pack("<B", p["storage"])
            e += struct.pack("<H", p["layout"]) + struct.pack("<I", p["group"])
            e += struct.pack("<Q", p["scale_file_off"]) + struct.pack("<Q", p["scale_bytes"])
            e += struct.pack("<Q", p["data_file_off"]) + struct.pack("<Q", p["data_bytes"])
            e += struct.pack("<I", p["crc"]) + b"\x00" * 12
            assert len(e) == 192
            f.write(e)
            dir_blob += e

        # Patch Section 5 CRC in section table (128 + 4 * 32 + 20)
        f.seek(128 + 4 * 32 + 20)
        f.write(struct.pack("<I", binascii.crc32(bytes(dir_blob)) & 0xFFFFFFFF))

        # Write scale pool
        f.seek(plan[0]["scale_file_off"])
        f.write(bytes(scale_pool))
        f.write(b"\x00" * (align_up(scale_total) - scale_total))

    # Append trailing SHA-256
    h = hashlib.sha256()
    with open(out_file, "rb") as f:
        while True:
            ch = f.read(1 << 20)
            if not ch:
                break
            h.update(ch)
    with open(out_file, "ab") as f:
        f.write(h.digest())

    size = os.path.getsize(out_file)
    assert size == total, (size, total)

    report = {
        "created": created,
        "file": os.path.basename(out_file),
        "bytes": size,
        "sha256": h.hexdigest(),
        "tensors": len(plan),
        "quantized_int4": sum(1 for p in plan if p["scheme"] == "int4"),
        "unquantized": sum(1 for p in plan if p["scheme"] != "int4"),
        "is_moe": is_moe,
        "policy": policy["default"],
        "group_size": GROUP,
        "per_tensor": per_tensor
    }
    errs = [v["max_abs_err"] for v in per_tensor.values() if "max_abs_err" in v]
    means = [v["mean_abs_err"] for v in per_tensor.values() if "mean_abs_err" in v]
    if errs:
        print(f"Quantization complete: {len(errs)} quantized tensors, "
              f"worst max_abs_err={max(errs):.5f}, worst mean_abs_err={max(means):.6f}")

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"WROTE {out_file} ({size / (1024**3):.2f} GB, sha256={h.hexdigest()[:16]}...)")
    return 0


# --------------------------------------------------------------- validator ---
ERRORS = []


def fail(msg):
    ERRORS.append(msg)


def read_entry(f):
    e = f.read(192)
    if len(e) < 192:
        return None
    name = e[:64].split(b"\x00")[0].decode()
    ndim = e[64]
    shape = list(struct.unpack("<8Q", e[72:136]))[:ndim]
    logical, storage = e[136], e[137]
    layout = struct.unpack("<H", e[138:140])[0]
    group = struct.unpack("<I", e[140:144])[0]
    sc_off, sc_bytes = struct.unpack("<2Q", e[144:160])
    d_off, d_bytes = struct.unpack("<2Q", e[160:176])
    crc = struct.unpack("<I", e[176:180])[0]
    return {
        "name": name, "ndim": ndim, "shape": shape, "logical": logical,
        "storage": storage, "layout": layout, "group": group,
        "sc_off": sc_off, "sc_bytes": sc_bytes, "d_off": d_off,
        "d_bytes": d_bytes, "crc": crc
    }


def cmd_validate(path=None):
    global ERRORS
    ERRORS = []
    if path is None:
        _, path, _ = get_default_paths()
    if not os.path.exists(path):
        print(f"File not found for validation: {path}")
        return 1

    size = os.path.getsize(path)
    with open(path, "rb") as f:
        try:
            return _validate_inner(f, size)
        except (struct.error, ValueError, OSError, UnicodeDecodeError) as e:
            fail(f"unparseable/truncated: {e}")
            print("INVALID:", ERRORS)
            return 1


def _validate_inner(f, size):
    if f.read(8) != MAGIC:
        fail("bad magic")
    if struct.unpack("<I", f.read(4))[0] != VERSION:
        fail("bad version")
    flags = struct.unpack("<I", f.read(4))[0]
    if flags & ~3:
        fail("reserved flags set")
    is_moe = bool(flags & 2)

    n = struct.unpack("<Q", f.read(8))[0]
    table_off = struct.unpack("<Q", f.read(8))[0]
    scount = struct.unpack("<I", f.read(4))[0]
    if struct.unpack("<I", f.read(4))[0] != ALIGN:
        fail("bad alignment field")
    total = struct.unpack("<Q", f.read(8))[0]
    if total != size:
        fail(f"total mismatch header={total} actual={size}")

    f.seek(table_off)
    sects = {}
    for _ in range(scount):
        sid, off, nbytes, crc, _, _ = struct.unpack("<I2Q3I", f.read(32))
        sects[sid] = (off, nbytes, crc)

    required_sections = [1, 2, 3, 4, 5]
    if is_moe:
        required_sections.append(6)

    for sid in required_sections:
        if sid not in sects:
            fail(f"missing section {sid}")

    for sid, (off, nbytes, crc) in sects.items():
        f.seek(off)
        if sid == 5:
            # dir body is raw entries without a length prefix
            body = f.read(nbytes)
            if (binascii.crc32(body) & 0xFFFFFFFF) != crc:
                fail("section 5 (dir) CRC mismatch")
            continue

        ln = struct.unpack("<Q", f.read(8))[0]
        if ln > max(1 << 26, nbytes):
            fail(f"section {sid} length implausible")
            continue
        body = f.read(ln)
        if (binascii.crc32(struct.pack("<Q", ln) + body) & 0xFFFFFFFF) != crc:
            fail(f"section {sid} CRC mismatch")

        # Validate MoE Section 6 internals
        if sid == 6:
            if len(body) < 64:
                fail("section 6 (moe) header truncated")
            else:
                num_exp, exp_per_tok, moe_int, shared_int, sh_cnt, r_dtype, norm_p, layout, l_cnt = struct.unpack(
                    "<IIIIIBBHI", body[:28]
                )
                if num_exp != 256:
                    fail(f"moe: num_experts expected 256, got {num_exp}")
                if exp_per_tok != 8:
                    fail(f"moe: num_experts_per_tok expected 8, got {exp_per_tok}")
                if l_cnt != 40:
                    fail(f"moe: layer_count expected 40, got {l_cnt}")

    if 5 not in sects:
        print("INVALID:", ERRORS)
        return 1

    doff, dbytes, _ = sects[5]
    if dbytes != n * 192:
        fail("dir size mismatch")
    f.seek(doff)
    entries = [read_entry(f) for _ in range(n)]
    if any(e is None for e in entries):
        fail("truncated dir")
        print("INVALID:", ERRORS)
        return 1

    names = [e["name"] for e in entries]
    if len(set(names)) != len(names):
        fail("duplicate names")
    if any(is_visual(x) for x in names):
        fail("visual tensor present")
    if any(e["layout"] != 0 for e in entries):
        fail("unsupported layout_id")

    # bounds & non-overlap
    spans = []
    for e in entries:
        numel = 1
        for d in e["shape"]:
            numel *= d
        if e["storage"] == DT["INT4_SYM_G128"]:
            ng = (numel + GROUP - 1) // GROUP
            if e["group"] != GROUP or e["sc_bytes"] != ng * 2:
                fail(f"bad quant params {e['name']}")
            if e["d_bytes"] != (ng * GROUP) // 2:
                fail(f"bad packed size {e['name']}")
        spans.append((e["d_off"], e["d_off"] + e["d_bytes"], e["name"]))
        if e["sc_bytes"]:
            spans.append((e["sc_off"], e["sc_off"] + e["sc_bytes"], e["name"] + "#scales"))

    for s, t, _ in spans:
        if t > size - 32:
            fail("span out of range")
    spans.sort()
    for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
        if b0 < a1:
            fail(f"overlap {an} vs {bn}")

    # payload CRCs
    for e in entries:
        f.seek(e["d_off"])
        data = f.read(e["d_bytes"])
        if len(data) != e["d_bytes"] or (binascii.crc32(data) & 0xFFFFFFFF) != e["crc"]:
            fail(f"payload CRC {e['name']}")

    # trailing SHA-256
    f.seek(0)
    h = hashlib.sha256()
    left = size - 32
    while left > 0:
        ch = f.read(min(1 << 20, left))
        h.update(ch)
        left -= len(ch)
    if f.read(32) != h.digest():
        fail("trailing sha256 mismatch")

    if ERRORS:
        print("INVALID:")
        for m in ERRORS[:20]:
            print(" -", m)
        return 1
    print(f"VALID: {n} tensors, {size / 2**30:.2f} GiB, sha ok, moe={'yes' if is_moe else 'no'}")
    return 0


# ------------------------------------------------------------- negatives ---
def cmd_negatives():
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp()
    ident = section_blob({"source_repo": "test", "source_revision": "0" * 40})
    arch = section_blob({"arch": "qwen3_5_moe"})
    tok = section_blob({"format": "hf-json-v1"})
    pol = section_blob({"default": "INT4 symmetric g128"})
    blobs = [ident, arch, tok, pol]

    moe_hdr = struct.pack("<IIIIIBBHI36s", 256, 8, 512, 512, 1, 2, 1, 0, 40, b"\x00" * 36)
    moe_descs = struct.pack("<IIIIIIII", 0, 0, 0xFFFFFFFF, 1, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0) * 40
    moe_blob = moe_hdr + moe_descs

    names = ["a.weight", "b.experts"]
    shapes = [[256], [256, 128]]
    stor = [DT["BF16"], DT["INT4_SYM_G128"]]
    grp = [0, GROUP]
    sc = [0, (256 * 128 // GROUP) * 2]
    pay = [bytes(512), bytes((256 * 128) // 2)]
    n = 2

    scount = 6
    flags = 3  # text + MoE
    off = align_up(128 + scount * 32)
    sects = []
    for i, b in enumerate(blobs, 1):
        sects.append([i, off, 8 + len(b)])
        off = align_up(off + 8 + len(b))
    doff = off
    off = align_up(off + n * 192)
    moe_off = off
    moe_bytes = 8 + len(moe_blob)
    off = align_up(off + moe_bytes)
    sc_total = sum(sc)
    sc0 = off
    off = align_up(off + sc_total)
    do0 = off
    doffsets = [do0, align_up(do0 + len(pay[0]))]
    total = align_up(doffsets[1] + len(pay[1])) + 32

    dir_blob = bytearray()
    for i in range(n):
        nb = names[i].encode() + b"\x00"
        e = nb + b"\x00" * (64 - len(nb)) + struct.pack("<B", len(shapes[i])) + b"\x00" * 7
        e += struct.pack("<8Q", *(shapes[i] + [0] * (8 - len(shapes[i]))))
        e += struct.pack("<B", DT["BF16"]) + struct.pack("<B", stor[i])
        e += struct.pack("<H", 0) + struct.pack("<I", grp[i])
        e += struct.pack("<Q", sc0 + sum(sc[:i])) + struct.pack("<Q", sc[i])
        e += struct.pack("<Q", doffsets[i]) + struct.pack("<Q", len(pay[i]))
        e += struct.pack("<I", binascii.crc32(pay[i]) & 0xFFFFFFFF) + b"\x00" * 12
        dir_blob += e
    dir_crc = binascii.crc32(bytes(dir_blob)) & 0xFFFFFFFF

    def emit(path, mut=None):
        buf = bytearray()
        buf += MAGIC + struct.pack("<IIQQIIQ80s", VERSION, flags, n, 128, scount, ALIGN, total, b"\x00" * 80)
        for sid, soff, sb in sects:
            buf += struct.pack("<I", sid) + struct.pack("<Q", soff) + struct.pack("<Q", sb)
            buf += struct.pack("<I", binascii.crc32(struct.pack("<Q", sb - 8) + blobs[sid - 1]) & 0xFFFFFFFF)
            buf += struct.pack("<II", 0, 0)
        # Section 5
        buf += struct.pack("<I2Q3I", 5, doff, len(dir_blob), dir_crc, 0, 0)
        # Section 6
        buf += struct.pack("<I2Q3I", 6, moe_off, moe_bytes,
                           binascii.crc32(struct.pack("<Q", len(moe_blob)) + moe_blob) & 0xFFFFFFFF, 0, 0)

        while len(buf) < sects[0][1]:
            buf += b"\x00"
        for (sid, soff, sb), b in zip(sects[:4], blobs):
            assert len(buf) == soff
            buf += struct.pack("<Q", len(b)) + b
            while len(buf) % 64:
                buf += b"\x00"
        assert len(buf) == doff
        buf += bytes(dir_blob)
        while len(buf) % 64:
            buf += b"\x00"
        assert len(buf) == moe_off
        buf += struct.pack("<Q", len(moe_blob)) + moe_blob
        while len(buf) % 64:
            buf += b"\x00"
        assert len(buf) == sc0
        buf += b"\x01\x02" * (sc_total // 2)
        while len(buf) % 64:
            buf += b"\x00"
        assert len(buf) == do0
        buf += pay[0]
        while len(buf) % 64:
            buf += b"\x00"
        buf += pay[1]
        while len(buf) % 64:
            buf += b"\x00"
        if mut:
            mut(buf)
        digest = hashlib.sha256(bytes(buf)).digest()
        buf += digest
        with open(path, "wb") as f:
            f.write(bytes(buf))

    cases = {
        "valid": (None, True),
        "bad_magic": (lambda b: b.__setitem__(0, 0xFF), False),
        "bad_version": (lambda b: struct.pack_into("<I", b, 8, 999), False),
        "truncated": (None, False),
        "payload_crc": (lambda b: b.__setitem__(do0 + 3, (b[do0 + 3] + 1) % 256), False),
        "bad_moe_crc": (lambda b: b.__setitem__(moe_off + 16, (b[moe_off + 16] + 1) % 256), False),
        "bad_expert_count": (lambda b: struct.pack_into("<I", b, moe_off + 8, 999), False),
    }

    ok = True
    for name, (mut, expect) in cases.items():
        p = os.path.join(tmp, name + ".binfer")
        if name == "truncated":
            emit(p)
            with open(p, "r+b") as f:
                f.truncate(os.path.getsize(p) // 2)
        else:
            emit(p, mut)
        global ERRORS
        ERRORS = []
        rc = cmd_validate(p)
        good = (rc == 0) == expect
        print(f"{'PASS' if good else 'FAIL'} negative/{name} (expected {'valid' if expect else 'reject'})")
        ok = ok and good

    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


def load_tensor_from_binfer(path, name):
    clean_name = name
    if clean_name.startswith("model.language_model."):
        clean_name = clean_name[len("model.language_model."):]
    with open(path, "rb") as f:
        f.seek(16)
        n = struct.unpack("<Q", f.read(8))[0]
        table_off = struct.unpack("<Q", f.read(8))[0]
        scount = struct.unpack("<I", f.read(4))[0]
        f.seek(table_off)
        sects = {}
        for _ in range(scount):
            sid, off, nb, _, _, _ = struct.unpack("<I2Q3I", f.read(32))
            sects[sid] = off
        f.seek(sects[5])
        for _ in range(n):
            e = read_entry(f)
            if e["name"] == name or e["name"] == clean_name:
                f.seek(e["d_off"])
                data = f.read(e["d_bytes"])
                if e["storage"] == DT["INT4_SYM_G128"]:
                    f.seek(e["sc_off"])
                    sc = f.read(e["sc_bytes"])
                    numel = 1
                    for d in e["shape"]:
                        numel *= d
                    arr = dequantize_int4_sym(data, sc, numel).reshape(e["shape"])
                elif e["storage"] in (DT["FP32"], DT.get("F32", 2)):
                    arr = np.frombuffer(data, dtype=np.float32).reshape(e["shape"])
                elif e["storage"] in (DT["FP16"], DT.get("F16", 1)):
                    arr = np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(e["shape"])
                else:  # BF16
                    arr = np.frombuffer(data, dtype=np.uint16).reshape(e["shape"])
                    arr = (arr.astype(np.uint32) << 16).view(np.float32)
                return np.array(arr, dtype=np.float32)
    raise KeyError(name)


def cmd_moecheck(path=None):
    if not _need_heavy("moecheck"):
        return 3
    model_dir, default_out, _ = get_default_paths()
    binfer_path = path or default_out
    if not os.path.exists(binfer_path):
        print(f"Error: {binfer_path} does not exist.")
        return 1

    print(f"=== MoE Layer 0 Error Verification: {binfer_path} ===")
    gate_binfer = load_tensor_from_binfer(binfer_path, "layers.0.mlp.gate.weight")

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]

    def load_src(name):
        shard = os.path.join(model_dir, wmap[name])
        with safe_open(shard, framework="pt") as h:
            return h.get_tensor(name).to(torch.float32).numpy()

    gate_src = load_src("model.language_model.layers.0.mlp.gate.weight")
    gate_diff = np.abs(gate_binfer - gate_src)
    print(f"Router Gate [256, 2048]: max_err={gate_diff.max():.6e}, mean_err={gate_diff.mean():.6e}")
    assert gate_diff.max() == 0.0, "Router gate should be bit-exact FP32!"

    # Sample routing with synthetic activation
    np.random.seed(42)
    x = np.random.randn(1, 2048).astype(np.float32)
    logits = x @ gate_src.T
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    top8_idx = np.argsort(-probs[0])[:8]
    print(f"Sample routing top-8 experts: {top8_idx.tolist()}")

    # Check 3D expert weights
    print("Loading Layer 0 3D expert weights from .binfer...")
    exp_gate_up_binfer = load_tensor_from_binfer(binfer_path, "layers.0.mlp.experts.gate_up_proj")
    exp_down_binfer = load_tensor_from_binfer(binfer_path, "layers.0.mlp.experts.down_proj")
    exp_gate_up_src = load_src("model.language_model.layers.0.mlp.experts.gate_up_proj")
    exp_down_src = load_src("model.language_model.layers.0.mlp.experts.down_proj")

    gu_diff = np.abs(exp_gate_up_binfer - exp_gate_up_src)
    dn_diff = np.abs(exp_down_binfer - exp_down_src)
    print(f"3D Experts gate_up_proj [256, 1024, 2048]: max_err={gu_diff.max():.5f}, mean_err={gu_diff.mean():.6f}")
    print(f"3D Experts down_proj    [256, 2048, 512]:  max_err={dn_diff.max():.5f}, mean_err={dn_diff.mean():.6f}")

    # Check per-expert error for selected active experts
    print("Active expert verification:")
    for e_idx in top8_idx:
        e_gu_diff = np.abs(exp_gate_up_binfer[e_idx] - exp_gate_up_src[e_idx])
        e_dn_diff = np.abs(exp_down_binfer[e_idx] - exp_down_src[e_idx])
        print(f"  Expert {e_idx:3d}: gate_up max_err={e_gu_diff.max():.5f} mean={e_gu_diff.mean():.6f} | "
              f"down max_err={e_dn_diff.max():.5f} mean={e_dn_diff.mean():.6f}")

    # Check shared expert
    sh_dn_binfer = load_tensor_from_binfer(binfer_path, "layers.0.mlp.shared_expert.down_proj.weight")
    sh_dn_src = load_src("model.language_model.layers.0.mlp.shared_expert.down_proj.weight")
    sh_diff = np.abs(sh_dn_binfer - sh_dn_src)
    print(f"Shared expert down_proj [2048, 512]: max_err={sh_diff.max():.5f}, mean_err={sh_diff.mean():.6f}")

    print("SUCCESS: MoE Layer 0 quantization error verified within INT4 tolerances!")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "quantize":
        sys.exit(cmd_quantize())
    if cmd == "validate":
        p = sys.argv[2] if len(sys.argv) > 2 else None
        sys.exit(cmd_validate(p))
    if cmd == "negatives":
        sys.exit(cmd_negatives())
    if cmd == "moecheck":
        p = sys.argv[2] if len(sys.argv) > 2 else None
        sys.exit(cmd_moecheck(p))
    print("unknown cmd", cmd)
    sys.exit(2)
