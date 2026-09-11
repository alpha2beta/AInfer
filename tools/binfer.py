#!/usr/bin/env python3
"""AInfer .binfer tools — T2.3 quantizer/writer, T2.2 validator/loader reference.

Usage (from repo root):
    python3 tools/binfer.py quantize    SafeTensors -> .binfer + conversion_report.json
    python3 tools/binfer.py validate    full spec §11 validation of the .binfer
    python3 tools/binfer.py negatives   malformed-input rejection tests (synthetic file)
    python3 tools/binfer.py mlpcheck    dequant layer-0 MLP from .binfer vs reference

Only stdlib is needed for `validate`/`negatives`; `quantize`/`mlpcheck` additionally
require torch + safetensors + numpy.
"""
import binascii
import hashlib
import json
import os
import struct
import sys

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
              "see AGENTS.md (project venv) for setup")
        return False
    return True

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "models", "Qwen3.8-27B")
REF_DIR = os.path.join(REPO_ROOT, "reference")
OUT_FILE = os.path.join(MODEL_DIR, "qwen3.8-27b-text-int4g128.binfer")
REPORT_FILE = os.path.join(MODEL_DIR, "conversion_report.json")

MAGIC = b"BINFER\x00\x01"
VERSION = 1
ALIGN = 64
GROUP = 128
DT = {"BF16": 0, "FP16": 1, "FP32": 2, "INT4_SYM_G128": 3, "INT8": 4, "FP8_E4M3": 5}
EMBED = "model.language_model.embed_tokens.weight"


def align_up(n, a=ALIGN):
    return (n + a - 1) // a * a


def is_visual(name):
    return (".visual." in name or ".merger." in name
            or "patch_embed" in name or "pos_embed" in name)


def quantize_policy(name, shape):
    """(scheme, logical, storage): 'int4' or 'copy'."""
    if len(shape) == 2 and name != EMBED:
        return "int4", DT["BF16"], DT["INT4_SYM_G128"]
    return "copy", DT["BF16"], DT["BF16"]


def f32_to_bf16_bits(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return (((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (np.ascontiguousarray(b, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def quantize_int4_sym(t_fp32):
    """t_fp32: torch tensor. Returns (packed_bytes, scales_bf16_bytes, max_err, mean_err)."""
    flat = t_fp32.reshape(-1).to(torch.float32).numpy()
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


def build_plan():
    manifest = json.load(open(os.path.join(MODEL_DIR, "manifest.json")))
    inv = manifest["tensors"]["inventory"]
    names = sorted(k for k in inv if not is_visual(k))
    plan = []
    for k in names:
        shape = inv[k]["shape"]
        scheme, logical, storage = quantize_policy(k, shape)
        numel = 1
        for d in shape:
            numel *= d
        if scheme == "int4":
            ng = (numel + GROUP - 1) // GROUP
            plan.append({"name": k, "shape": shape, "scheme": "int4",
                         "logical": logical, "storage": storage, "layout": 0,
                         "group": GROUP, "scale_bytes": ng * 2,
                         "data_bytes": (ng * GROUP) // 2, "numel": numel})
        else:
            plan.append({"name": k, "shape": shape, "scheme": "copy",
                         "logical": logical, "storage": storage, "layout": 0,
                         "group": 0, "scale_bytes": 0,
                         "data_bytes": inv[k]["bytes"], "numel": numel})
    return manifest, plan


def section_blob(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def cmd_quantize():
    if not _need_heavy("quantize"):
        return 3
    manifest, plan = build_plan()
    text = manifest["text"]
    created = manifest["generated_at"]
    identity = {"source_repo": manifest["source"]["repo"],
                "source_revision": manifest["source"]["revision"],
                "source_total_bytes": int(manifest["source"]["index_total_size_bytes"]),
                "converter": "tools/binfer.py", "converter_version": "1.0",
                "created_date": created}
    arch = {"arch": manifest["architecture"], "text_layers": text["num_hidden_layers"],
            "hidden": text["hidden_size"], "intermediate": text["intermediate_size"],
            "vocab": text["vocab_size"], "q_heads": text["full_attention"]["num_heads"],
            "kv_heads": text["full_attention"]["num_kv_heads"],
            "head_dim": text["full_attention"]["head_dim"], "rope": text["rope"],
            "mtp_layers": manifest["mtp"]["num_hidden_layers"],
            "vision": "deferred", "v1_context_cap": 4096}
    tok_assets = {}
    for fn in ["tokenizer.json", "chat_template.jinja", "tokenizer_config.json", "merges.txt"]:
        with open(os.path.join(MODEL_DIR, fn), "rb") as f:
            tok_assets[fn] = f.read()
    tokenizer = {"format": "hf-json-v1", "vocab_size": manifest["tokenizer"].get("vocab_size", 248077),
                 "embedded": ["tokenizer.json", "chat_template.jinja"],
                 "sha256": {fn: hashlib.sha256(tok_assets[fn]).hexdigest() for fn in tok_assets}}
    policy = {"default": "INT4 symmetric g128 BF16 scales", "group_size": GROUP,
              "exceptions": [EMBED + " BF16", "1-D norms source precision",
                             "A_log/dt_bias/conv1d source precision", "visual rejected"]}
    blobs = [section_blob(x) for x in (identity, arch, tokenizer, policy)]

    # layout computation
    off = align_up(128 + 5 * 32)
    sects = []
    for i, b in enumerate(blobs, 1):
        sects.append([i, off, 8 + len(b)])
        off = align_up(off + 8 + len(b))
    dir_off = off
    dir_bytes = len(plan) * 192
    off = align_up(off + dir_bytes)
    scale_off = off
    scale_total = sum(p["scale_bytes"] for p in plan)
    off = align_up(off + scale_total)
    data_off = off
    for p in plan:
        p["scale_file_off"] = scale_off
        scale_off += p["scale_bytes"]
        p["data_file_off"] = data_off
        data_off = align_up(data_off + p["data_bytes"])
    total = data_off + 32  # trailing sha256

    # tensor -> shard map (shards opened one tensor at a time: 55GB cannot stay mmapped
    # on this box — full 18-handle mapping exhausts the Windows paging file)
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]

    def shard_of(name):
        return os.path.join(MODEL_DIR, wmap[name])

    w = Writer(OUT_FILE)
    w.write(MAGIC + struct.pack("<I", VERSION) + struct.pack("<I", 1)
            + struct.pack("<Q", len(plan)) + struct.pack("<Q", 128)
            + struct.pack("<I", 5) + struct.pack("<I", ALIGN)
            + struct.pack("<Q", total) + b"\x00" * 80)
    assert w.pos == 128
    for sid, soff, sbytes in sects:
        w.write(struct.pack("<I", sid) + struct.pack("<Q", soff)
                + struct.pack("<Q", sbytes) + struct.pack("<I", binascii.crc32(
                    struct.pack("<Q", sbytes - 8) + blobs[sid - 1]) & 0xFFFFFFFF)
                + struct.pack("<I", 0) + struct.pack("<I", 0))
    # 5th table entry: tensor dir (CRC patched in r+b phase; off/bytes known now)
    w.write(struct.pack("<I2Q3I", 5, dir_off, dir_bytes, 0, 0, 0))
    w.pad()
    for (sid, soff, sbytes), b in zip(sects, blobs):
        assert w.pos == soff, (w.pos, soff)
        w.write(struct.pack("<Q", len(b)) + b)
        w.pad()

    # tensor directory: reserve zeros, stream payloads (single pass over source),
    # then patch dir entries + section-5 CRC via r+b
    assert w.pos == dir_off
    dir_pos = w.pos
    w.write(b"\x00" * dir_bytes)
    w.pad()
    # scale pool + payloads streamed per tensor in plan order
    scale_pool = bytearray()
    per_tensor = {}
    for idx, p in enumerate(plan):
        if p["scheme"] == "int4":
            # all tensor use happens inside the mmap lifetime
            with safe_open(shard_of(p["name"]), framework="pt") as h:
                t = h.get_tensor(p["name"])  # BF16 view; convert per chunk
                rows = t.shape[0]
                row_elems = t.numel() // rows
                # chunk rows so each fp32 chunk is <= 64MB
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
            # tail padding note: quantize pads final group with zeros; entry shape
            # gives true numel so dequant callers slice [:numel]
            assert len(packed_all) == p["data_bytes"] and len(scale_all) == p["scale_bytes"], \
                (p["name"], len(packed_all), p["data_bytes"])
            scale_pool += scale_all
            crc = binascii.crc32(packed_all) & 0xFFFFFFFF
            # write payload at its offset (zero-fill any alignment gap first)
            while w.pos < p["data_file_off"]:
                w.write(b"\x00" * min(1 << 20, p["data_file_off"] - w.pos))
            w.write(bytes(packed_all))
            w.pad()
            del packed_all, scale_all
            per_tensor[p["name"]] = {"max_abs_err": mx, "mean_abs_err": se / flat_n,
                                     "src_bytes": p["numel"] * 2, "q_bytes": p["data_bytes"],
                                     "scale_bytes": p["scale_bytes"]}
        else:
            with safe_open(shard_of(p["name"]), framework="pt") as h:
                raw = bytes(h.get_tensor(p["name"]).untyped_storage())
            assert len(raw) == p["data_bytes"]
            while w.pos < p["data_file_off"]:
                w.write(b"\x00" * min(1 << 20, p["data_file_off"] - w.pos))
            crc = binascii.crc32(raw) & 0xFFFFFFFF
            w.write(raw)
            w.pad()
            del raw
            per_tensor[p["name"]] = {"copied_bf16": True, "src_bytes": p["data_bytes"]}
        p["crc"] = crc
        if (idx + 1) % 100 == 0:
            print(f"  ... {idx + 1}/{len(plan)}", flush=True)
    # write scale pool into its reserved region: need random access -> reopen r+b
    w.f.flush()
    with open(OUT_FILE, "r+b") as f:
        # dir entries (section 5 body is raw entries, no length prefix;
        # section-table CRC covers the raw dir bytes)
        f.seek(dir_pos)
        dir_blob = bytearray()
        for p in plan:
            name_b = p["name"].encode() + b"\x00"
            assert len(name_b) <= 64
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
        f.seek(128 + 4 * 32 + 20)  # section-5 table entry CRC field
        f.write(struct.pack("<I", binascii.crc32(bytes(dir_blob)) & 0xFFFFFFFF))
        # scale pool starts at the first tensor's scale offset; pool bytes then zero pad
        f.seek(plan[0]["scale_file_off"])
        f.write(bytes(scale_pool))
        # pad already zeros from initial write? Initial write only wrote dir zeros; regions
        # between were filled by sequential payload writes with explicit zero-fill. Scale
        # region was never written -> write pool then pad remainder explicitly.
        end = plan[0]["scale_file_off"] + len(scale_pool)
        f.write(b"\x00" * (align_up(scale_total) - scale_total))
    # trailing sha256 over all preceding bytes
    w.f.flush()
    h = hashlib.sha256()
    with open(OUT_FILE, "rb") as f:
        while True:
            ch = f.read(1 << 20)
            if not ch:
                break
            h.update(ch)
    with open(OUT_FILE, "ab") as f:
        f.write(h.digest())
    size = os.path.getsize(OUT_FILE)
    assert size == total, (size, total)
    report = {"created": created, "file": os.path.basename(OUT_FILE), "bytes": size,
              "sha256": h.hexdigest(), "tensors": len(plan),
              "quantized_2d": sum(1 for p in plan if p["scheme"] == "int4"),
              "policy": policy["default"], "group_size": GROUP,
              "determinism": "payload+dir byte-identical given same inputs; created_date from manifest",
              "per_tensor": per_tensor}
    print("WROTE", OUT_FILE, size, "sha256", h.hexdigest()[:16], "...")
    errs = [v["max_abs_err"] for v in per_tensor.values() if "max_abs_err" in v]
    means = [v["mean_abs_err"] for v in per_tensor.values() if "mean_abs_err" in v]
    print(f"quantized: {len(errs)} tensors, worst max_abs_err={max(errs):.5f}, "
          f"worst mean_abs_err={max(means):.6f}")
    worst = sorted(((v["max_abs_err"], k) for k, v in per_tensor.items()
                    if "max_abs_err" in v), reverse=True)[:10]
    report["worst_tensors"] = [{"name": k, "max_abs_err": e} for e, k in worst]
    json.dump(report, open(REPORT_FILE, "w"), indent=1)
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
    return {"name": name, "ndim": ndim, "shape": shape, "logical": logical,
            "storage": storage, "layout": layout, "group": group,
            "sc_off": sc_off, "sc_bytes": sc_bytes, "d_off": d_off,
            "d_bytes": d_bytes, "crc": crc}


def cmd_validate(path=None):
    global ERRORS
    ERRORS = []
    path = path or OUT_FILE
    size = os.path.getsize(path)
    f = open(path, "rb")
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
    if flags & ~1:
        fail("reserved flags set")
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
    for sid in (1, 2, 3, 4, 5):
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
    # bounds + overlap over payload+scale spans
    spans = []
    for e in entries:
        numel = 1
        for d in e["shape"]:
            numel *= d
        if e["storage"] == DT["INT4_SYM_G128"]:
            if e["group"] != GROUP or e["sc_bytes"] != ((numel + GROUP - 1) // GROUP) * 2:
                fail(f"bad quant params {e['name']}")
            if e["d_bytes"] != ((numel + GROUP - 1) // GROUP) * GROUP // 2:
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
    # trailing sha
    f.seek(0)
    h = hashlib.sha256()
    left = size - 32
    while left > 0:
        ch = f.read(min(1 << 20, left))
        h.update(ch)
        left -= len(ch)
    if f.read(32) != h.digest():
        fail("trailing sha256 mismatch")
    f.close()
    if ERRORS:
        print("INVALID:")
        for m in ERRORS[:20]:
            print(" -", m)
        return 1
    print(f"VALID: {n} tensors, {size / 2**30:.2f} GiB, sha ok")
    return 0


def cmd_negatives():
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp()
    tiny = os.path.join(tmp, "tiny.binfer")
    # minimal valid file: reuse writer pieces with 2 fake tensors
    ident = section_blob({"source_repo": "t", "source_revision": "0" * 40})
    arch = section_blob({"arch": "t"})
    tok = section_blob({"format": "x"})
    pol = section_blob({"default": "y"})
    blobs = [ident, arch, tok, pol]
    names = ["a.weight", "b.weight"]
    shapes = [[256], [256]]
    stor = [DT["INT4_SYM_G128"], DT["BF16"]]
    grp = [GROUP, 0]
    sc = [4, 0]
    pay = [bytes(128), bytes(512)]
    n = 2
    off = align_up(128 + 5 * 32)
    sects = []
    for i, b in enumerate(blobs, 1):
        sects.append([i, off, 8 + len(b)])
        off = align_up(off + 8 + len(b))
    doff = off
    off = align_up(off + n * 192)
    sc_total = sum(sc)
    sc0 = off
    off = align_up(off + sc_total)
    do0 = off
    doffsets = [do0, align_up(do0 + len(pay[0]))]
    total = align_up(doffsets[1] + len(pay[1])) + 32
    # dir body (raw entries, no length prefix); payloads are fixed so CRCs are known
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
    sects.append([5, doff, len(dir_blob)])
    dir_crc = binascii.crc32(bytes(dir_blob)) & 0xFFFFFFFF

    def emit(path, mut=None):
        buf = bytearray()
        buf += MAGIC + struct.pack("<IIQQIIQ80s", VERSION, 1, n, 128, 5, ALIGN, total, b"\x00" * 80)
        for sid, soff, sb in sects:
            buf += struct.pack("<I", sid) + struct.pack("<Q", soff) + struct.pack("<Q", sb)
            if sid == 5:
                buf += struct.pack("<I", dir_crc)
            else:
                buf += struct.pack("<I", binascii.crc32(struct.pack("<Q", sb - 8) + blobs[sid - 1]) & 0xFFFFFFFF)
            buf += struct.pack("<II", 0, 0)
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
        buf += b"\x01\x02" * 2  # fake scales
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
        # fix trailing sha unless mut says skip (sha computed over final bytes pre-tail)
        if not getattr(emit, "no_fix_sha", False):
            digest = hashlib.sha256(bytes(buf)).digest()
        else:
            digest = b"\x00" * 32
        buf += digest
        open(path, "wb").write(bytes(buf))

    cases = {
        "valid": (None, True),
        "bad_magic": (lambda b: b.__setitem__(0, 0xFF), False),
        "bad_version": (lambda b: struct.pack_into("<I", b, 8, 999), False),
        "truncated": (None, False),  # handled specially
        "payload_crc": (lambda b: b.__setitem__(do0 + 3, (b[do0 + 3] + 1) % 256), False),
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
    # unknown layout + visual tensor: mutate a valid copy's dir entry
    p = os.path.join(tmp, "layout.binfer")
    emit(p)
    with open(p, "r+b") as f:
        f.seek(doff + 138)
        f.write(struct.pack("<H", 7))
    ERRORS = []
    rc = cmd_validate(p)
    print(f"{'PASS' if rc != 0 else 'FAIL'} negative/unknown_layout (expected reject)")
    ok = ok and rc != 0
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


def load_tensor_from_binfer(path, name):
    f = open(path, "rb")
    f.seek(16)  # count Q @16, table_off Q @24
    n = struct.unpack("<Q", f.read(8))[0]
    table_off = struct.unpack("<Q", f.read(8))[0]
    # dir is section id 5; scan the 5 section entries
    f.seek(table_off)
    sects = {}
    for _ in range(5):
        sid, off, nb, _, _, _ = struct.unpack("<I2Q3I", f.read(32))
        sects[sid] = off
    f.seek(sects[5])
    for _ in range(n):
        e = read_entry(f)
        if e["name"] == name:
            f.seek(e["d_off"])
            data = f.read(e["d_bytes"])
            if e["storage"] == DT["INT4_SYM_G128"]:
                f2 = open(path, "rb")
                f2.seek(e["sc_off"])
                sc = f2.read(e["sc_bytes"])
                f2.close()
                numel = 1
                for d in e["shape"]:
                    numel *= d
                arr = dequantize_int4_sym(data, sc, numel).reshape(e["shape"])
            else:
                arr = np.frombuffer(data, dtype=np.uint16).reshape(e["shape"])
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            f.close()
            return np.array(arr, dtype=np.float32)
    f.close()
    raise KeyError(name)


def cmd_mlpcheck():
    if not _need_heavy("mlpcheck"):
        return 3
    ref = json.load(open(os.path.join(REF_DIR, "mlp_layer0.json")))
    x = np.array(ref["in"], dtype=np.float32)
    P = "model.language_model.layers.0."
    normw = load_tensor_from_binfer(OUT_FILE, P + "input_layernorm.weight").reshape(-1)
    g = load_tensor_from_binfer(OUT_FILE, P + "mlp.gate_proj.weight")
    u = load_tensor_from_binfer(OUT_FILE, P + "mlp.up_proj.weight")
    d = load_tensor_from_binfer(OUT_FILE, P + "mlp.down_proj.weight")
    # Qwen3.5/3.8 RMSNorm parameters are zero-centered and applied as (1+w).
    nn = x * (1 / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6)) * (1 + normw)
    nn = nn.astype(np.float32)
    gate = nn @ g.T
    h = (gate / (1 + np.exp(-gate))) * (nn @ u.T)
    y = x + h @ d.T
    exp = np.array(ref["out"], dtype=np.float32)
    diff = np.abs(y - exp)
    print(f"mlpcheck: max_abs_diff={diff.max():.5f} mean={diff.mean():.6f} (INT4 vs BF16 source)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "quantize":
        sys.exit(cmd_quantize())
    if cmd == "validate":
        sys.exit(cmd_validate())
    if cmd == "negatives":
        sys.exit(cmd_negatives())
    if cmd == "mlpcheck":
        sys.exit(cmd_mlpcheck())
    print("unknown cmd", cmd)
    sys.exit(2)
