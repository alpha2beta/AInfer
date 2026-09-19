"""T3.6 C++ Level Zero loader negative suite: malformed-input + MoE rejection.

Builds minimal synthetic .binfer files (including Section 6 MoE metadata)
and runs the `l0load` binary against each mutant.
Exit 0 iff all cases behave as expected (reject malformed input cleanly before device allocation).
"""
import binascii
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile

MAGIC = b"BINFER\x00\x01"
VERSION = 1
ALIGN = 64


def align_up(n, a=ALIGN):
    return (n + a - 1) // a * a


def build(path):
    """Minimal valid MoE .binfer file. Returns dict of patch points."""
    n = 1
    flags = 3  # text + MoE
    scount = 6
    blobs = [b"I" * 16, b"A" * 16, b"T" * 16, b"Q" * 16]

    # Section 6 MoE metadata
    moe_hdr = struct.pack("<IIIIIBBHI36s", 256, 8, 512, 512, 1, 2, 1, 0, 40, b"\x00" * 36)
    # 40 layer descriptors pointing to tensor 0
    moe_descs = struct.pack("<IIIIIIII", 0, 0, 0, 0, 0, 0, 0, 0)
    for l in range(1, 40):
        moe_descs += struct.pack("<IIIIIIII", l, 0, 0, 0, 0, 0, 0, 0)
    moe_blob = moe_hdr + moe_descs

    pay = b"\xab" * 256
    sc = b"\x01\x02" * 2
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

    sc0 = off
    off = align_up(off + len(sc))
    do0 = off
    total = align_up(do0 + len(pay)) + 32

    nb = b"t0\x00"
    e = (nb + b"\x00" * (64 - len(nb)) + struct.pack("<B", 2) + b"\x00" * 7
         + struct.pack("<8Q", 16, 16, 0, 0, 0, 0, 0, 0)
         + struct.pack("<B", 0) + struct.pack("<B", 3)
         + struct.pack("<H", 0) + struct.pack("<I", 128)
         + struct.pack("<Q", sc0) + struct.pack("<Q", len(sc))
         + struct.pack("<Q", do0) + struct.pack("<Q", len(pay))
         + struct.pack("<I", binascii.crc32(pay) & 0xFFFFFFFF) + b"\x00" * 12)
    assert len(e) == 192
    dir_blob = bytearray(e)
    sects.append([5, doff, len(dir_blob)])
    dir_crc = binascii.crc32(bytes(dir_blob)) & 0xFFFFFFFF

    sects.append([6, moe_off, moe_bytes])
    moe_crc = binascii.crc32(struct.pack("<Q", len(moe_blob)) + moe_blob) & 0xFFFFFFFF

    buf = bytearray()
    buf += MAGIC + struct.pack("<IIQQIIQ80s", VERSION, flags, n, 128, scount, ALIGN,
                               total, b"\x00" * 80)
    for sid, soff, sb in sects:
        buf += struct.pack("<I", sid) + struct.pack("<Q", soff) + struct.pack("<Q", sb)
        if sid == 5:
            buf += struct.pack("<I", dir_crc)
        elif sid == 6:
            buf += struct.pack("<I", moe_crc)
        else:
            b = blobs[sid - 1]
            buf += struct.pack("<I", binascii.crc32(struct.pack("<Q", len(b)) + b) & 0xFFFFFFFF)
        buf += struct.pack("<II", 0, 0)

    while len(buf) < sects[0][1]:
        buf += b"\x00"
    for (sid, soff, sb), b in zip(sects[:4], blobs):
        assert len(buf) == soff
        buf += struct.pack("<Q", len(b)) + b
        while len(buf) % 64:
            buf += b"\x00"
    assert len(buf) == doff
    dir_pos = len(buf)
    buf += bytes(dir_blob)
    while len(buf) % 64:
        buf += b"\x00"
    assert len(buf) == moe_off
    moe_pos = len(buf)
    buf += struct.pack("<Q", len(moe_blob)) + moe_blob
    while len(buf) % 64:
        buf += b"\x00"
    assert len(buf) == sc0
    buf += sc
    while len(buf) % 64:
        buf += b"\x00"
    assert len(buf) == do0
    buf += pay
    while len(buf) % 64:
        buf += b"\x00"

    with open(path, "wb") as f:
        f.write(bytes(buf))

    return {
        "dir_pos": dir_pos,
        "moe_pos": moe_pos,
        "layout_pos": dir_pos + 138,
        "doff_pos": dir_pos + 160,
        "dlen_pos": dir_pos + 168,
        "moe_sect_crc_pos": 128 + 5 * 32 + 20,
    }


def emit(base, name, mut=None, fix_crc=True, truncate=False):
    p = os.path.join(base, name + ".binfer")
    pts = build(p)
    with open(p, "rb") as f:
        buf = bytearray(f.read())
    if mut:
        mut(buf, pts)
    if truncate:
        buf = buf[:len(buf) // 2]
        with open(p, "wb") as f:
            f.write(bytes(buf))
        return p
    if fix_crc:
        # Recompute dir CRC
        e = bytes(buf[pts["dir_pos"]:pts["dir_pos"] + 192])
        crc = binascii.crc32(e) & 0xFFFFFFFF
        struct.pack_into("<I", buf, 128 + 4 * 32 + 20, crc)
        # Recompute MoE CRC
        mlen = struct.unpack_from("<Q", buf, pts["moe_pos"])[0]
        mblob = bytes(buf[pts["moe_pos"] + 8:pts["moe_pos"] + 8 + mlen])
        mcrc = binascii.crc32(struct.pack("<Q", mlen) + mblob) & 0xFFFFFFFF
        struct.pack_into("<I", buf, pts["moe_sect_crc_pos"], mcrc)

        digest = hashlib.sha256(bytes(buf)).digest()
        buf = buf + digest
    else:
        buf = buf + b"\x00" * 32
    with open(p, "wb") as f:
        f.write(bytes(buf))
    return p


def main():
    if len(sys.argv) < 2:
        print("usage: negatives.py <l0load-binary> [report.json]")
        return 2
    loader = sys.argv[1]
    base = tempfile.mkdtemp(prefix="l0neg_")
    fsz = os.path.getsize(emit(base, "probe"))

    cases = [
        ("valid", {}, 0),
        ("bad_magic", {"mut": lambda b, p: b.__setitem__(0, 0xFF)}, 2),
        ("bad_version", {"mut": lambda b, p: struct.pack_into("<I", b, 8, 999)}, 2),
        ("truncated", {"truncate": True}, 2),
        ("dir_crc", {"fix_crc": False,
                     "mut": lambda b, p: b.__setitem__(p["dir_pos"] + 5, 0xFF)}, 2),
        ("unknown_layout", {"mut": lambda b, p: struct.pack_into("<H", b, p["layout_pos"], 7)}, 2),
        ("payload_oob", {"mut": lambda b, p: struct.pack_into("<Q", b, p["doff_pos"], fsz + 4096)}, 2),
        ("payload_len_oob", {"mut": lambda b, p: struct.pack_into("<Q", b, p["dlen_pos"], fsz)}, 2),
        ("bad_moe_crc", {"fix_crc": False,
                         "mut": lambda b, p: b.__setitem__(p["moe_pos"] + 16, (b[p["moe_pos"] + 16] + 1) % 256)}, 2),
        ("bad_expert_count", {"mut": lambda b, p: struct.pack_into("<I", b, p["moe_pos"] + 8, 999)}, 2),
        ("bad_layer_count", {"mut": lambda b, p: struct.pack_into("<I", b, p["moe_pos"] + 8 + 24, 999)}, 2),
        ("bad_descriptor_index", {"mut": lambda b, p: struct.pack_into("<I", b, p["moe_pos"] + 8 + 64 + 4, 99999)}, 2),
    ]

    env = os.environ.copy()
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "toolchain/sysroot/usr/lib")
    if os.path.isdir(lib_path):
        env["LD_LIBRARY_PATH"] = lib_path + ":" + env.get("LD_LIBRARY_PATH", "")

    results, ok = {}, True
    for name, kw, expect in cases:
        mut = kw.get("mut")
        p = emit(base, name, mut=mut, fix_crc=kw.get("fix_crc", True),
                 truncate=kw.get("truncate", False))
        r = subprocess.run([loader, p], capture_output=True, text=True,
                           timeout=60, env=env)
        good = (r.returncode == expect) if expect == 0 else (r.returncode != 0)
        results[name] = {"rc": r.returncode, "expect": expect,
                         "pass": good,
                         "diag": (r.stderr.strip().splitlines() or [""])[-1][:100]}
        print(f"{'PASS' if good else 'FAIL'} l0neg/{name}: rc={r.returncode} "
              f"(expect {'0' if expect == 0 else 'nonzero'})", flush=True)
        ok = ok and good

    rep = {"tool": "l0load negatives", "cases": results, "all_pass": ok}
    if len(sys.argv) > 2:
        json.dump(rep, open(sys.argv[2], "w"), indent=1)
    else:
        print(json.dumps(rep, indent=1))
    import shutil
    shutil.rmtree(base, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
