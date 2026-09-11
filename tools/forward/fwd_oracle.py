"""T4.2 numerical oracle: CPU loop-decode with EXACTLY the device policy.

Device policy (from tools/decode/decode.cpp):
- weights: INT4 sym g128 (nibbles) + BF16 scales, read from .binfer
- activations: per-tensor INT8, sq=max|x|/127 (fp32), round-half-away, [-127,127]
- GEMV: int64 group dots (nib*iq), fp32 accumulate across groups in order
- RMSNorm: float64 accum, float32 inv, y=x*inv*(1+w)
- l2norm (Q/K): float32 accum, q scaled by 1/sqrt(128) as 0.0883883476f
- RoPE: NeoX pairs (d,d+32), d<32, first 64 dims only; tables = cos/sin float32
  of double-precision angles, theta=1e7
- GQA: fp32 dots in d-order, /16, stable softmax, sigmoid gate
- linear: conv from persistent 3-state (zero-init) + silu; recurrent from
  persistent S (zero-init) with exp(g)/beta; norm-gated with w DIRECT
- beta=sigmoid(b); g=-exp(Alog)*softplus(a+dt), softplus branch sa>20
- final norm (1+w), lm_head, argmax first-max wins ties

Teacher-forces device ids, compares logits/top5 per generate step.
Usage: fwd_oracle.py [report.json]
"""
import json
import os
import struct
import sys
import time

import numpy as np

REPO = "/mnt/usb/AInfer"
BINFER = os.path.join(REPO, "models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer")
REF = os.path.join(REPO, "reference")
sys.path.insert(0, os.path.join(REPO, "tools"))
from binfer import DT  # noqa: E402

PROMPT = [248045, 846, 198, 3710]
DEVICE_GEN = json.load(open("/tmp/decode_oracle.json"))["generated"]
IDS = PROMPT + DEVICE_GEN[:3]  # teacher-force steps 0..6
NSTEPS = len(IDS)
GEN_STEPS = [3, 4, 5, 6]
DEV_TOP5 = {r["step"]: (r["top5"], r["top5v"])
            for r in json.load(open("/tmp/decode_oracle.json"))["top5_per_step"]}
F32 = np.float32


def bf16_to_f32(b):
    return (np.ascontiguousarray(b, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


class Binfer:
    def __init__(self, path):
        self.f = open(path, "rb")
        f = self.f
        f.seek(16)
        n = struct.unpack("<Q", f.read(8))[0]
        table_off = struct.unpack("<Q", f.read(8))[0]
        f.seek(table_off)
        sects = {}
        for _ in range(5):
            sid, off, nb, _, _, _ = struct.unpack("<I2Q3I", f.read(32))
            sects[sid] = off
        f.seek(sects[5])
        self.entries = {}
        for _ in range(n):
            raw = f.read(192)
            name = raw[:64].split(b"\x00")[0].decode()
            ndim = raw[64]
            shape = list(struct.unpack("<8Q", raw[72:136]))[:ndim]
            storage = raw[137]
            sc_off, sc_b, d_off, d_b = struct.unpack("<4Q", raw[144:176])
            self.entries[name] = (shape, storage, sc_off, sc_b, d_off, d_b)

    def raw(self, name):
        shape, storage, sc_off, sc_b, d_off, d_b = self.entries[name]
        f = self.f
        f.seek(d_off)
        data = f.read(d_b)
        sc = None
        if sc_b:
            f.seek(sc_off)
            sc = f.read(sc_b)
        return shape, storage, data, sc

    def int4(self, name):
        """-> (nibbles int8 [M,K], scales f32 [M,G])."""
        shape, storage, data, sc = self.raw(name)
        assert storage == DT["INT4_SYM_G128"], (name, storage)
        M, K = shape
        G = K // 128
        raw = np.frombuffer(data, dtype=np.uint8)
        nib = np.empty(raw.size * 2, dtype=np.int8)
        nib[0::2] = raw & np.uint8(0xF)
        nib[1::2] = (raw >> np.uint8(4)) & np.uint8(0xF)
        nib[nib >= 8] -= 16
        scales = bf16_to_f32(np.frombuffer(sc, dtype=np.uint16)).astype(F32)
        return nib.reshape(M, K).astype(np.int8), scales.reshape(M, G)

    def bf16(self, name):
        shape, storage, data, _ = self.raw(name)
        assert storage == DT["BF16"], (name, storage)
        return bf16_to_f32(np.frombuffer(data, dtype=np.uint16)).astype(
            F32).reshape(shape)


BF = Binfer(BINFER)
_WCACHE = {}


def preload_weights():
    names = ["model.language_model.embed_tokens.weight",
             "model.language_model.norm.weight", "lm_head.weight"]
    for L in range(64):
        P = f"model.language_model.layers.{L}."
        names += [P + "input_layernorm.weight",
                  P + "post_attention_layernorm.weight",
                  P + "mlp.gate_proj.weight", P + "mlp.up_proj.weight",
                  P + "mlp.down_proj.weight"]
        if is_full(L):
            names += [P + f"self_attn.{p}.weight" for p in
                      ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
        else:
            names += [P + f"linear_attn.{p}.weight" for p in
                      ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                       "conv1d", "out_proj", "norm")]
            names += [P + "linear_attn.dt_bias", P + "linear_attn.A_log"]
    for n in names:
        shape, storage = BF.entries[n][:2]
        if storage == DT["INT4_SYM_G128"]:
            _WCACHE[n] = ("int4",) + _orig_int4(BF, n)
        else:
            _WCACHE[n] = ("bf16", _orig_bf16(BF, n))
    print(f"preloaded {len(_WCACHE)} tensors", flush=True)


_orig_int4 = Binfer.int4
_orig_bf16 = Binfer.bf16


def _cached_int4(self, name):
    return _WCACHE[name][1], _WCACHE[name][2]


def _cached_bf16(self, name):
    return _WCACHE[name][1]


Binfer.int4 = _cached_int4
Binfer.bf16 = _cached_bf16

MAXCTX = 16
COS = np.zeros((MAXCTX, 64), F32)
SIN = np.zeros((MAXCTX, 64), F32)
for t in range(MAXCTX):
    for i in range(64):
        inv = 1.0 / (10000000.0 ** ((2 * (i % 32)) / 64.0))
        COS[t, i] = np.float32(np.cos(t * inv))
        SIN[t, i] = np.float32(np.sin(t * inv))


def rmsnorm_1pw(x, w):
    ss = np.sum(x.astype(np.float64) ** 2)
    inv = np.float32(1.0 / np.sqrt(np.float32(ss / x.size) + np.float32(1e-6)))
    return (x * inv * (1.0 + w)).astype(F32)


def quant_act(x):
    sq = np.float32(np.max(np.abs(x)) / np.float32(127.0))
    q = np.where(x >= 0, x / sq + 0.5, x / sq - 0.5).astype(np.int32)
    return np.clip(q, -127, 127).astype(np.int8), sq


def gemv_int4(nib, sc, xq, sq, mchunk=32768):
    M, K = nib.shape
    G = K // 128
    xqg = xq.reshape(G, 128).astype(np.int64)
    out = np.empty(M, F32)
    for m0 in range(0, M, mchunk):
        blk = nib[m0:m0 + mchunk].reshape(-1, G, 128).astype(np.int64)
        isum = np.einsum("mgj,gj->mg", blk, xqg)
        out[m0:m0 + mchunk] = (isum.astype(F32) * sc[m0:m0 + mchunk] *
                               np.float32(sq)).astype(F32).sum(axis=1)
        del blk, isum
    return out


def is_full(L):
    return (L % 4) == 3


def layer_weights(L):
    P = f"model.language_model.layers.{L}."
    W = {"ln1": BF.bf16(P + "input_layernorm.weight").reshape(-1),
         "ln2": BF.bf16(P + "post_attention_layernorm.weight").reshape(-1),
         "gate": BF.int4(P + "mlp.gate_proj.weight"),
         "up": BF.int4(P + "mlp.up_proj.weight"),
         "down": BF.int4(P + "mlp.down_proj.weight")}
    if is_full(L):
        W["q"] = BF.int4(P + "self_attn.q_proj.weight")
        W["k"] = BF.int4(P + "self_attn.k_proj.weight")
        W["v"] = BF.int4(P + "self_attn.v_proj.weight")
        W["o"] = BF.int4(P + "self_attn.o_proj.weight")
        W["qn"] = BF.bf16(P + "self_attn.q_norm.weight").reshape(-1)
        W["kn"] = BF.bf16(P + "self_attn.k_norm.weight").reshape(-1)
    else:
        W["qkv"] = BF.int4(P + "linear_attn.in_proj_qkv.weight")
        W["z"] = BF.int4(P + "linear_attn.in_proj_z.weight")
        W["b"] = BF.int4(P + "linear_attn.in_proj_b.weight")
        W["a"] = BF.int4(P + "linear_attn.in_proj_a.weight")
        W["conv"] = BF.bf16(P + "linear_attn.conv1d.weight").reshape(10240, 4)
        W["dt"] = BF.bf16(P + "linear_attn.dt_bias").reshape(-1)
        W["alog"] = BF.bf16(P + "linear_attn.A_log").reshape(-1)
        W["nw"] = BF.bf16(P + "linear_attn.norm.weight").reshape(-1)
        W["o"] = BF.int4(P + "linear_attn.out_proj.weight")
    return W


def decode_step(x, L, pos, kv, conv, S, W=None):
    """x: [5120] fp32. kv/conv/S persistent dicts keyed by slot. Returns new x."""
    P = f"model.language_model.layers.{L}."
    h = rmsnorm_1pw(x, W["ln1"])
    xq, sq = quant_act(h)
    if is_full(L):
        slot = L // 4
        q = gemv_int4(*W["q"], xq, sq)
        k = gemv_int4(*W["k"], xq, sq)
        v = gemv_int4(*W["v"], xq, sq)
        C = q.reshape(24, 512)
        content, gate = C[:, :256].copy(), C[:, 256:].copy()
        qn_w, kn_w = W["qn"], W["kn"]
        content = rmsnorm_1pw(content.reshape(-1), np.tile(qn_w, 24)).reshape(24, 256)
        kk = rmsnorm_1pw(k.reshape(-1), np.tile(kn_w, 4)).reshape(4, 256)
        c, s = COS[pos], SIN[pos]
        x0 = content[:, :32].copy()
        x1 = content[:, 32:64].copy()
        content[:, :32] = x0 * c[:32] - x1 * s[:32]
        content[:, 32:64] = x0 * s[:32] + x1 * c[:32]
        x0 = kk[:, :32].copy()
        x1 = kk[:, 32:64].copy()
        kk[:, :32] = x0 * c[:32] - x1 * s[:32]
        kk[:, 32:64] = x0 * s[:32] + x1 * c[:32]
        K = kv["K"]  # [16,MAXCTX,4,256]
        V = kv["V"]
        K[slot, pos] = kk
        V[slot, pos] = v.reshape(4, 256)
        out = np.zeros((24, 256), F32)
        for hh in range(24):
            kvh = hh // 6
            sc = np.zeros(pos + 1, F32)
            for t in range(pos + 1):
                sc[t] = np.float32(np.sum(
                    content[hh].astype(np.float64) * K[slot, t, kvh].astype(np.float64)) / 16.0)
            mx = sc.max()
            w = np.exp(sc - mx).astype(F32)
            w /= w.sum()
            acc = np.zeros(256, F32)
            for t in range(pos + 1):
                acc += np.float32(w[t]) * V[slot, t, kvh]
            out[hh] = acc / (1.0 + np.exp(-gate[hh])).astype(F32)
        o_nib, o_sc = W["o"]
        oq, osq = quant_act(out.reshape(-1))
        mix = gemv_int4(o_nib, o_sc, oq, osq)
    else:
        sl = L - (L + 1) // 4
        qkv = gemv_int4(*W["qkv"], xq, sq)
        z = gemv_int4(*W["z"], xq, sq)
        b = gemv_int4(*W["b"], xq, sq)
        a = gemv_int4(*W["a"], xq, sq)
        cw = W["conv"]
        cs = conv[sl]  # [10240,3]
        acc = cs[:, 0] * cw[:, 0] + cs[:, 1] * cw[:, 1] + cs[:, 2] * cw[:, 2] \
            + qkv * cw[:, 3]
        mx = acc / (1.0 + np.exp(-acc)).astype(F32)
        cs[:, 0] = cs[:, 1]
        cs[:, 1] = cs[:, 2]
        cs[:, 2] = qkv
        q16 = mx[:2048].reshape(16, 128)
        k16 = mx[2048:4096].reshape(16, 128)
        vv = mx[4096:].reshape(48, 128)
        qq = np.repeat(q16, 3, axis=0)
        kk = np.repeat(k16, 3, axis=0)
        qn = (qq / np.sqrt((qq.astype(F32) ** 2).sum(-1, keepdims=True) + 1e-6)
              * np.float32(0.0883883476)).astype(F32)
        kn = (kk / np.sqrt((kk.astype(F32) ** 2).sum(-1, keepdims=True) + 1e-6)).astype(F32)
        alog, dt = W["alog"], W["dt"]
        beta = (1.0 / (1.0 + np.exp(-b))).astype(F32)
        sa = a + dt
        soft = np.where(sa > 20, sa, np.log1p(np.exp(sa))).astype(F32)
        gg = (-np.exp(alog) * soft).astype(F32)
        S0 = S[sl]  # [48,128,128]
        S0 *= np.exp(gg)[:, None, None]
        core = np.zeros((48, 128), F32)
        for hh in range(48):
            kvm = (S0[hh] * kk[hh][:, None]).sum(axis=0)
            dl = (vv[hh] - kvm) * beta[hh]
            S0[hh] += kk[hh][:, None] * dl[None, :]
            core[hh] = (S0[hh] * qn[hh][:, None]).sum(axis=0)
        nw = W["nw"]
        ss = (core.astype(F32) ** 2).sum(-1, keepdims=True)
        gated = (nw * core * (1.0 / np.sqrt(ss + 1e-6))).astype(F32) * (
            z.reshape(48, 128) / (1.0 + np.exp(-z.reshape(48, 128)))).astype(F32)
        o_nib, o_sc = W["o"]
        oq, osq = quant_act(gated.reshape(-1))
        mix = gemv_int4(o_nib, o_sc, oq, osq)
    h2 = x + mix
    h3 = rmsnorm_1pw(h2, W["ln2"])
    mq, msq = quant_act(h3)
    g = gemv_int4(*W["gate"], mq, msq)
    u = gemv_int4(*W["up"], mq, msq)
    fu = (g / (1.0 + np.exp(-g))) * u
    fq, fsq = quant_act(fu.astype(F32))
    dn = gemv_int4(*W["down"], fq, fsq)
    return h2 + dn


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/oracle.json"
    t_all = time.time()
    preload_weights()
    kv = {"K": np.zeros((16, MAXCTX, 4, 256), F32),
          "V": np.zeros((16, MAXCTX, 4, 256), F32)}
    conv = {sl: np.zeros((10240, 3), F32) for sl in range(48)}
    S = {sl: np.zeros((48, 128, 128), F32) for sl in range(48)}
    emb = BF.bf16("model.language_model.embed_tokens.weight")
    results = []
    xs = {step: emb[IDS[step]].astype(F32).copy() for step in range(NSTEPS)}
    del emb
    ostates = {}
    ostates0 = {}
    for L in range(64):
        t0 = time.time()
        W = layer_weights(L)
        for step in range(NSTEPS):
            xs[step] = decode_step(xs[step], L, step, kv, conv, S, W)
        del W
        if 3 in GEN_STEPS or True:
            ostates[L] = xs[3].copy()
        ostates0[L] = xs[0].copy()
        if (L + 1) % 16 == 0:
            print(f"  layer {L + 1}/64 ({time.time() - t0:.0f}s/layer)", flush=True)
    np.save("/tmp/ostates.npy", np.stack([ostates[L] for L in range(64)]))
    np.save("/tmp/ostates_s0.npy", np.stack([ostates0[L] for L in range(64)]))
    print("saved /tmp/ostates.npy + ostates_s0.npy", flush=True)
    fmn = BF.bf16("model.language_model.norm.weight").reshape(-1)
    lm_nib, lm_sc = BF.int4("lm_head.weight")
    for step in GEN_STEPS:
        x = xs[step]
        h = rmsnorm_1pw(x, fmn)
        hq, hsq = quant_act(h)
        logits = gemv_int4(lm_nib, lm_sc, hq, hsq)
        top5 = np.argsort(-logits)[:5]
        results.append({"step": step,
                        "top5": [int(i) for i in top5],
                        "top5v": [round(float(logits[i]), 4) for i in top5]})
        print(f"oracle step {step}: top5 {results[-1]['top5']} "
              f"v={[round(float(logits[i]), 3) for i in top5]}",
              flush=True)
    rep = {"policy": "INT4-g128 + per-tensor INT8 act, loop-decode, teacher-forced",
           "ids": IDS, "seconds": round(time.time() - t_all, 1),
           "results": results}
    json.dump(rep, open(out_path, "w"), indent=1)
    print(f"WROTE {out_path} in {rep['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
