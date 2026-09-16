"""T6.3 INT8-KV diagnostic (CPU, real K/V from kvcap.npz).

Q: does per-TOKEN symmetric INT8 on K and V (dynamic row scales, uniform
kernels, no calibration) keep attention outputs within budget?
For each full layer: quantize K/V rows (scale = max|row|/127), run the
decode-style causal attention (all T queries over 0..t) vs the BF16-stored
reference (qb), report worst output rel err + the implied LSB.
Bar: worst-rel <= 5e-3 (fraction of the 1-2% system envelope).
"""
import numpy as np

D = np.load("/tmp/kvcap.npz")
VLOSS = []


def q8(row):
    mx = np.abs(row).max(axis=-1, keepdims=True)
    mx = np.where(mx == 0, 1.0, mx)
    q = np.clip(np.rint(row / mx * 127), -127, 127) / 127 * mx
    return q.astype(np.float64), float(mx.max())


def main():
    worst = 0.0
    for L in sorted({int(k[1:]) for k in D.files if k.startswith("K")}):
        K = D[f"K{L}"][0].astype(np.float64)  # (4, T, 256)
        V = D[f"V{L}"][0].astype(np.float64)
        NK, T, DD = K.shape
        Kq = np.stack([q8(K[h])[0] for h in range(NK)])
        Vq = np.stack([q8(V[h])[0] for h in range(NK)])
        # random queries per head per t (decode-style, causal)
        rng = np.random.default_rng(1000 + L)
        lw = 0.0
        for h in range(24):
            kv = h // 6
            Q = rng.standard_normal((T, DD))
            for t in range(T):
                s = (Q[t] * Kq[kv, :t + 1]).sum(axis=-1) / 16.0
                s0 = (Q[t] * K[kv, :t + 1]).sum(axis=-1) / 16.0
                w = np.exp(s - s.max())
                w0 = np.exp(s0 - s0.max())
                o = (w / w.sum()) @ Vq[kv, :t + 1]
                o0 = (w0 / w0.sum()) @ V[kv, :t + 1]
                denom = np.abs(o0).max()
                r = np.abs(o - o0).max() / (denom if denom > 0 else 1)
                lw = max(lw, r)
        worst = max(worst, lw)
        print(f"L{L}: worst-rel {lw:.2e}", flush=True)
    print(f"DIAG_KVINT8 worst {worst:.2e} -> "
          f"{'GO per-token' if worst <= 5e-3 else 'NEEDS per-channel-K or residual'}",
          flush=True)


if __name__ == "__main__":
    main()
