#!/usr/bin/env python3
"""Prototype and verify Chunked Parallel DeltaNet Scan math against serial reference."""

import numpy as np

def serial_deltanet(q, k, v, g, beta, S0):
    """
    Serial reference matching deltanet_recurrent_batch.
    q: [B, d]
    k: [B, d]
    v: [B, d]
    g: [B]
    beta: [B]
    S0: [d, d]
    Returns: out [B, d], S_final [d, d]
    """
    B, d = q.shape
    S = S0.copy()
    out = np.zeros((B, d), dtype=np.float32)
    SCALE_128 = 0.08838834764831845

    for b in range(B):
        # kv_acc for col j: S^T k
        # in matrix: kv = S.T @ k[b]
        kv = S.T @ k[b]
        kv_j = kv * g[b]
        delta = (v[b] - kv_j) * beta[b]

        # S_new = g * S_old + outer(k, delta)
        S = g[b] * S + np.outer(k[b], delta)

        # out = S.T @ q * SCALE
        o = S.T @ q[b]
        out[b] = o * SCALE_128

    return out, S

def chunked_deltanet(q, k, v, g, beta, S0, C=16):
    """
    Chunked formulation with chunk size C.
    """
    B, d = q.shape
    assert B % C == 0
    num_chunks = B // C
    out = np.zeros((B, d), dtype=np.float32)
    SCALE_128 = 0.08838834764831845

    S = S0.copy()

    for c in range(num_chunks):
        t0 = c * C
        t1 = t0 + C
        qc = q[t0:t1]       # [C, d]
        kc = k[t0:t1]       # [C, d]
        vc = v[t0:t1]       # [C, d]
        gc = g[t0:t1]       # [C]
        bc = beta[t0:t1]    # [C]

        # Compute cumulative decay within chunk:
        # gamma[t] = prod_{i=0}^t gc[i]
        # decay from start of chunk to step t (inclusive):
        gamma_from_0 = np.cumprod(gc) # gamma_from_0[t] = prod_{i=0}^t gc[i]

        # gamma_step[t, s] = prod_{i=s+1}^t gc[i] for s < t, 1 for s == t
        # Using log/cumsum for numerical stability or direct cumprod ratio:
        gamma_mat = np.zeros((C, C), dtype=np.float32)
        for t in range(C):
            for s in range(t + 1):
                if s == t:
                    gamma_mat[t, s] = 1.0
                else:
                    gamma_mat[t, s] = gamma_from_0[t] / gamma_from_0[s]

        # Compute strictly lower-triangular M matrix:
        # G[t, s] = gamma_mat[t, s+1] * (kc[t] @ kc[s]) for s < t
        # M = diag(1/beta) + G
        KKT = kc @ kc.T # [C, C]
        M = np.zeros((C, C), dtype=np.float32)
        for t in range(C):
            M[t, t] = 1.0 / bc[t]
            for s in range(t):
                # decay from s+1 to t:
                # gamma_mat[t, s+1] is prod_{i=s+1}^t gc[i]
                decay_s_plus_1_to_t = gamma_from_0[t] / gamma_from_0[s]
                # wait! in serial loop:
                # delta_t = beta_t * (v_t - g_t * kv_t)
                # kv_t has decay g_t.
                # Let's check exact formula!
                M[t, s] = decay_s_plus_1_to_t * KKT[t, s]

        # V_init: initial state effect on each token
        # V_init[t] = gamma_from_0[t] * (S.T @ kc[t])
        V_init = np.zeros((C, d), dtype=np.float32)
        for t in range(C):
            V_init[t] = gamma_from_0[t] * (S.T @ kc[t])

        # Solve M @ D = vc - V_init
        RHS = vc - V_init
        # Since M is lower triangular, we can solve by forward substitution
        # or D = inv(M) @ RHS
        D = np.linalg.solve(M, RHS) # [C, d]

        # Output computation for each token t in chunk:
        # out[t] = gamma_from_0[t] * (S.T @ qc[t]) + sum_{s=0}^t gamma_mat[t, s+1] * (qc[t] @ kc[s]) * D[s]
        # In matrix:
        # QKT = qc @ kc.T # [C, C]
        QKT = qc @ kc.T
        A_intra = np.zeros((C, C), dtype=np.float32)
        for t in range(C):
            for s in range(t + 1):
                A_intra[t, s] = gamma_mat[t, s] * QKT[t, s] # wait, check decay index for s == t vs s < t

        # Let's verify each token's out[t]
        for t in range(C):
            o_state = gamma_from_0[t] * (S.T @ qc[t])
            o_intra = np.zeros(d, dtype=np.float32)
            for s in range(t + 1):
                # for s, what is the weight of D[s]?
                # At step s: S_s has + kc[s] outer D[s]
                # At step t: that state is decayed by prod_{i=s+1}^t gc[i]
                decay = gamma_mat[t, s] # prod_{i=s+1}^t gc[i]
                o_intra += decay * (qc[t] @ kc[s]) * D[s]
            out[t0 + t] = (o_state + o_intra) * SCALE_128

        # Update final state at end of chunk:
        # S_end = gamma_from_0[C-1] * S + sum_{s=0}^{C-1} (decay to C-1) * (kc[s] outer D[s])
        S_new = gamma_from_0[C - 1] * S
        for s in range(C):
            decay = gamma_mat[C - 1, s]
            S_new += decay * np.outer(kc[s], D[s])
        S = S_new

    return out, S

def main():
    np.random.seed(42)
    B = 32
    d = 128
    C = 16

    q = np.random.randn(B, d).astype(np.float32)
    k = np.random.randn(B, d).astype(np.float32)
    v = np.random.randn(B, d).astype(np.float32)
    # Normalize q and k as in model (head l2 norm)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)

    g = np.random.uniform(0.8, 0.99, size=B).astype(np.float32)
    beta = np.random.uniform(0.1, 0.9, size=B).astype(np.float32)
    S0 = np.random.randn(d, d).astype(np.float32) * 0.01

    out_ser, S_ser = serial_deltanet(q, k, v, g, beta, S0)
    out_chk, S_chk = chunked_deltanet(q, k, v, g, beta, S0, C=C)

    out_diff = np.max(np.abs(out_ser - out_chk))
    s_diff = np.max(np.abs(S_ser - S_chk))

    print(f"Max out diff: {out_diff:.6e}")
    print(f"Max State diff: {s_diff:.6e}")
    if out_diff < 1e-4 and s_diff < 1e-4:
        print("SUCCESS! Chunked math matches serial recurrence!")
    else:
        print("MISMATCH! Need to refine decay indices.")

if __name__ == "__main__":
    main()
