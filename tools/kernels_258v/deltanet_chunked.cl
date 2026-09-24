// deltanet_chunked.cl — DeltaNet Recurrent vs Chunked Parallel Scan Kernels

#define S_V 128
#define H_V 32
#define H_K 16
#define C_QKV 8192
#define SCALE_128 0.08838834764831845f

// =========================================================================
// DeltaNet Recurrence v2 (Current Baseline)
// =========================================================================
__kernel void deltanet_recurrent_batch_v2(
    __global float * restrict out,
    __global float * restrict state,
    __global const float * restrict q,
    __global const float * restrict k,
    __global const float * restrict v,
    __global const float * restrict g,
    __global const float * restrict beta,
    int B
) {
    int h = get_group_id(0);
    if (h >= H_V) return;
    int j = get_local_id(0);

    __local float s_q[2][4][S_V];
    __local float s_k[2][4][S_V];

    int kh = h / 2;
    __global float * S_h = state + (size_t)h * (S_V * S_V);

    float s_col[S_V];
    #pragma unroll 2
    for (int i = 0; i < S_V; ++i) {
        s_col[i] = S_h[i * S_V + j];
    }

    int cur = 0;
    int steps0 = min(4, B);
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        if (s < steps0) {
            s_q[cur][s][j] = q[(size_t)s * (H_K * S_V) + kh * S_V + j];
            s_k[cur][s][j] = k[(size_t)s * (H_K * S_V) + kh * S_V + j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int b0 = 0; b0 < B; b0 += 4) {
        int steps = min(4, B - b0);
        int nxt = cur ^ 1;

        #pragma unroll 1
        for (int s = 0; s < 4; ++s) {
            if (s < steps) {
                int b = b0 + s;
                float v_val = v[(size_t)b * C_QKV + 4096 + h * S_V + j];
                float g_val = g[(size_t)b * H_V + h];
                float b_val = beta[(size_t)b * H_V + h];

                float kv0 = 0.0f, kv1 = 0.0f, kv2 = 0.0f, kv3 = 0.0f;
                #pragma unroll 2
                for (int i = 0; i < S_V; i += 4) {
                    kv0 += s_col[i]     * s_k[cur][s][i];
                    kv1 += s_col[i + 1] * s_k[cur][s][i + 1];
                    kv2 += s_col[i + 2] * s_k[cur][s][i + 2];
                    kv3 += s_col[i + 3] * s_k[cur][s][i + 3];
                }
                float kv_acc = (kv0 + kv1) + (kv2 + kv3);
                float kv_j = kv_acc * g_val;
                float delta_j = (v_val - kv_j) * b_val;

                float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
                #pragma unroll 2
                for (int i = 0; i < S_V; i += 4) {
                    float s_old0 = s_col[i];
                    float s_new0 = g_val * s_old0 + s_k[cur][s][i] * delta_j;
                    s_col[i] = s_new0;
                    o0 += s_new0 * s_q[cur][s][i];

                    float s_old1 = s_col[i + 1];
                    float s_new1 = g_val * s_old1 + s_k[cur][s][i + 1] * delta_j;
                    s_col[i + 1] = s_new1;
                    o1 += s_new1 * s_q[cur][s][i + 1];

                    float s_old2 = s_col[i + 2];
                    float s_new2 = g_val * s_old2 + s_k[cur][s][i + 2] * delta_j;
                    s_col[i + 2] = s_new2;
                    o2 += s_new2 * s_q[cur][s][i + 2];

                    float s_old3 = s_col[i + 3];
                    float s_new3 = g_val * s_old3 + s_k[cur][s][i + 3] * delta_j;
                    s_col[i + 3] = s_new3;
                    o3 += s_new3 * s_q[cur][s][i + 3];
                }
                float o_acc = (o0 + o1) + (o2 + o3);
                out[(size_t)b * (H_V * S_V) + h * S_V + j] = o_acc * SCALE_128;
            }
        }
        if (b0 + 4 < B) {
            int steps_n = min(4, B - b0 - 4);
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                if (s < steps_n) {
                    int b = b0 + 4 + s;
                    s_q[nxt][s][j] = q[(size_t)b * (H_K * S_V) + kh * S_V + j];
                    s_k[nxt][s][j] = k[(size_t)b * (H_K * S_V) + kh * S_V + j];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            cur = nxt;
        }
    }

    #pragma unroll 2
    for (int i = 0; i < S_V; ++i) {
        S_h[i * S_V + j] = s_col[i];
    }
}

// =========================================================================
// DeltaNet Chunked Parallel Scan Batch (I2.3)
// =========================================================================
__kernel void deltanet_chunked_batch(
    __global float * restrict out,
    __global float * restrict state,
    __global const float * restrict q,
    __global const float * restrict k,
    __global const float * restrict v,
    __global const float * restrict g,
    __global const float * restrict beta,
    int B
) {
    int h = get_group_id(0);
    if (h >= H_V) return;
    int j = get_local_id(0); // 0..127

    int kh = h / 2;
    __global float * S_h = state + (size_t)h * (S_V * S_V);

    // Staging buffers in SLM (C = 16)
    __local float s_q[16][128];
    __local float s_k[16][128];
    __local float s_KKT[16][16];
    __local float s_QKT[16][16];
    __local float s_gamma_mat[16][16];
    __local float s_gamma_from_0[16];
    __local float s_beta[16];

    // Load initial state column j into private registers
    float s_col[128];
    #pragma unroll 4
    for (int i = 0; i < 128; ++i) {
        s_col[i] = S_h[i * 128 + j];
    }

    for (int t0 = 0; t0 < B; t0 += 16) {
        int c_len = min(16, B - t0);

        // 1. Cooperative SLM staging
        for (int t = 0; t < c_len; ++t) {
            s_k[t][j] = k[(size_t)(t0 + t) * (H_K * S_V) + kh * S_V + j];
            s_q[t][j] = q[(size_t)(t0 + t) * (H_K * S_V) + kh * S_V + j];
        }

        if (j < c_len) {
            s_beta[j] = beta[(size_t)(t0 + j) * H_V + h];
        }

        if (j == 0) {
            float prod = 1.0f;
            for (int t = 0; t < c_len; ++t) {
                prod *= g[(size_t)(t0 + t) * H_V + h];
                s_gamma_from_0[t] = prod;
            }
            for (int t = 0; t < c_len; ++t) {
                for (int s = 0; s <= t; ++s) {
                    s_gamma_mat[t][s] = (s == t) ? 1.0f : (s_gamma_from_0[t] / s_gamma_from_0[s]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2. Parallel KKT and QKT computation
        // 128 threads compute 256 matrix cells
        for (int p_idx = 0; p_idx < 2; ++p_idx) {
            int p = j * 2 + p_idx;
            int t = p >> 4;
            int s = p & 15;
            if (t < c_len && s < c_len) {
                if (s < t) {
                    float k_acc = 0.0f;
                    #pragma unroll 4
                    for (int i = 0; i < 128; ++i) {
                        k_acc += s_k[t][i] * s_k[s][i];
                    }
                    s_KKT[t][s] = k_acc;
                }
                if (s <= t) {
                    float q_acc = 0.0f;
                    #pragma unroll 4
                    for (int i = 0; i < 128; ++i) {
                        q_acc += s_q[t][i] * s_k[s][i];
                    }
                    s_QKT[t][s] = q_acc;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. Compute V_init and O_state from s_col
        float V_init[16];
        float O_state[16];
        float v_local[16];
        for (int t = 0; t < c_len; ++t) {
            v_local[t] = v[(size_t)(t0 + t) * C_QKV + 4096 + h * S_V + j];

            float v_acc0 = 0.0f, v_acc1 = 0.0f, v_acc2 = 0.0f, v_acc3 = 0.0f;
            float o_acc0 = 0.0f, o_acc1 = 0.0f, o_acc2 = 0.0f, o_acc3 = 0.0f;
            #pragma unroll 2
            for (int i = 0; i < 128; i += 4) {
                v_acc0 += s_col[i]     * s_k[t][i];
                v_acc1 += s_col[i + 1] * s_k[t][i + 1];
                v_acc2 += s_col[i + 2] * s_k[t][i + 2];
                v_acc3 += s_col[i + 3] * s_k[t][i + 3];

                o_acc0 += s_col[i]     * s_q[t][i];
                o_acc1 += s_col[i + 1] * s_q[t][i + 1];
                o_acc2 += s_col[i + 2] * s_q[t][i + 2];
                o_acc3 += s_col[i + 3] * s_q[t][i + 3];
            }
            float v_acc = (v_acc0 + v_acc1) + (v_acc2 + v_acc3);
            float o_acc = (o_acc0 + o_acc1) + (o_acc2 + o_acc3);

            V_init[t] = s_gamma_from_0[t] * v_acc;
            O_state[t] = s_gamma_from_0[t] * o_acc;
        }

        // 4. Forward substitution to solve M D = V - V_init
        float D[16];
        for (int t = 0; t < c_len; ++t) {
            float rhs = v_local[t] - V_init[t];
            float sum = 0.0f;
            for (int s = 0; s < t; ++s) {
                sum += s_gamma_mat[t][s] * s_KKT[t][s] * D[s];
            }
            D[t] = (rhs - sum) * s_beta[t];
        }

        // 5. Compute output for tokens in this chunk
        for (int t = 0; t < c_len; ++t) {
            float o_intra = 0.0f;
            for (int s = 0; s <= t; ++s) {
                o_intra += s_gamma_mat[t][s] * s_QKT[t][s] * D[s];
            }
            float out_val = (O_state[t] + o_intra) * SCALE_128;
            out[(size_t)(t0 + t) * (H_V * S_V) + h * S_V + j] = out_val;
        }

        // 6. Update state for next chunk
        float chunk_decay = s_gamma_from_0[c_len - 1];
        float w[16];
        #pragma unroll
        for (int s = 0; s < c_len; ++s) {
            w[s] = s_gamma_mat[c_len - 1][s] * D[s];
        }
        #pragma unroll 2
        for (int i = 0; i < 128; ++i) {
            float d_acc = 0.0f;
            #pragma unroll
            for (int s = 0; s < c_len; ++s) {
                d_acc += w[s] * s_k[s][i];
            }
            s_col[i] = chunk_decay * s_col[i] + d_acc;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 7. Write final state
    #pragma unroll 4
    for (int i = 0; i < 128; ++i) {
        S_h[i * 128 + j] = s_col[i];
    }
}
