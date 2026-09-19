// AInfer Unified SPIR-V Kernel Suite for Qwen3.5-MoE on Intel Arc 140V (Xe2)
// Contains all kernels for both DeltaNet (30 layers) and Full-Attention (10 layers) blocks.

#define GROUP_SIZE 128
#define HIDDEN_DIM 2048
#define NUM_EXPERTS 256
#define TOP_K 8
#define S_V 128
#define H_V 32
#define H_K 16
#define C_QKV 8192
#define HEAD_DIM 256
#define ROTARY_DIM 64
#define ROTARY_HALF 32
#define NUM_Q_HEADS 16
#define NUM_KV_HEADS 2
#define GQA_GROUP_SIZE 8
#define ROPE_THETA 10000000.0f
#define ATTN_SCALE 0.0625f
#define SCALE_128 0.08838834764831845f
#define EPS 1e-6f

// Helper conversions
static inline float bf16_to_fp32(ushort b) {
    uint u = ((uint)b) << 16;
    return as_float(u);
}

static inline float bf16_to_float(ushort b) {
    return bf16_to_fp32(b);
}

static inline ushort float_to_bf16(float x) {
    uint u = as_uint(x);
    return (ushort)((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

// =========================================================================
// 1. INT4 Symmetric Group-128 GEMV
// =========================================================================
__kernel void int4_gemv_m1(
    __global float * restrict y,              // [M]
    __global const uchar * restrict w_packed, // [M, K / 2]
    __global const ushort * restrict w_scale, // [M, K / 128]
    __global const float * restrict x,        // [K]
    int M,
    int K
) {
    int m = get_global_id(0);
    if (m >= M) return;

    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w = w_packed + (size_t)m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)m * num_groups;

    float total_sum = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        float scale = bf16_to_fp32(row_s[g]);
        __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);
        __global const float *grp_x = x + g * GROUP_SIZE;

        // GROUP_SIZE = 128 elements = 64 bytes of packed weights
        // Read as 4 x uchar16 (16 bytes = 32 weights per vector load)
        __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;
        __global const float8 *x_vec8 = (__global const float8 *)grp_x;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uchar16 wb = w_vec16[v];

            // 32 weights = 4 float8 chunks
            float8 x0 = x_vec8[v * 4 + 0];
            float8 x1 = x_vec8[v * 4 + 1];
            float8 x2 = x_vec8[v * 4 + 2];
            float8 x3 = x_vec8[v * 4 + 3];

            // Byte 0..3 -> 8 weights for x0
            int n0  = (int)((char)(wb.s0 << 4)) >> 4;
            int n1  = (int)((char)wb.s0) >> 4;
            int n2  = (int)((char)(wb.s1 << 4)) >> 4;
            int n3  = (int)((char)wb.s1) >> 4;
            int n4  = (int)((char)(wb.s2 << 4)) >> 4;
            int n5  = (int)((char)wb.s2) >> 4;
            int n6  = (int)((char)(wb.s3 << 4)) >> 4;
            int n7  = (int)((char)wb.s3) >> 4;

            acc0 += (float)n0 * x0.s0 + (float)n1 * x0.s1
                  + (float)n2 * x0.s2 + (float)n3 * x0.s3
                  + (float)n4 * x0.s4 + (float)n5 * x0.s5
                  + (float)n6 * x0.s6 + (float)n7 * x0.s7;

            // Byte 4..7 -> 8 weights for x1
            int n8  = (int)((char)(wb.s4 << 4)) >> 4;
            int n9  = (int)((char)wb.s4) >> 4;
            int n10 = (int)((char)(wb.s5 << 4)) >> 4;
            int n11 = (int)((char)wb.s5) >> 4;
            int n12 = (int)((char)(wb.s6 << 4)) >> 4;
            int n13 = (int)((char)wb.s6) >> 4;
            int n14 = (int)((char)(wb.s7 << 4)) >> 4;
            int n15 = (int)((char)wb.s7) >> 4;

            acc1 += (float)n8 * x1.s0 + (float)n9 * x1.s1
                  + (float)n10 * x1.s2 + (float)n11 * x1.s3
                  + (float)n12 * x1.s4 + (float)n13 * x1.s5
                  + (float)n14 * x1.s6 + (float)n15 * x1.s7;

            // Byte 8..11 -> 8 weights for x2
            int n16 = (int)((char)(wb.s8 << 4)) >> 4;
            int n17 = (int)((char)wb.s8) >> 4;
            int n18 = (int)((char)(wb.s9 << 4)) >> 4;
            int n19 = (int)((char)wb.s9) >> 4;
            int n20 = (int)((char)(wb.sa << 4)) >> 4;
            int n21 = (int)((char)wb.sa) >> 4;
            int n22 = (int)((char)(wb.sb << 4)) >> 4;
            int n23 = (int)((char)wb.sb) >> 4;

            acc0 += (float)n16 * x2.s0 + (float)n17 * x2.s1
                  + (float)n18 * x2.s2 + (float)n19 * x2.s3
                  + (float)n20 * x2.s4 + (float)n21 * x2.s5
                  + (float)n22 * x2.s6 + (float)n23 * x2.s7;

            // Byte 12..15 -> 8 weights for x3
            int n24 = (int)((char)(wb.sc << 4)) >> 4;
            int n25 = (int)((char)wb.sc) >> 4;
            int n26 = (int)((char)(wb.sd << 4)) >> 4;
            int n27 = (int)((char)wb.sd) >> 4;
            int n28 = (int)((char)(wb.se << 4)) >> 4;
            int n29 = (int)((char)wb.se) >> 4;
            int n30 = (int)((char)(wb.sf << 4)) >> 4;
            int n31 = (int)((char)wb.sf) >> 4;

            acc1 += (float)n24 * x3.s0 + (float)n25 * x3.s1
                  + (float)n26 * x3.s2 + (float)n27 * x3.s3
                  + (float)n28 * x3.s4 + (float)n29 * x3.s5
                  + (float)n30 * x3.s6 + (float)n31 * x3.s7;
        }

        total_sum += (acc0 + acc1) * scale;
    }

    y[m] = total_sum;
}

// =========================================================================
// 1b. INT4 Symmetric Group-128 Batched Prefill GEMM (Batch B <= 32)
// =========================================================================
__kernel void int4_gemm_prefill(
    __global float * restrict Y,              // [B, M] row-major: Y[b * M + m]
    __global const uchar * restrict w_packed, // [M, K / 2]
    __global const ushort * restrict w_scale, // [M, K / 128]
    __global const float * restrict X,        // [B, K] row-major: X[b * K + k]
    int M,
    int K,
    int B
) {
    int m = get_global_id(0);
    if (m >= M) return;

    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w = w_packed + (size_t)m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)m * num_groups;

    float total_sums[32];
    #pragma unroll
    for (int b = 0; b < 32; ++b) {
        total_sums[b] = 0.0f;
    }

    for (int g = 0; g < num_groups; ++g) {
        float scale = bf16_to_fp32(row_s[g]);
        __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);

        float grp_acc[32];
        #pragma unroll
        for (int b = 0; b < 32; ++b) {
            grp_acc[b] = 0.0f;
        }

        __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uchar16 wb = w_vec16[v];

            float4 w0, w1, w2, w3, w4, w5, w6, w7;

            int n0 = (int)((char)(wb.s0 << 4)) >> 4;
            int n1 = (int)((char)wb.s0) >> 4;
            int n2 = (int)((char)(wb.s1 << 4)) >> 4;
            int n3 = (int)((char)wb.s1) >> 4;
            int n4 = (int)((char)(wb.s2 << 4)) >> 4;
            int n5 = (int)((char)wb.s2) >> 4;
            int n6 = (int)((char)(wb.s3 << 4)) >> 4;
            int n7 = (int)((char)wb.s3) >> 4;
            w0 = (float4)((float)n0, (float)n1, (float)n2, (float)n3);
            w1 = (float4)((float)n4, (float)n5, (float)n6, (float)n7);

            int n8  = (int)((char)(wb.s4 << 4)) >> 4;
            int n9  = (int)((char)wb.s4) >> 4;
            int n10 = (int)((char)(wb.s5 << 4)) >> 4;
            int n11 = (int)((char)wb.s5) >> 4;
            int n12 = (int)((char)(wb.s6 << 4)) >> 4;
            int n13 = (int)((char)wb.s6) >> 4;
            int n14 = (int)((char)(wb.s7 << 4)) >> 4;
            int n15 = (int)((char)wb.s7) >> 4;
            w2 = (float4)((float)n8,  (float)n9,  (float)n10, (float)n11);
            w3 = (float4)((float)n12, (float)n13, (float)n14, (float)n15);

            int n16 = (int)((char)(wb.s8 << 4)) >> 4;
            int n17 = (int)((char)wb.s8) >> 4;
            int n18 = (int)((char)(wb.s9 << 4)) >> 4;
            int n19 = (int)((char)wb.s9) >> 4;
            int n20 = (int)((char)(wb.sa << 4)) >> 4;
            int n21 = (int)((char)wb.sa) >> 4;
            int n22 = (int)((char)(wb.sb << 4)) >> 4;
            int n23 = (int)((char)wb.sb) >> 4;
            w4 = (float4)((float)n16, (float)n17, (float)n18, (float)n19);
            w5 = (float4)((float)n20, (float)n21, (float)n22, (float)n23);

            int n24 = (int)((char)(wb.sc << 4)) >> 4;
            int n25 = (int)((char)wb.sc) >> 4;
            int n26 = (int)((char)(wb.sd << 4)) >> 4;
            int n27 = (int)((char)wb.sd) >> 4;
            int n28 = (int)((char)(wb.se << 4)) >> 4;
            int n29 = (int)((char)wb.se) >> 4;
            int n30 = (int)((char)(wb.sf << 4)) >> 4;
            int n31 = (int)((char)wb.sf) >> 4;
            w6 = (float4)((float)n24, (float)n25, (float)n26, (float)n27);
            w7 = (float4)((float)n28, (float)n29, (float)n30, (float)n31);

            int k_sub = g * GROUP_SIZE + v * 32;

            for (int b = 0; b < B; ++b) {
                __global const float4 *x_vec4 = (__global const float4 *)(X + (size_t)b * K + k_sub);
                float4 x0 = x_vec4[0];
                float4 x1 = x_vec4[1];
                float4 x2 = x_vec4[2];
                float4 x3 = x_vec4[3];
                float4 x4 = x_vec4[4];
                float4 x5 = x_vec4[5];
                float4 x6 = x_vec4[6];
                float4 x7 = x_vec4[7];

                grp_acc[b] += dot(w0, x0) + dot(w1, x1) + dot(w2, x2) + dot(w3, x3)
                            + dot(w4, x4) + dot(w5, x5) + dot(w6, x6) + dot(w7, x7);
            }
        }

        for (int b = 0; b < B; ++b) {
            total_sums[b] += grp_acc[b] * scale;
        }
    }

    for (int b = 0; b < B; ++b) {
        Y[(size_t)b * M + m] = total_sums[b];
    }
}

// =========================================================================
// 2. Deterministic MoE Top-8 Router
// =========================================================================
__kernel void moe_topk_router(
    __global const float * restrict x,           // [HIDDEN_DIM]
    __global const float * restrict w_gate,      // [NUM_EXPERTS, HIDDEN_DIM]
    __global const float * restrict w_shared,    // [HIDDEN_DIM] (optional, can be NULL)
    __global uint * restrict out_topk_indices,   // [TOP_K]
    __global float * restrict out_topk_weights,  // [TOP_K]
    __global float * restrict out_shared_gate    // [1]
) {
    int lid = get_local_id(0); // 0 .. 255
    __local float local_logits[NUM_EXPERTS];
    __local float local_probs[NUM_EXPERTS];
    __local float local_max;
    __local float local_sum;

    __global const float4 *x_vec = (__global const float4 *)x;
    __global const float4 *w_vec = (__global const float4 *)(w_gate + lid * HIDDEN_DIM);

    float sum = 0.0f;
    for (int j = 0; j < HIDDEN_DIM / 4; ++j) {
        float4 xv = x_vec[j];
        float4 wv = w_vec[j];
        sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }
    local_logits[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float m = local_logits[0];
        for (int i = 1; i < NUM_EXPERTS; ++i) {
            if (local_logits[i] > m) {
                m = local_logits[i];
            }
        }
        local_max = m;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float exp_val = exp(local_logits[lid] - local_max);
    local_probs[lid] = exp_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float s = 0.0f;
        for (int i = 0; i < NUM_EXPERTS; ++i) {
            s += local_probs[i];
        }
        local_sum = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    local_probs[lid] = local_probs[lid] / local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float best_prob[TOP_K];
        uint best_idx[TOP_K];

        for (int k = 0; k < TOP_K; ++k) {
            best_prob[k] = -1.0f;
            best_idx[k] = 0xFFFFFFFF;
        }

        for (uint i = 0; i < NUM_EXPERTS; ++i) {
            float p = local_probs[i];
            if (p > best_prob[TOP_K - 1]) {
                int pos = TOP_K - 1;
                while (pos > 0 && (p > best_prob[pos - 1] || (p == best_prob[pos - 1] && i < best_idx[pos - 1]))) {
                    pos--;
                }
                for (int j = TOP_K - 1; j > pos; --j) {
                    best_prob[j] = best_prob[j - 1];
                    best_idx[j] = best_idx[j - 1];
                }
                best_prob[pos] = p;
                best_idx[pos] = i;
            }
        }

        float top_sum = 0.0f;
        for (int k = 0; k < TOP_K; ++k) {
            top_sum += best_prob[k];
        }

        for (int k = 0; k < TOP_K; ++k) {
            out_topk_indices[k] = best_idx[k];
            out_topk_weights[k] = best_prob[k] / top_sum;
        }
    }

    if (w_shared != 0 && lid == 0) {
        __global const float4 *ws_vec = (__global const float4 *)w_shared;
        float sh_sum = 0.0f;
        for (int j = 0; j < HIDDEN_DIM / 4; ++j) {
            float4 xv = x_vec[j];
            float4 wv = ws_vec[j];
            sh_sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
        out_shared_gate[0] = 1.0f / (1.0f + exp(-sh_sum));
    }
}

// =========================================================================
// 3. Elementwise Kernels
// =========================================================================
__kernel void rmsnorm_2048(
    __global float * restrict y,
    __global const float * restrict x,
    __global const float * restrict w
) {
    int lid = get_local_id(0);
    __local float local_ss[256];

    float sum_sq = 0.0f;
    for (int j = lid; j < HIDDEN_DIM; j += 256) {
        float val = x[j];
        sum_sq += val * val;
    }
    local_ss[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            local_ss[lid] += local_ss[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_rms = rsqrt(local_ss[0] / (float)HIDDEN_DIM + EPS);

    for (int j = lid; j < HIDDEN_DIM; j += 256) {
        y[j] = x[j] * inv_rms * (1.0f + w[j]);
    }
}

__kernel void rmsnorm_head_256(
    __global float * restrict y,
    __global const float * restrict x,
    __global const float * restrict w,
    int num_heads
) {
    int h = get_group_id(0);
    if (h >= num_heads) return;
    int lid = get_local_id(0);

    __local float local_ss[64];
    __global const float * x_head = x + h * 256;
    __global float * y_head = y + h * 256;

    float sum_sq = 0.0f;
    for (int j = lid; j < 256; j += 64) {
        float val = x_head[j];
        sum_sq += val * val;
    }
    local_ss[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 32; s > 0; s >>= 1) {
        if (lid < s) {
            local_ss[lid] += local_ss[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_rms = rsqrt(local_ss[0] / 256.0f + EPS);

    for (int j = lid; j < 256; j += 64) {
        y_head[j] = x_head[j] * inv_rms * (1.0f + w[j]);
    }
}

__kernel void silu_mul_512(
    __global float * restrict z,
    __global const float * restrict g,
    __global const float * restrict u
) {
    int idx = get_global_id(0);
    if (idx >= 512) return;
    float g_val = g[idx];
    float u_val = u[idx];
    float silu_g = g_val / (1.0f + exp(-g_val));
    z[idx] = silu_g * u_val;
}

__kernel void residual_add_2048(
    __global float * restrict y,
    __global const float * restrict x1,
    __global const float * restrict x2
) {
    int idx = get_global_id(0);
    if (idx >= HIDDEN_DIM) return;
    y[idx] = x1[idx] + x2[idx];
}

__kernel void moe_accumulate_expert(
    __global float * restrict out,
    __global const float * restrict expert_out,
    float weight,
    int is_first
) {
    int idx = get_global_id(0);
    if (idx >= HIDDEN_DIM) return;
    if (is_first) {
        out[idx] = expert_out[idx] * weight;
    } else {
        out[idx] += expert_out[idx] * weight;
    }
}

__kernel void moe_add_shared_expert(
    __global float * restrict out,
    __global const float * restrict shared_out,
    float shared_gate_val
) {
    int idx = get_global_id(0);
    if (idx >= HIDDEN_DIM) return;
    out[idx] += shared_out[idx] * shared_gate_val;
}

__kernel void argmax_stage1(
    __global const float * restrict logits,
    __global float * restrict group_max_val,
    __global uint * restrict group_max_idx,
    uint N
) {
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int idx = gid * 256 + lid;

    __local float s_val[256];
    __local uint s_idx[256];

    if (idx < N) {
        s_val[lid] = logits[idx];
        s_idx[lid] = idx;
    } else {
        s_val[lid] = -1e30f;
        s_idx[lid] = 0xFFFFFFFF;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            if (s_val[lid + s] > s_val[lid] || (s_val[lid + s] == s_val[lid] && s_idx[lid + s] < s_idx[lid])) {
                s_val[lid] = s_val[lid + s];
                s_idx[lid] = s_idx[lid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        group_max_val[gid] = s_val[0];
        group_max_idx[gid] = s_idx[0];
    }
}

__kernel void argmax_stage2(
    __global const float * restrict group_max_val,
    __global const uint * restrict group_max_idx,
    __global uint * restrict out_token_id,
    uint num_groups
) {
    int lid = get_local_id(0);
    if (lid == 0) {
        float best_val = group_max_val[0];
        uint best_id = group_max_idx[0];
        for (uint i = 1; i < num_groups; ++i) {
            float v = group_max_val[i];
            uint id = group_max_idx[i];
            if (v > best_val || (v == best_val && id < best_id)) {
                best_val = v;
                best_id = id;
            }
        }
        out_token_id[0] = best_id;
    }
}

// =========================================================================
// 4. DeltaNet Linear Attention Kernels
// =========================================================================
__kernel void conv1d_update_silu(
    __global float * restrict qkv_out,
    __global const float * restrict qkv_in,
    __global float * restrict conv_state,
    __global const float * restrict conv_weight
) {
    int c = get_global_id(0);
    if (c >= C_QKV) return;

    int s_base = c * 3;
    float s0 = conv_state[s_base + 0];
    float s1 = conv_state[s_base + 1];
    float s2 = conv_state[s_base + 2];
    float x  = qkv_in[c];

    int w_base = c * 4;
    float w0 = conv_weight[w_base + 0];
    float w1 = conv_weight[w_base + 1];
    float w2 = conv_weight[w_base + 2];
    float w3 = conv_weight[w_base + 3];

    float sum = s0 * w0 + s1 * w1 + s2 * w2 + x * w3;
    float silu_val = sum / (1.0f + exp(-sum));
    qkv_out[c] = silu_val;

    conv_state[s_base + 0] = s1;
    conv_state[s_base + 1] = s2;
    conv_state[s_base + 2] = x;
}

__kernel void head_l2_norm_128(
    __global float * restrict y,
    __global const float * restrict x,
    int num_heads
) {
    int h = get_group_id(0);
    if (h >= num_heads) return;
    int lid = get_local_id(0);

    __local float local_ss[128];
    __global const float * x_head = x + h * S_V;
    __global float * y_head = y + h * S_V;

    float val = x_head[lid];
    local_ss[lid] = val * val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 64; s > 0; s >>= 1) {
        if (lid < s) {
            local_ss[lid] += local_ss[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_l2 = 1.0f / fmax(sqrt(local_ss[0]), EPS);
    y_head[lid] = val * inv_l2;
}

__kernel void deltanet_gate_prep(
    __global float * restrict g_out,
    __global float * restrict beta_out,
    __global const float * restrict a_in,
    __global const float * restrict b_in,
    __global const float * restrict dt_bias,
    __global const float * restrict A_log
) {
    int h = get_global_id(0);
    if (h >= H_V) return;

    float a_val = a_in[h] + dt_bias[h];
    float dt = (a_val > 20.0f) ? a_val : log(1.0f + exp(a_val));
    float gate = -exp(A_log[h]) * dt;
    g_out[h] = exp(gate);
    beta_out[h] = 1.0f / (1.0f + exp(-b_in[h]));
}

__kernel void deltanet_recurrent_decode(
    __global float * restrict out,
    __global float * restrict state,
    __global const float * restrict q,
    __global const float * restrict k,
    __global const float * restrict v,
    __global const float * restrict g,
    __global const float * restrict beta
) {
    int h = get_group_id(0);
    if (h >= H_V) return;
    int j = get_local_id(0);

    __local float s_q[S_V];
    __local float s_k[S_V];

    int kh = h / 2;

    s_q[j] = q[kh * S_V + j];
    s_k[j] = k[kh * S_V + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    float v_val = v[h * S_V + j];
    float g_val = g[h];
    float b_val = beta[h];

    __global float * S_h = state + h * (S_V * S_V);

    float kv_acc = 0.0f;
    for (int i = 0; i < S_V; ++i) {
        kv_acc += S_h[i * S_V + j] * s_k[i];
    }
    float kv_j = kv_acc * g_val;
    float delta_j = (v_val - kv_j) * b_val;

    float o_acc = 0.0f;
    for (int i = 0; i < S_V; ++i) {
        float s_old = S_h[i * S_V + j];
        float s_new = g_val * s_old + s_k[i] * delta_j;
        S_h[i * S_V + j] = s_new;
        o_acc += s_new * s_q[i];
    }

    out[h * S_V + j] = o_acc * SCALE_128;
}

__kernel void deltanet_head_norm_silu_z(
    __global float * restrict final_out,
    __global const float * restrict attn_out,
    __global const float * restrict z_gate,
    __global const float * restrict ssm_norm_w
) {
    int h = get_group_id(0);
    if (h >= H_V) return;
    int j = get_local_id(0);

    __local float local_ss[128];
    __global const float * x_head = attn_out + h * S_V;
    __global const float * z_head = z_gate + h * S_V;
    __global float * y_head = final_out + h * S_V;

    float val = x_head[j];
    local_ss[j] = val * val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 64; s > 0; s >>= 1) {
        if (j < s) {
            local_ss[j] += local_ss[j + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_rms = rsqrt(local_ss[0] / 128.0f + EPS);
    float norm_val = val * inv_rms * ssm_norm_w[j];
    float zv = z_head[j];
    float silu_z = zv / (1.0f + exp(-zv));
    y_head[j] = norm_val * silu_z;
}

// =========================================================================
// 5. Full Attention & RoPE Kernels
// =========================================================================
__kernel void rope_and_kv_append_bf16(
    __global float * restrict q,           // [16, 256]
    __global float * restrict k,           // [2, 256]
    __global const float * restrict v,     // [2, 256]
    __global ushort * restrict k_cache,    // [2, max_ctx, 256]
    __global ushort * restrict v_cache,    // [2, max_ctx, 256]
    uint pos,
    uint max_ctx
) {
    int gid = get_global_id(0);
    __local float s_buf[HEAD_DIM];

    float cos_t = 1.0f;
    float sin_t = 0.0f;
    if (gid < ROTARY_HALF) {
        float inv_freq = pow(ROPE_THETA, -((float)gid / (float)ROTARY_HALF));
        float theta = (float)pos * inv_freq;
        cos_t = cos(theta);
        sin_t = sin(theta);
    } else if (gid < ROTARY_DIM) {
        int i = gid - ROTARY_HALF;
        float inv_freq = pow(ROPE_THETA, -((float)i / (float)ROTARY_HALF));
        float theta = (float)pos * inv_freq;
        cos_t = cos(theta);
        sin_t = sin(theta);
    }

    for (int h = 0; h < NUM_Q_HEADS; ++h) {
        __global float * q_head = q + h * HEAD_DIM;
        s_buf[gid] = q_head[gid];
        barrier(CLK_LOCAL_MEM_FENCE);

        float rot_val;
        if (gid < ROTARY_HALF) {
            float x0 = s_buf[gid];
            float x1 = s_buf[gid + ROTARY_HALF];
            rot_val = x0 * cos_t - x1 * sin_t;
        } else if (gid < ROTARY_DIM) {
            float x0 = s_buf[gid - ROTARY_HALF];
            float x1 = s_buf[gid];
            rot_val = x0 * sin_t + x1 * cos_t;
        } else {
            rot_val = s_buf[gid];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        q_head[gid] = rot_val;
    }

    for (int h = 0; h < NUM_KV_HEADS; ++h) {
        __global float * k_head = k + h * HEAD_DIM;
        s_buf[gid] = k_head[gid];
        barrier(CLK_LOCAL_MEM_FENCE);

        float rot_val;
        if (gid < ROTARY_HALF) {
            float x0 = s_buf[gid];
            float x1 = s_buf[gid + ROTARY_HALF];
            rot_val = x0 * cos_t - x1 * sin_t;
        } else if (gid < ROTARY_DIM) {
            float x0 = s_buf[gid - ROTARY_HALF];
            float x1 = s_buf[gid];
            rot_val = x0 * sin_t + x1 * cos_t;
        } else {
            rot_val = s_buf[gid];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        k_head[gid] = rot_val;

        __global ushort * k_slot = k_cache + (h * max_ctx + pos) * HEAD_DIM;
        __global ushort * v_slot = v_cache + (h * max_ctx + pos) * HEAD_DIM;
        k_slot[gid] = float_to_bf16(rot_val);
        v_slot[gid] = float_to_bf16(v[h * HEAD_DIM + gid]);
    }
}

__kernel void gqa_attn_decode_bf16(
    __global float * restrict out,
    __global const float * restrict q,
    __global const float * restrict gate,
    __global const ushort * restrict k_cache,
    __global const ushort * restrict v_cache,
    uint pos,
    uint max_ctx
) {
    int qh = get_group_id(0);
    if (qh >= NUM_Q_HEADS) return;
    int tid = get_local_id(0);

    int kv_h = qh / GQA_GROUP_SIZE;

    __local float s_q[HEAD_DIM];
    __local float s_red[HEAD_DIM];

    s_q[tid] = q[qh * HEAD_DIM + tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float run_acc = 0.0f;

    uint total_tokens = pos + 1;

    for (uint t = 0; t < total_tokens; ++t) {
        __global const ushort * k_slot = k_cache + (kv_h * max_ctx + t) * HEAD_DIM;
        float k_val = bf16_to_float(k_slot[tid]);

        s_red[tid] = s_q[tid] * k_val;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) {
                s_red[tid] += s_red[tid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float score = s_red[0] * ATTN_SCALE;
        barrier(CLK_LOCAL_MEM_FENCE);

        __global const ushort * v_slot = v_cache + (kv_h * max_ctx + t) * HEAD_DIM;
        float v_val = bf16_to_float(v_slot[tid]);

        if (score > run_max) {
            float exp_diff = exp(run_max - score);
            run_max = score;
            run_sum = run_sum * exp_diff + 1.0f;
            run_acc = run_acc * exp_diff + v_val;
        } else {
            float exp_diff = exp(score - run_max);
            run_sum += exp_diff;
            run_acc += exp_diff * v_val;
        }
    }

    float attn_val = run_acc / run_sum;
    float g_val = gate[qh * HEAD_DIM + tid];
    float sig_g = 1.0f / (1.0f + exp(-g_val));

    out[qh * HEAD_DIM + tid] = attn_val * sig_g;
}

// =========================================================================
// 6. Device-Driven Runtime Control Block Kernels (T5.3 / T5.4)
// =========================================================================

__kernel void embed_gather(
    __global float * restrict out,
    __global const ushort * restrict emb_table,
    __global const int * restrict ctrl // ctrl[0] = token_id
) {
    int gid = get_global_id(0);
    if (gid >= HIDDEN_DIM) return;
    int token_id = ctrl[0];
    ushort b = emb_table[(size_t)token_id * HIDDEN_DIM + gid];
    out[gid] = bf16_to_fp32(b);
}

__kernel void rope_and_kv_append_ctrl(
    __global float * restrict q,           // [16, 256]
    __global float * restrict k,           // [2, 256]
    __global const float * restrict v,     // [2, 256]
    __global ushort * restrict k_cache,    // [2, max_ctx, 256]
    __global ushort * restrict v_cache,    // [2, max_ctx, 256]
    __global const int * restrict ctrl,    // ctrl[1] = position
    uint max_ctx
) {
    uint pos = (uint)ctrl[1];
    int gid = get_global_id(0);
    __local float s_buf[HEAD_DIM];

    float cos_t = 1.0f;
    float sin_t = 0.0f;

    if (gid < ROTARY_HALF) {
        float inv_freq = pow(ROPE_THETA, -((float)gid / (float)ROTARY_HALF));
        float theta = (float)pos * inv_freq;
        cos_t = cos(theta);
        sin_t = sin(theta);
    } else if (gid < ROTARY_DIM) {
        int i = gid - ROTARY_HALF;
        float inv_freq = pow(ROPE_THETA, -((float)i / (float)ROTARY_HALF));
        float theta = (float)pos * inv_freq;
        cos_t = cos(theta);
        sin_t = sin(theta);
    }

    for (int h = 0; h < NUM_Q_HEADS; ++h) {
        __global float * q_head = q + h * HEAD_DIM;
        s_buf[gid] = q_head[gid];
        barrier(CLK_LOCAL_MEM_FENCE);

        float rot_val;
        if (gid < ROTARY_HALF) {
            float x0 = s_buf[gid];
            float x1 = s_buf[gid + ROTARY_HALF];
            rot_val = x0 * cos_t - x1 * sin_t;
        } else if (gid < ROTARY_DIM) {
            float x0 = s_buf[gid - ROTARY_HALF];
            float x1 = s_buf[gid];
            rot_val = x0 * sin_t + x1 * cos_t;
        } else {
            rot_val = s_buf[gid];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        q_head[gid] = rot_val;
    }

    for (int h = 0; h < NUM_KV_HEADS; ++h) {
        __global float * k_head = k + h * HEAD_DIM;
        s_buf[gid] = k_head[gid];
        barrier(CLK_LOCAL_MEM_FENCE);

        float rot_val;
        if (gid < ROTARY_HALF) {
            float x0 = s_buf[gid];
            float x1 = s_buf[gid + ROTARY_HALF];
            rot_val = x0 * cos_t - x1 * sin_t;
        } else if (gid < ROTARY_DIM) {
            float x0 = s_buf[gid - ROTARY_HALF];
            float x1 = s_buf[gid];
            rot_val = x0 * sin_t + x1 * cos_t;
        } else {
            rot_val = s_buf[gid];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        k_head[gid] = rot_val;

        __global ushort * k_slot = k_cache + (h * max_ctx + pos) * HEAD_DIM;
        __global ushort * v_slot = v_cache + (h * max_ctx + pos) * HEAD_DIM;
        k_slot[gid] = float_to_bf16(rot_val);
        v_slot[gid] = float_to_bf16(v[h * HEAD_DIM + gid]);
    }
}

__kernel void deinterleave_q_gate(
    __global float * restrict q_out,           // [16, 256] = 4096
    __global float * restrict gate_out,        // [16, 256] = 4096
    __global const float * restrict q_proj_in  // [16, 512] = 8192
) {
    int gid = get_global_id(0);
    if (gid >= NUM_Q_HEADS * HEAD_DIM) return;
    int h = gid / HEAD_DIM;
    int d = gid % HEAD_DIM;

    int in_base = h * (2 * HEAD_DIM);
    q_out[gid] = q_proj_in[in_base + d];
    gate_out[gid] = q_proj_in[in_base + HEAD_DIM + d];
}

__kernel void gqa_attn_decode_ctrl(
    __global float * restrict out,
    __global const float * restrict q,
    __global const float * restrict gate,
    __global const ushort * restrict k_cache,
    __global const ushort * restrict v_cache,
    __global const int * restrict ctrl,    // ctrl[1] = position
    uint max_ctx
) {
    uint pos = (uint)ctrl[1];
    int qh = get_group_id(0);
    if (qh >= NUM_Q_HEADS) return;
    int tid = get_local_id(0);

    int kv_h = qh / GQA_GROUP_SIZE;

    __local float s_q[HEAD_DIM];
    __local float s_red[HEAD_DIM];

    s_q[tid] = q[qh * HEAD_DIM + tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float run_acc = 0.0f;

    uint total_tokens = pos + 1;

    for (uint t = 0; t < total_tokens; ++t) {
        __global const ushort * k_slot = k_cache + (kv_h * max_ctx + t) * HEAD_DIM;
        float k_val = bf16_to_float(k_slot[tid]);

        s_red[tid] = s_q[tid] * k_val;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) {
                s_red[tid] += s_red[tid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float score = s_red[0] * ATTN_SCALE;
        barrier(CLK_LOCAL_MEM_FENCE);

        __global const ushort * v_slot = v_cache + (kv_h * max_ctx + t) * HEAD_DIM;
        float v_val = bf16_to_float(v_slot[tid]);

        if (score > run_max) {
            float exp_diff = exp(run_max - score);
            run_max = score;
            run_sum = run_sum * exp_diff + 1.0f;
            run_acc = run_acc * exp_diff + v_val;
        } else {
            float exp_diff = exp(score - run_max);
            run_sum += exp_diff;
            run_acc += exp_diff * v_val;
        }
    }

    float attn_val = run_acc / run_sum;
    float g_val = gate[qh * HEAD_DIM + tid];
    float sig_g = 1.0f / (1.0f + exp(-g_val));

    out[qh * HEAD_DIM + tid] = attn_val * sig_g;
}

__kernel void moe_expert_gemv_ctrl(
    __global float * restrict y,               // [M]
    __global const uchar * restrict w_bank,    // [256, M, K/2]
    __global const ushort * restrict s_bank,   // [256, M, K/128]
    __global const float * restrict x,         // [K]
    __global const uint * restrict top_idx,    // [8]
    int k_slot,                                // 0..7
    int M,
    int K
) {
    int m = get_global_id(0);
    if (m >= M) return;

    uint eid = top_idx[k_slot];
    size_t row_idx = (size_t)eid * M + m;
    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w = w_bank + row_idx * (K / 2);
    __global const ushort *row_s = s_bank + row_idx * num_groups;

    float total_sum = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        float scale = bf16_to_fp32(row_s[g]);
        __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);
        __global const float *grp_x = x + g * GROUP_SIZE;

        // GROUP_SIZE = 128 elements = 64 bytes of packed weights
        // Read as 4 x uchar16 (16 bytes = 32 weights per vector load)
        __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;
        __global const float8 *x_vec8 = (__global const float8 *)grp_x;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uchar16 wb = w_vec16[v];

            // 32 weights = 4 float8 chunks
            float8 x0 = x_vec8[v * 4 + 0];
            float8 x1 = x_vec8[v * 4 + 1];
            float8 x2 = x_vec8[v * 4 + 2];
            float8 x3 = x_vec8[v * 4 + 3];

            // Byte 0..3 -> 8 weights for x0
            int n0  = (int)((char)(wb.s0 << 4)) >> 4;
            int n1  = (int)((char)wb.s0) >> 4;
            int n2  = (int)((char)(wb.s1 << 4)) >> 4;
            int n3  = (int)((char)wb.s1) >> 4;
            int n4  = (int)((char)(wb.s2 << 4)) >> 4;
            int n5  = (int)((char)wb.s2) >> 4;
            int n6  = (int)((char)(wb.s3 << 4)) >> 4;
            int n7  = (int)((char)wb.s3) >> 4;

            acc0 += (float)n0 * x0.s0 + (float)n1 * x0.s1
                  + (float)n2 * x0.s2 + (float)n3 * x0.s3
                  + (float)n4 * x0.s4 + (float)n5 * x0.s5
                  + (float)n6 * x0.s6 + (float)n7 * x0.s7;

            // Byte 4..7 -> 8 weights for x1
            int n8  = (int)((char)(wb.s4 << 4)) >> 4;
            int n9  = (int)((char)wb.s4) >> 4;
            int n10 = (int)((char)(wb.s5 << 4)) >> 4;
            int n11 = (int)((char)wb.s5) >> 4;
            int n12 = (int)((char)(wb.s6 << 4)) >> 4;
            int n13 = (int)((char)wb.s6) >> 4;
            int n14 = (int)((char)(wb.s7 << 4)) >> 4;
            int n15 = (int)((char)wb.s7) >> 4;

            acc1 += (float)n8 * x1.s0 + (float)n9 * x1.s1
                  + (float)n10 * x1.s2 + (float)n11 * x1.s3
                  + (float)n12 * x1.s4 + (float)n13 * x1.s5
                  + (float)n14 * x1.s6 + (float)n15 * x1.s7;

            // Byte 8..11 -> 8 weights for x2
            int n16 = (int)((char)(wb.s8 << 4)) >> 4;
            int n17 = (int)((char)wb.s8) >> 4;
            int n18 = (int)((char)(wb.s9 << 4)) >> 4;
            int n19 = (int)((char)wb.s9) >> 4;
            int n20 = (int)((char)(wb.sa << 4)) >> 4;
            int n21 = (int)((char)wb.sa) >> 4;
            int n22 = (int)((char)(wb.sb << 4)) >> 4;
            int n23 = (int)((char)wb.sb) >> 4;

            acc0 += (float)n16 * x2.s0 + (float)n17 * x2.s1
                  + (float)n18 * x2.s2 + (float)n19 * x2.s3
                  + (float)n20 * x2.s4 + (float)n21 * x2.s5
                  + (float)n22 * x2.s6 + (float)n23 * x2.s7;

            // Byte 12..15 -> 8 weights for x3
            int n24 = (int)((char)(wb.sc << 4)) >> 4;
            int n25 = (int)((char)wb.sc) >> 4;
            int n26 = (int)((char)(wb.sd << 4)) >> 4;
            int n27 = (int)((char)wb.sd) >> 4;
            int n28 = (int)((char)(wb.se << 4)) >> 4;
            int n29 = (int)((char)wb.se) >> 4;
            int n30 = (int)((char)(wb.sf << 4)) >> 4;
            int n31 = (int)((char)wb.sf) >> 4;

            acc1 += (float)n24 * x3.s0 + (float)n25 * x3.s1
                  + (float)n26 * x3.s2 + (float)n27 * x3.s3
                  + (float)n28 * x3.s4 + (float)n29 * x3.s5
                  + (float)n30 * x3.s6 + (float)n31 * x3.s7;
        }

        total_sum += (acc0 + acc1) * scale;
    }

    y[m] = total_sum;
}

__kernel void moe_accumulate_expert_ctrl(
    __global float * restrict acc,             // [2048]
    __global const float * restrict expert_out,// [2048]
    __global const float * restrict top_wt,    // [8]
    int k_slot                                 // 0..7
) {
    int gid = get_global_id(0);
    if (gid >= HIDDEN_DIM) return;
    float wt = top_wt[k_slot];
    if (k_slot == 0) {
        acc[gid] = expert_out[gid] * wt;
    } else {
        acc[gid] += expert_out[gid] * wt;
    }
}

__kernel void moe_add_shared_expert_ctrl(
    __global float * restrict acc,             // [2048]
    __global const float * restrict shared_out,// [2048]
    __global const float * restrict sh_gate_val_ptr
) {
    int gid = get_global_id(0);
    if (gid >= HIDDEN_DIM) return;
    float sig = *sh_gate_val_ptr;
    acc[gid] += shared_out[gid] * sig;
}

__kernel void moe_expert_down_accum_ctrl(
    __global float * restrict acc,             // [HIDDEN_DIM = 2048]
    __global const uchar * restrict w_bank,    // [256, M, K/2]
    __global const ushort * restrict s_bank,   // [256, M, K/128]
    __global const float * restrict x,         // [K = 512]
    __global const uint * restrict top_idx,    // [8]
    __global const float * restrict top_wt,    // [8]
    int k_slot,                                // 0..7
    int M,                                     // 2048
    int K                                      // 512
) {
    int m = get_global_id(0);
    if (m >= M) return;

    uint eid = top_idx[k_slot];
    size_t row_idx = (size_t)eid * M + m;
    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w = w_bank + row_idx * (K / 2);
    __global const ushort *row_s = s_bank + row_idx * num_groups;

    float total_sum = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        float scale = bf16_to_fp32(row_s[g]);
        __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);
        __global const float *grp_x = x + g * GROUP_SIZE;

        __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;
        __global const float8 *x_vec8 = (__global const float8 *)grp_x;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uchar16 wb = w_vec16[v];

            float8 x0 = x_vec8[v * 4 + 0];
            float8 x1 = x_vec8[v * 4 + 1];
            float8 x2 = x_vec8[v * 4 + 2];
            float8 x3 = x_vec8[v * 4 + 3];

            int n0  = (int)((char)(wb.s0 << 4)) >> 4;
            int n1  = (int)((char)wb.s0) >> 4;
            int n2  = (int)((char)(wb.s1 << 4)) >> 4;
            int n3  = (int)((char)wb.s1) >> 4;
            int n4  = (int)((char)(wb.s2 << 4)) >> 4;
            int n5  = (int)((char)wb.s2) >> 4;
            int n6  = (int)((char)(wb.s3 << 4)) >> 4;
            int n7  = (int)((char)wb.s3) >> 4;

            acc0 += (float)n0 * x0.s0 + (float)n1 * x0.s1
                  + (float)n2 * x0.s2 + (float)n3 * x0.s3
                  + (float)n4 * x0.s4 + (float)n5 * x0.s5
                  + (float)n6 * x0.s6 + (float)n7 * x0.s7;

            int n8  = (int)((char)(wb.s4 << 4)) >> 4;
            int n9  = (int)((char)wb.s4) >> 4;
            int n10 = (int)((char)(wb.s5 << 4)) >> 4;
            int n11 = (int)((char)wb.s5) >> 4;
            int n12 = (int)((char)(wb.s6 << 4)) >> 4;
            int n13 = (int)((char)wb.s6) >> 4;
            int n14 = (int)((char)(wb.s7 << 4)) >> 4;
            int n15 = (int)((char)wb.s7) >> 4;

            acc1 += (float)n8 * x1.s0 + (float)n9 * x1.s1
                  + (float)n10 * x1.s2 + (float)n11 * x1.s3
                  + (float)n12 * x1.s4 + (float)n13 * x1.s5
                  + (float)n14 * x1.s6 + (float)n15 * x1.s7;

            int n16 = (int)((char)(wb.s8 << 4)) >> 4;
            int n17 = (int)((char)wb.s8) >> 4;
            int n18 = (int)((char)(wb.s9 << 4)) >> 4;
            int n19 = (int)((char)wb.s9) >> 4;
            int n20 = (int)((char)(wb.sa << 4)) >> 4;
            int n21 = (int)((char)wb.sa) >> 4;
            int n22 = (int)((char)(wb.sb << 4)) >> 4;
            int n23 = (int)((char)wb.sb) >> 4;

            acc0 += (float)n16 * x2.s0 + (float)n17 * x2.s1
                  + (float)n18 * x2.s2 + (float)n19 * x2.s3
                  + (float)n20 * x2.s4 + (float)n21 * x2.s5
                  + (float)n22 * x2.s6 + (float)n23 * x2.s7;

            int n24 = (int)((char)(wb.sc << 4)) >> 4;
            int n25 = (int)((char)wb.sc) >> 4;
            int n26 = (int)((char)(wb.sd << 4)) >> 4;
            int n27 = (int)((char)wb.sd) >> 4;
            int n28 = (int)((char)(wb.se << 4)) >> 4;
            int n29 = (int)((char)wb.se) >> 4;
            int n30 = (int)((char)(wb.sf << 4)) >> 4;
            int n31 = (int)((char)wb.sf) >> 4;

            acc1 += (float)n24 * x3.s0 + (float)n25 * x3.s1
                  + (float)n26 * x3.s2 + (float)n27 * x3.s3
                  + (float)n28 * x3.s4 + (float)n29 * x3.s5
                  + (float)n30 * x3.s6 + (float)n31 * x3.s7;
        }

        total_sum += (acc0 + acc1) * scale;
    }

    float wt = top_wt[k_slot];
    float scaled = total_sum * wt;
    if (k_slot == 0) {
        acc[m] = scaled;
    } else {
        acc[m] += scaled;
    }
}

__kernel void int4_gemv_m1_add_scaled(
    __global float * restrict y,              // [M] (d_moe_acc_)
    __global const uchar * restrict w_packed, // [M, K / 2]
    __global const ushort * restrict w_scale, // [M, K / 128]
    __global const float * restrict x,        // [K]
    __global const float * restrict scale_ptr,// [1] (points to d_sh_gate_val)
    int M,
    int K
) {
    int m = get_global_id(0);
    if (m >= M) return;

    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w = w_packed + (size_t)m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)m * num_groups;

    float total_sum = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        float scale = bf16_to_fp32(row_s[g]);
        __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);
        __global const float *grp_x = x + g * GROUP_SIZE;

        __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;
        __global const float8 *x_vec8 = (__global const float8 *)grp_x;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uchar16 wb = w_vec16[v];

            float8 x0 = x_vec8[v * 4 + 0];
            float8 x1 = x_vec8[v * 4 + 1];
            float8 x2 = x_vec8[v * 4 + 2];
            float8 x3 = x_vec8[v * 4 + 3];

            int n0  = (int)((char)(wb.s0 << 4)) >> 4;
            int n1  = (int)((char)wb.s0) >> 4;
            int n2  = (int)((char)(wb.s1 << 4)) >> 4;
            int n3  = (int)((char)wb.s1) >> 4;
            int n4  = (int)((char)(wb.s2 << 4)) >> 4;
            int n5  = (int)((char)wb.s2) >> 4;
            int n6  = (int)((char)(wb.s3 << 4)) >> 4;
            int n7  = (int)((char)wb.s3) >> 4;

            acc0 += (float)n0 * x0.s0 + (float)n1 * x0.s1
                  + (float)n2 * x0.s2 + (float)n3 * x0.s3
                  + (float)n4 * x0.s4 + (float)n5 * x0.s5
                  + (float)n6 * x0.s6 + (float)n7 * x0.s7;

            int n8  = (int)((char)(wb.s4 << 4)) >> 4;
            int n9  = (int)((char)wb.s4) >> 4;
            int n10 = (int)((char)(wb.s5 << 4)) >> 4;
            int n11 = (int)((char)wb.s5) >> 4;
            int n12 = (int)((char)(wb.s6 << 4)) >> 4;
            int n13 = (int)((char)wb.s6) >> 4;
            int n14 = (int)((char)(wb.s7 << 4)) >> 4;
            int n15 = (int)((char)wb.s7) >> 4;

            acc1 += (float)n8 * x1.s0 + (float)n9 * x1.s1
                  + (float)n10 * x1.s2 + (float)n11 * x1.s3
                  + (float)n12 * x1.s4 + (float)n13 * x1.s5
                  + (float)n14 * x1.s6 + (float)n15 * x1.s7;

            int n16 = (int)((char)(wb.s8 << 4)) >> 4;
            int n17 = (int)((char)wb.s8) >> 4;
            int n18 = (int)((char)(wb.s9 << 4)) >> 4;
            int n19 = (int)((char)wb.s9) >> 4;
            int n20 = (int)((char)(wb.sa << 4)) >> 4;
            int n21 = (int)((char)wb.sa) >> 4;
            int n22 = (int)((char)(wb.sb << 4)) >> 4;
            int n23 = (int)((char)wb.sb) >> 4;

            acc0 += (float)n16 * x2.s0 + (float)n17 * x2.s1
                  + (float)n18 * x2.s2 + (float)n19 * x2.s3
                  + (float)n20 * x2.s4 + (float)n21 * x2.s5
                  + (float)n22 * x2.s6 + (float)n23 * x2.s7;

            int n24 = (int)((char)(wb.sc << 4)) >> 4;
            int n25 = (int)((char)wb.sc) >> 4;
            int n26 = (int)((char)(wb.sd << 4)) >> 4;
            int n27 = (int)((char)wb.sd) >> 4;
            int n28 = (int)((char)(wb.se << 4)) >> 4;
            int n29 = (int)((char)wb.se) >> 4;
            int n30 = (int)((char)(wb.sf << 4)) >> 4;
            int n31 = (int)((char)wb.sf) >> 4;

            acc1 += (float)n24 * x3.s0 + (float)n25 * x3.s1
                  + (float)n26 * x3.s2 + (float)n27 * x3.s3
                  + (float)n28 * x3.s4 + (float)n29 * x3.s5
                  + (float)n30 * x3.s6 + (float)n31 * x3.s7;
        }

        total_sum += (acc0 + acc1) * scale;
    }

    float factor = *scale_ptr;
    y[m] += total_sum * factor;
}

__kernel void argmax_stage2_ctrl(
    __global const float * restrict stage1_max_vals,
    __global const uint * restrict stage1_max_indices,
    __global int * restrict ctrl,              // ctrl[3] = selected_token
    uint num_groups
) {
    int tid = get_local_id(0);
    __local float s_vals[256];
    __local uint s_idxs[256];

    float local_max = -1e30f;
    uint local_idx = 0xFFFFFFFF;

    for (uint i = tid; i < num_groups; i += 256) {
        float v = stage1_max_vals[i];
        uint idx = stage1_max_indices[i];
        if (v > local_max || (v == local_max && idx < local_idx)) {
            local_max = v;
            local_idx = idx;
        }
    }

    s_vals[tid] = local_max;
    s_idxs[tid] = local_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_vals[tid + s] > s_vals[tid] || (s_vals[tid + s] == s_vals[tid] && s_idxs[tid + s] < s_idxs[tid])) {
                s_vals[tid] = s_vals[tid + s];
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        ctrl[3] = (int)s_idxs[0];
    }
}

__kernel void moe_gateup_all8_ctrl(
    __global float * restrict all_gu,         // [8, 1024] = [8192]
    __global const uchar * restrict w_bank,   // [256, 1024, K/2]
    __global const ushort * restrict s_bank,  // [256, 1024, K/128]
    __global const float * restrict x,        // [K = 2048]
    __global const uint * restrict top_idx,   // [8]
    int K                                     // 2048
) {
    int gid = get_global_id(0); // 0 .. 8191
    if (gid >= 8 * 1024) return;

    int k_slot = gid / 1024;    // 0 .. 7
    int m = gid % 1024;         // 0 .. 1023

    uint eid = top_idx[k_slot];
    size_t row_idx = (size_t)eid * 1024 + m;
    int num_groups = K / GROUP_SIZE; // 16

    __global const uchar *row_w = w_bank + row_idx * (K / 2);
    __global const ushort *row_s = s_bank + row_idx * num_groups;

    float total_sum = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        float scale = bf16_to_fp32(row_s[g]);
        __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);
        __global const float *grp_x = x + g * GROUP_SIZE;

        __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;
        __global const float8 *x_vec8 = (__global const float8 *)grp_x;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uchar16 wb = w_vec16[v];

            float8 x0 = x_vec8[v * 4 + 0];
            float8 x1 = x_vec8[v * 4 + 1];
            float8 x2 = x_vec8[v * 4 + 2];
            float8 x3 = x_vec8[v * 4 + 3];

            int n0  = (int)((char)(wb.s0 << 4)) >> 4;
            int n1  = (int)((char)wb.s0) >> 4;
            int n2  = (int)((char)(wb.s1 << 4)) >> 4;
            int n3  = (int)((char)wb.s1) >> 4;
            int n4  = (int)((char)(wb.s2 << 4)) >> 4;
            int n5  = (int)((char)wb.s2) >> 4;
            int n6  = (int)((char)(wb.s3 << 4)) >> 4;
            int n7  = (int)((char)wb.s3) >> 4;

            acc0 += (float)n0 * x0.s0 + (float)n1 * x0.s1
                  + (float)n2 * x0.s2 + (float)n3 * x0.s3
                  + (float)n4 * x0.s4 + (float)n5 * x0.s5
                  + (float)n6 * x0.s6 + (float)n7 * x0.s7;

            int n8  = (int)((char)(wb.s4 << 4)) >> 4;
            int n9  = (int)((char)wb.s4) >> 4;
            int n10 = (int)((char)(wb.s5 << 4)) >> 4;
            int n11 = (int)((char)wb.s5) >> 4;
            int n12 = (int)((char)(wb.s6 << 4)) >> 4;
            int n13 = (int)((char)wb.s6) >> 4;
            int n14 = (int)((char)(wb.s7 << 4)) >> 4;
            int n15 = (int)((char)wb.s7) >> 4;

            acc1 += (float)n8  * x1.s0 + (float)n9  * x1.s1
                  + (float)n10 * x1.s2 + (float)n11 * x1.s3
                  + (float)n12 * x1.s4 + (float)n13 * x1.s5
                  + (float)n14 * x1.s6 + (float)n15 * x1.s7;

            int n16 = (int)((char)(wb.s8 << 4)) >> 4;
            int n17 = (int)((char)wb.s8) >> 4;
            int n18 = (int)((char)(wb.s9 << 4)) >> 4;
            int n19 = (int)((char)wb.s9) >> 4;
            int n20 = (int)((char)(wb.sa << 4)) >> 4;
            int n21 = (int)((char)wb.sa) >> 4;
            int n22 = (int)((char)(wb.sb << 4)) >> 4;
            int n23 = (int)((char)wb.sb) >> 4;

            acc0 += (float)n16 * x2.s0 + (float)n17 * x2.s1
                  + (float)n18 * x2.s2 + (float)n19 * x2.s3
                  + (float)n20 * x2.s4 + (float)n21 * x2.s5
                  + (float)n22 * x2.s6 + (float)n23 * x2.s7;

            int n24 = (int)((char)(wb.sc << 4)) >> 4;
            int n25 = (int)((char)wb.sc) >> 4;
            int n26 = (int)((char)(wb.sd << 4)) >> 4;
            int n27 = (int)((char)wb.sd) >> 4;
            int n28 = (int)((char)(wb.se << 4)) >> 4;
            int n29 = (int)((char)wb.se) >> 4;
            int n30 = (int)((char)(wb.sf << 4)) >> 4;
            int n31 = (int)((char)wb.sf) >> 4;

            acc1 += (float)n24 * x3.s0 + (float)n25 * x3.s1
                  + (float)n26 * x3.s2 + (float)n27 * x3.s3
                  + (float)n28 * x3.s4 + (float)n29 * x3.s5
                  + (float)n30 * x3.s6 + (float)n31 * x3.s7;
        }

        total_sum += (acc0 + acc1) * scale;
    }

    all_gu[gid] = total_sum;
}

__kernel void silu_mul_all8(
    __global float * restrict all_act,         // [8, 512] = [4096]
    __global const float * restrict all_gu     // [8, 1024] = [8192]
) {
    int gid = get_global_id(0); // 0 .. 4095
    if (gid >= 8 * 512) return;

    int k_slot = gid / 512;
    int i = gid % 512;

    int gu_base = k_slot * 1024;
    float g_val = all_gu[gu_base + i];
    float u_val = all_gu[gu_base + 512 + i];

    float silu_g = g_val / (1.0f + exp(-g_val));
    all_act[gid] = silu_g * u_val;
}

__kernel void moe_down_accum_all8_ctrl(
    __global float * restrict acc,             // [2048]
    __global const uchar * restrict w_bank,    // [256, 2048, 256]
    __global const ushort * restrict s_bank,   // [256, 2048, 4]
    __global const float * restrict all_act,   // [8, 512] = [4096]
    __global const uint * restrict top_idx,    // [8]
    __global const float * restrict top_wt,    // [8]
    int M,                                     // 2048
    int K                                      // 512
) {
    int m = get_global_id(0);
    if (m >= M) return;

    int num_groups = K / GROUP_SIZE; // 4

    float total_expert_sum = 0.0f;

    for (int k = 0; k < 8; ++k) {
        uint eid = top_idx[k];
        float wt = top_wt[k];
        size_t row_idx = (size_t)eid * M + m;

        __global const uchar *row_w = w_bank + row_idx * (K / 2);
        __global const ushort *row_s = s_bank + row_idx * num_groups;
        __global const float *x = all_act + k * 512;

        float k_sum = 0.0f;

        for (int g = 0; g < num_groups; ++g) {
            float scale = bf16_to_fp32(row_s[g]);
            __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);
            __global const float *grp_x = x + g * GROUP_SIZE;

            __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;
            __global const float8 *x_vec8 = (__global const float8 *)grp_x;

            float acc0 = 0.0f;
            float acc1 = 0.0f;

            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                uchar16 wb = w_vec16[v];

                float8 x0 = x_vec8[v * 4 + 0];
                float8 x1 = x_vec8[v * 4 + 1];
                float8 x2 = x_vec8[v * 4 + 2];
                float8 x3 = x_vec8[v * 4 + 3];

                int n0  = (int)((char)(wb.s0 << 4)) >> 4;
                int n1  = (int)((char)wb.s0) >> 4;
                int n2  = (int)((char)(wb.s1 << 4)) >> 4;
                int n3  = (int)((char)wb.s1) >> 4;
                int n4  = (int)((char)(wb.s2 << 4)) >> 4;
                int n5  = (int)((char)wb.s2) >> 4;
                int n6  = (int)((char)(wb.s3 << 4)) >> 4;
                int n7  = (int)((char)wb.s3) >> 4;

                acc0 += (float)n0 * x0.s0 + (float)n1 * x0.s1
                      + (float)n2 * x0.s2 + (float)n3 * x0.s3
                      + (float)n4 * x0.s4 + (float)n5 * x0.s5
                      + (float)n6 * x0.s6 + (float)n7 * x0.s7;

                int n8  = (int)((char)(wb.s4 << 4)) >> 4;
                int n9  = (int)((char)wb.s4) >> 4;
                int n10 = (int)((char)(wb.s5 << 4)) >> 4;
                int n11 = (int)((char)wb.s5) >> 4;
                int n12 = (int)((char)(wb.s6 << 4)) >> 4;
                int n13 = (int)((char)wb.s6) >> 4;
                int n14 = (int)((char)(wb.s7 << 4)) >> 4;
                int n15 = (int)((char)wb.s7) >> 4;

                acc1 += (float)n8  * x1.s0 + (float)n9  * x1.s1
                      + (float)n10 * x1.s2 + (float)n11 * x1.s3
                      + (float)n12 * x1.s4 + (float)n13 * x1.s5
                      + (float)n14 * x1.s6 + (float)n15 * x1.s7;

                int n16 = (int)((char)(wb.s8 << 4)) >> 4;
                int n17 = (int)((char)wb.s8) >> 4;
                int n18 = (int)((char)(wb.s9 << 4)) >> 4;
                int n19 = (int)((char)wb.s9) >> 4;
                int n20 = (int)((char)(wb.sa << 4)) >> 4;
                int n21 = (int)((char)wb.sa) >> 4;
                int n22 = (int)((char)(wb.sb << 4)) >> 4;
                int n23 = (int)((char)wb.sb) >> 4;

                acc0 += (float)n16 * x2.s0 + (float)n17 * x2.s1
                      + (float)n18 * x2.s2 + (float)n19 * x2.s3
                      + (float)n20 * x2.s4 + (float)n21 * x2.s5
                      + (float)n22 * x2.s6 + (float)n23 * x2.s7;

                int n24 = (int)((char)(wb.sc << 4)) >> 4;
                int n25 = (int)((char)wb.sc) >> 4;
                int n26 = (int)((char)(wb.sd << 4)) >> 4;
                int n27 = (int)((char)wb.sd) >> 4;
                int n28 = (int)((char)(wb.se << 4)) >> 4;
                int n29 = (int)((char)wb.se) >> 4;
                int n30 = (int)((char)(wb.sf << 4)) >> 4;
                int n31 = (int)((char)wb.sf) >> 4;

                acc1 += (float)n24 * x3.s0 + (float)n25 * x3.s1
                      + (float)n26 * x3.s2 + (float)n27 * x3.s3
                      + (float)n28 * x3.s4 + (float)n29 * x3.s5
                      + (float)n30 * x3.s6 + (float)n31 * x3.s7;
            }

            k_sum += (acc0 + acc1) * scale;
        }

        total_expert_sum += k_sum * wt;
    }

    acc[m] = total_expert_sum;
}

__kernel void int4_gemv_lm_head_argmax1(
    __global float * restrict y,              // [M = 248320]
    __global const uchar * restrict w_packed, // [M, K / 2]
    __global const ushort * restrict w_scale, // [M, K / 128]
    __global const float * restrict x,        // [K = 2048]
    __global float * restrict stage1_vals,    // [970]
    __global uint * restrict stage1_idxs,     // [970]
    int M,
    int K
) {
    int m = get_global_id(0);
    int lid = get_local_id(0);
    int gid = get_group_id(0);

    float total_sum = -1e30f;
    uint my_idx = (m < M) ? (uint)m : 0xFFFFFFFF;

    if (m < M) {
        int num_groups = K / GROUP_SIZE;
        __global const uchar *row_w = w_packed + (size_t)m * (K / 2);
        __global const ushort *row_s = w_scale + (size_t)m * num_groups;

        total_sum = 0.0f;

        for (int g = 0; g < num_groups; ++g) {
            float scale = bf16_to_fp32(row_s[g]);
            __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);
            __global const float *grp_x = x + g * GROUP_SIZE;

            __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;
            __global const float8 *x_vec8 = (__global const float8 *)grp_x;

            float acc0 = 0.0f;
            float acc1 = 0.0f;

            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                uchar16 wb = w_vec16[v];

                float8 x0 = x_vec8[v * 4 + 0];
                float8 x1 = x_vec8[v * 4 + 1];
                float8 x2 = x_vec8[v * 4 + 2];
                float8 x3 = x_vec8[v * 4 + 3];

                int n0  = (int)((char)(wb.s0 << 4)) >> 4;
                int n1  = (int)((char)wb.s0) >> 4;
                int n2  = (int)((char)(wb.s1 << 4)) >> 4;
                int n3  = (int)((char)wb.s1) >> 4;
                int n4  = (int)((char)(wb.s2 << 4)) >> 4;
                int n5  = (int)((char)wb.s2) >> 4;
                int n6  = (int)((char)(wb.s3 << 4)) >> 4;
                int n7  = (int)((char)wb.s3) >> 4;

                acc0 += (float)n0 * x0.s0 + (float)n1 * x0.s1
                      + (float)n2 * x0.s2 + (float)n3 * x0.s3
                      + (float)n4 * x0.s4 + (float)n5 * x0.s5
                      + (float)n6 * x0.s6 + (float)n7 * x0.s7;

                int n8  = (int)((char)(wb.s4 << 4)) >> 4;
                int n9  = (int)((char)wb.s4) >> 4;
                int n10 = (int)((char)(wb.s5 << 4)) >> 4;
                int n11 = (int)((char)wb.s5) >> 4;
                int n12 = (int)((char)(wb.s6 << 4)) >> 4;
                int n13 = (int)((char)wb.s6) >> 4;
                int n14 = (int)((char)(wb.s7 << 4)) >> 4;
                int n15 = (int)((char)wb.s7) >> 4;

                acc1 += (float)n8  * x1.s0 + (float)n9  * x1.s1
                      + (float)n10 * x1.s2 + (float)n11 * x1.s3
                      + (float)n12 * x1.s4 + (float)n13 * x1.s5
                      + (float)n14 * x1.s6 + (float)n15 * x1.s7;

                int n16 = (int)((char)(wb.s8 << 4)) >> 4;
                int n17 = (int)((char)wb.s8) >> 4;
                int n18 = (int)((char)(wb.s9 << 4)) >> 4;
                int n19 = (int)((char)wb.s9) >> 4;
                int n20 = (int)((char)(wb.sa << 4)) >> 4;
                int n21 = (int)((char)wb.sa) >> 4;
                int n22 = (int)((char)(wb.sb << 4)) >> 4;
                int n23 = (int)((char)wb.sb) >> 4;

                acc0 += (float)n16 * x2.s0 + (float)n17 * x2.s1
                      + (float)n18 * x2.s2 + (float)n19 * x2.s3
                      + (float)n20 * x2.s4 + (float)n21 * x2.s5
                      + (float)n22 * x2.s6 + (float)n23 * x2.s7;

                int n24 = (int)((char)(wb.sc << 4)) >> 4;
                int n25 = (int)((char)wb.sc) >> 4;
                int n26 = (int)((char)(wb.sd << 4)) >> 4;
                int n27 = (int)((char)wb.sd) >> 4;
                int n28 = (int)((char)(wb.se << 4)) >> 4;
                int n29 = (int)((char)wb.se) >> 4;
                int n30 = (int)((char)(wb.sf << 4)) >> 4;
                int n31 = (int)((char)wb.sf) >> 4;

                acc1 += (float)n24 * x3.s0 + (float)n25 * x3.s1
                      + (float)n26 * x3.s2 + (float)n27 * x3.s3
                      + (float)n28 * x3.s4 + (float)n29 * x3.s5
                      + (float)n30 * x3.s6 + (float)n31 * x3.s7;
            }

            total_sum += (acc0 + acc1) * scale;
        }

        if (y != NULL) {
            y[m] = total_sum;
        }
    }

    __local float s_val[256];
    __local uint s_idx[256];

    s_val[lid] = total_sum;
    s_idx[lid] = my_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            if (s_val[lid + s] > s_val[lid] || (s_val[lid + s] == s_val[lid] && s_idx[lid + s] < s_idx[lid])) {
                s_val[lid] = s_val[lid + s];
                s_idx[lid] = s_idx[lid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        stage1_vals[gid] = s_val[0];
        stage1_idxs[gid] = s_idx[0];
    }
}

// =========================================================================
// 8. Chunked Prefill Batch Kernels (T3.1 / T5.2)
// =========================================================================

__kernel void embed_gather_batch(
    __global float * restrict out,
    __global const ushort * restrict emb_table,
    __global const int * restrict tokens,
    int B
) {
    int gid = get_global_id(0);
    if (gid >= B * HIDDEN_DIM) return;
    int b = gid / HIDDEN_DIM;
    int d = gid % HIDDEN_DIM;
    int token_id = tokens[b];
    ushort val = emb_table[(size_t)token_id * HIDDEN_DIM + d];
    out[gid] = bf16_to_fp32(val);
}

__kernel void rmsnorm_2048_batch(
    __global float * restrict y,
    __global const float * restrict x,
    __global const float * restrict w,
    int B
) {
    int b = get_group_id(0);
    if (b >= B) return;
    int lid = get_local_id(0);
    __local float local_ss[256];

    __global const float *x_tok = x + (size_t)b * HIDDEN_DIM;
    __global float *y_tok = y + (size_t)b * HIDDEN_DIM;

    float sum_sq = 0.0f;
    for (int j = lid; j < HIDDEN_DIM; j += 256) {
        float val = x_tok[j];
        sum_sq += val * val;
    }
    local_ss[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            local_ss[lid] += local_ss[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_rms = rsqrt(local_ss[0] / (float)HIDDEN_DIM + EPS);

    for (int j = lid; j < HIDDEN_DIM; j += 256) {
        y_tok[j] = x_tok[j] * inv_rms * (1.0f + w[j]);
    }
}

__kernel void conv1d_update_silu_batch(
    __global float * restrict qkv_out,
    __global const float * restrict qkv_in,
    __global float * restrict conv_state,
    __global const float * restrict conv_weight,
    int B
) {
    int c = get_global_id(0);
    if (c >= C_QKV) return;

    int s_base = c * 3;
    float s0 = conv_state[s_base + 0];
    float s1 = conv_state[s_base + 1];
    float s2 = conv_state[s_base + 2];

    int w_base = c * 4;
    float w0 = conv_weight[w_base + 0];
    float w1 = conv_weight[w_base + 1];
    float w2 = conv_weight[w_base + 2];
    float w3 = conv_weight[w_base + 3];

    for (int b = 0; b < B; ++b) {
        float x = qkv_in[(size_t)b * C_QKV + c];
        float sum = s0 * w0 + s1 * w1 + s2 * w2 + x * w3;
        float silu_val = sum / (1.0f + exp(-sum));
        qkv_out[(size_t)b * C_QKV + c] = silu_val;

        s0 = s1;
        s1 = s2;
        s2 = x;
    }

    conv_state[s_base + 0] = s0;
    conv_state[s_base + 1] = s1;
    conv_state[s_base + 2] = s2;
}

__kernel void head_l2_norm_qk_batch(
    __global float * restrict q_out,
    __global float * restrict k_out,
    __global const float * restrict qkv_conv,
    int B
) {
    int group_id = get_group_id(0);
    if (group_id >= B * H_K) return;
    int b = group_id / H_K;
    int h = group_id % H_K;
    int lid = get_local_id(0);

    __local float local_ss_q[128];
    __local float local_ss_k[128];

    size_t tok_base = (size_t)b * C_QKV;
    size_t q_off = tok_base + (size_t)h * S_V + lid;
    size_t k_off = tok_base + 2048 + (size_t)h * S_V + lid;

    float val_q = qkv_conv[q_off];
    float val_k = qkv_conv[k_off];

    local_ss_q[lid] = val_q * val_q;
    local_ss_k[lid] = val_k * val_k;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 64; s > 0; s >>= 1) {
        if (lid < s) {
            local_ss_q[lid] += local_ss_q[lid + s];
            local_ss_k[lid] += local_ss_k[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_norm_q = rsqrt(local_ss_q[0] + EPS);
    float inv_norm_k = rsqrt(local_ss_k[0] + EPS);

    size_t out_off = (size_t)b * (H_K * S_V) + (size_t)h * S_V + lid;
    q_out[out_off] = val_q * inv_norm_q;
    k_out[out_off] = val_k * inv_norm_k;
}

__kernel void gate_prep_batch(
    __global float * restrict g_out,
    __global float * restrict beta_out,
    __global const float * restrict a_in,
    __global const float * restrict b_in,
    __global const float * restrict dt_bias,
    __global const float * restrict A_log,
    int B
) {
    int gid = get_global_id(0);
    if (gid >= B * H_V) return;
    int b = gid / H_V;
    int h = gid % H_V;

    float a_val = a_in[(size_t)b * H_V + h] + dt_bias[h];
    float dt = (a_val > 20.0f) ? a_val : log(1.0f + exp(a_val));
    float gate = -exp(A_log[h]) * dt;
    g_out[gid] = exp(gate);
    beta_out[gid] = 1.0f / (1.0f + exp(-b_in[(size_t)b * H_V + h]));
}

__kernel void deltanet_recurrent_batch(
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

    __local float s_q[S_V];
    __local float s_k[S_V];

    int kh = h / 2;
    __global float * S_h = state + (size_t)h * (S_V * S_V);

    for (int b = 0; b < B; ++b) {
        s_q[j] = q[(size_t)b * (H_K * S_V) + kh * S_V + j];
        s_k[j] = k[(size_t)b * (H_K * S_V) + kh * S_V + j];
        barrier(CLK_LOCAL_MEM_FENCE);

        float v_val = v[(size_t)b * C_QKV + 4096 + h * S_V + j];
        float g_val = g[(size_t)b * H_V + h];
        float b_val = beta[(size_t)b * H_V + h];

        float kv_acc = 0.0f;
        for (int i = 0; i < S_V; ++i) {
            kv_acc += S_h[i * S_V + j] * s_k[i];
        }
        float kv_j = kv_acc * g_val;
        float delta_j = (v_val - kv_j) * b_val;

        float o_acc = 0.0f;
        for (int i = 0; i < S_V; ++i) {
            float s_old = S_h[i * S_V + j];
            float s_new = g_val * s_old + s_k[i] * delta_j;
            S_h[i * S_V + j] = s_new;
            o_acc += s_new * s_q[i];
        }

        out[(size_t)b * (H_V * S_V) + h * S_V + j] = o_acc * SCALE_128;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void deltanet_head_norm_silu_z_batch(
    __global float * restrict final_out,
    __global const float * restrict attn_out,
    __global const float * restrict z_gate,
    __global const float * restrict ssm_norm_w,
    int B
) {
    int h = get_group_id(0);
    if (h >= B * H_V) return;
    int j = get_local_id(0);

    __local float local_ss[128];
    __global const float * x_head = attn_out + (size_t)h * S_V;
    __global const float * z_head = z_gate + (size_t)h * S_V;
    __global float * y_head = final_out + (size_t)h * S_V;

    float val = x_head[j];
    local_ss[j] = val * val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 64; s > 0; s >>= 1) {
        if (j < s) {
            local_ss[j] += local_ss[j + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_rms = rsqrt(local_ss[0] / 128.0f + EPS);
    float norm_val = val * inv_rms * ssm_norm_w[j];
    float zv = z_head[j];
    float silu_z = zv / (1.0f + exp(-zv));
    y_head[j] = norm_val * silu_z;
}

__kernel void deinterleave_q_gate_batch(
    __global float * restrict q_out,
    __global float * restrict gate_out,
    __global const float * restrict q_proj_in,
    int B
) {
    int gid = get_global_id(0);
    if (gid >= B * NUM_Q_HEADS * HEAD_DIM) return;
    int b = gid / (NUM_Q_HEADS * HEAD_DIM);
    int rem = gid % (NUM_Q_HEADS * HEAD_DIM);
    int h = rem / HEAD_DIM;
    int d = rem % HEAD_DIM;

    size_t in_base = (size_t)b * (2 * NUM_Q_HEADS * HEAD_DIM) + h * (2 * HEAD_DIM);
    q_out[gid] = q_proj_in[in_base + d];
    gate_out[gid] = q_proj_in[in_base + HEAD_DIM + d];
}

__kernel void rope_and_kv_append_batch(
    __global float * restrict q,
    __global float * restrict k,
    __global const float * restrict v,
    __global ushort * restrict k_cache,
    __global ushort * restrict v_cache,
    __global const int * restrict ctrl,
    uint max_ctx,
    int B
) {
    int b = get_group_id(0);
    if (b >= B) return;
    uint pos = (uint)ctrl[1] + (uint)b;
    int gid = get_local_id(0);
    __local float s_buf[HEAD_DIM];

    float cos_t = 1.0f;
    float sin_t = 0.0f;

    if (gid < ROTARY_HALF) {
        float inv_freq = pow(ROPE_THETA, -((float)gid / (float)ROTARY_HALF));
        float theta = (float)pos * inv_freq;
        cos_t = cos(theta);
        sin_t = sin(theta);
    } else if (gid < ROTARY_DIM) {
        int i = gid - ROTARY_HALF;
        float inv_freq = pow(ROPE_THETA, -((float)i / (float)ROTARY_HALF));
        float theta = (float)pos * inv_freq;
        cos_t = cos(theta);
        sin_t = sin(theta);
    }

    __global float *q_tok = q + (size_t)b * (NUM_Q_HEADS * HEAD_DIM);
    for (int h = 0; h < NUM_Q_HEADS; ++h) {
        __global float * q_head = q_tok + h * HEAD_DIM;
        s_buf[gid] = q_head[gid];
        barrier(CLK_LOCAL_MEM_FENCE);

        float rot_val;
        if (gid < ROTARY_HALF) {
            float x0 = s_buf[gid];
            float x1 = s_buf[gid + ROTARY_HALF];
            rot_val = x0 * cos_t - x1 * sin_t;
        } else if (gid < ROTARY_DIM) {
            float x0 = s_buf[gid - ROTARY_HALF];
            float x1 = s_buf[gid];
            rot_val = x0 * sin_t + x1 * cos_t;
        } else {
            rot_val = s_buf[gid];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        q_head[gid] = rot_val;
    }

    __global float *k_tok = k + (size_t)b * (NUM_KV_HEADS * HEAD_DIM);
    __global const float *v_tok = v + (size_t)b * (NUM_KV_HEADS * HEAD_DIM);
    for (int h = 0; h < NUM_KV_HEADS; ++h) {
        __global float * k_head = k_tok + h * HEAD_DIM;
        s_buf[gid] = k_head[gid];
        barrier(CLK_LOCAL_MEM_FENCE);

        float rot_val;
        if (gid < ROTARY_HALF) {
            float x0 = s_buf[gid];
            float x1 = s_buf[gid + ROTARY_HALF];
            rot_val = x0 * cos_t - x1 * sin_t;
        } else if (gid < ROTARY_DIM) {
            float x0 = s_buf[gid - ROTARY_HALF];
            float x1 = s_buf[gid];
            rot_val = x0 * sin_t + x1 * cos_t;
        } else {
            rot_val = s_buf[gid];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        k_head[gid] = rot_val;

        __global ushort * k_slot = k_cache + ((size_t)h * max_ctx + pos) * HEAD_DIM;
        __global ushort * v_slot = v_cache + ((size_t)h * max_ctx + pos) * HEAD_DIM;
        k_slot[gid] = float_to_bf16(rot_val);
        v_slot[gid] = float_to_bf16(v_tok[h * HEAD_DIM + gid]);
    }
}

__kernel void gqa_attn_prefill_batch(
    __global float * restrict out,
    __global const float * restrict q,
    __global const float * restrict gate,
    __global const ushort * restrict k_cache,
    __global const ushort * restrict v_cache,
    __global const int * restrict ctrl,
    uint max_ctx,
    int B
) {
    int group_id = get_group_id(0);
    if (group_id >= B * NUM_Q_HEADS) return;
    int b = group_id / NUM_Q_HEADS;
    int qh = group_id % NUM_Q_HEADS;
    int tid = get_local_id(0);

    uint pos = (uint)ctrl[1] + (uint)b;
    int kv_h = qh / GQA_GROUP_SIZE;

    __local float s_q[HEAD_DIM];
    __local float s_red[HEAD_DIM];

    s_q[tid] = q[(size_t)b * (NUM_Q_HEADS * HEAD_DIM) + qh * HEAD_DIM + tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float run_acc = 0.0f;

    uint total_tokens = pos + 1;

    for (uint t = 0; t < total_tokens; ++t) {
        __global const ushort * k_slot = k_cache + ((size_t)kv_h * max_ctx + t) * HEAD_DIM;
        float k_val = bf16_to_float(k_slot[tid]);

        s_red[tid] = s_q[tid] * k_val;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) {
                s_red[tid] += s_red[tid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float score = s_red[0] * ATTN_SCALE;
        barrier(CLK_LOCAL_MEM_FENCE);

        __global const ushort * v_slot = v_cache + ((size_t)kv_h * max_ctx + t) * HEAD_DIM;
        float v_val = bf16_to_float(v_slot[tid]);

        if (score > run_max) {
            float exp_diff = exp(run_max - score);
            run_max = score;
            run_sum = run_sum * exp_diff + 1.0f;
            run_acc = run_acc * exp_diff + v_val;
        } else {
            float exp_diff = exp(score - run_max);
            run_sum += exp_diff;
            run_acc += exp_diff * v_val;
        }
    }

    float attn_val = run_acc / run_sum;
    float g_val = gate[(size_t)b * (NUM_Q_HEADS * HEAD_DIM) + qh * HEAD_DIM + tid];
    float sig_g = 1.0f / (1.0f + exp(-g_val));

    out[(size_t)b * (NUM_Q_HEADS * HEAD_DIM) + qh * HEAD_DIM + tid] = attn_val * sig_g;
}

__kernel void moe_topk_router_batch(
    __global const float * restrict x,
    __global const float * restrict w_gate,
    __global const float * restrict w_shared,
    __global uint * restrict out_topk_indices,
    __global float * restrict out_topk_weights,
    __global float * restrict out_shared_gate,
    int B
) {
    int b = get_group_id(0);
    if (b >= B) return;
    int lid = get_local_id(0);

    __local float local_logits[NUM_EXPERTS];
    __local float local_probs[NUM_EXPERTS];
    __local float local_max;
    __local float local_sum;

    __global const float4 *x_vec = (__global const float4 *)(x + (size_t)b * HIDDEN_DIM);
    __global const float4 *w_vec = (__global const float4 *)(w_gate + (size_t)lid * HIDDEN_DIM);

    float sum = 0.0f;
    for (int j = 0; j < HIDDEN_DIM / 4; ++j) {
        float4 xv = x_vec[j];
        float4 wv = w_vec[j];
        sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }

    local_logits[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float m = local_logits[0];
        for (int i = 1; i < NUM_EXPERTS; ++i) {
            if (local_logits[i] > m) {
                m = local_logits[i];
            }
        }
        local_max = m;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float exp_val = exp(local_logits[lid] - local_max);
    local_probs[lid] = exp_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float s = 0.0f;
        for (int i = 0; i < NUM_EXPERTS; ++i) {
            s += local_probs[i];
        }
        local_sum = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    local_probs[lid] = local_probs[lid] / local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float best_prob[TOP_K];
        uint best_idx[TOP_K];

        for (int k = 0; k < TOP_K; ++k) {
            best_prob[k] = -1.0f;
            best_idx[k] = 0xFFFFFFFF;
        }

        for (uint i = 0; i < NUM_EXPERTS; ++i) {
            float p = local_probs[i];
            if (p > best_prob[TOP_K - 1]) {
                int pos = TOP_K - 1;
                while (pos > 0 && (p > best_prob[pos - 1] || (p == best_prob[pos - 1] && i < best_idx[pos - 1]))) {
                    pos--;
                }
                for (int j = TOP_K - 1; j > pos; --j) {
                    best_prob[j] = best_prob[j - 1];
                    best_idx[j] = best_idx[j - 1];
                }
                best_prob[pos] = p;
                best_idx[pos] = i;
            }
        }

        float top_sum = 0.0f;
        for (int k = 0; k < TOP_K; ++k) {
            top_sum += best_prob[k];
        }

        for (int k = 0; k < TOP_K; ++k) {
            out_topk_indices[b * TOP_K + k] = best_idx[k];
            out_topk_weights[b * TOP_K + k] = best_prob[k] / top_sum;
        }

        if (w_shared != NULL) {
            float sh_sum = 0.0f;
            __global const float4 *ws_vec = (__global const float4 *)w_shared;
            for (int j = 0; j < HIDDEN_DIM / 4; ++j) {
                float4 xv = x_vec[j];
                float4 wv = ws_vec[j];
                sh_sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
            }
            out_shared_gate[b] = 1.0f / (1.0f + exp(-sh_sum));
        }
    }
}

__kernel void moe_gateup_all8_batch(
    __global float * restrict all_gu,
    __global const uchar * restrict w_bank,
    __global const ushort * restrict s_bank,
    __global const float * restrict x,
    __global const uint * restrict top_idx,
    int K,
    int B
) {
    int gid = get_global_id(0);
    if (gid >= B * 8 * 1024) return;

    int b = gid / (8 * 1024);
    int rem = gid % (8 * 1024);
    int k_slot = rem / 1024;
    int m = rem % 1024;

    uint eid = top_idx[b * 8 + k_slot];
    size_t row_idx = (size_t)eid * 1024 + m;
    int num_groups = K / GROUP_SIZE;

    __global const uchar *row_w = w_bank + row_idx * (K / 2);
    __global const ushort *row_s = s_bank + row_idx * num_groups;
    __global const float *x_tok = x + (size_t)b * K;

    float total_sum = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        float scale = bf16_to_fp32(row_s[g]);
        __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);
        __global const float *grp_x = x_tok + g * GROUP_SIZE;

        __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;
        __global const float8 *x_vec8 = (__global const float8 *)grp_x;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uchar16 wb = w_vec16[v];

            float8 x0 = x_vec8[v * 4 + 0];
            float8 x1 = x_vec8[v * 4 + 1];
            float8 x2 = x_vec8[v * 4 + 2];
            float8 x3 = x_vec8[v * 4 + 3];

            int n0  = (int)((char)(wb.s0 << 4)) >> 4;
            int n1  = (int)((char)wb.s0) >> 4;
            int n2  = (int)((char)(wb.s1 << 4)) >> 4;
            int n3  = (int)((char)wb.s1) >> 4;
            int n4  = (int)((char)(wb.s2 << 4)) >> 4;
            int n5  = (int)((char)wb.s2) >> 4;
            int n6  = (int)((char)(wb.s3 << 4)) >> 4;
            int n7  = (int)((char)wb.s3) >> 4;

            acc0 += (float)n0 * x0.s0 + (float)n1 * x0.s1
                  + (float)n2 * x0.s2 + (float)n3 * x0.s3
                  + (float)n4 * x0.s4 + (float)n5 * x0.s5
                  + (float)n6 * x0.s6 + (float)n7 * x0.s7;

            int n8  = (int)((char)(wb.s4 << 4)) >> 4;
            int n9  = (int)((char)wb.s4) >> 4;
            int n10 = (int)((char)(wb.s5 << 4)) >> 4;
            int n11 = (int)((char)wb.s5) >> 4;
            int n12 = (int)((char)(wb.s6 << 4)) >> 4;
            int n13 = (int)((char)wb.s6) >> 4;
            int n14 = (int)((char)(wb.s7 << 4)) >> 4;
            int n15 = (int)((char)wb.s7) >> 4;

            acc1 += (float)n8  * x1.s0 + (float)n9  * x1.s1
                  + (float)n10 * x1.s2 + (float)n11 * x1.s3
                  + (float)n12 * x1.s4 + (float)n13 * x1.s5
                  + (float)n14 * x1.s6 + (float)n15 * x1.s7;

            int n16 = (int)((char)(wb.s8 << 4)) >> 4;
            int n17 = (int)((char)wb.s8) >> 4;
            int n18 = (int)((char)(wb.s9 << 4)) >> 4;
            int n19 = (int)((char)wb.s9) >> 4;
            int n20 = (int)((char)(wb.sa << 4)) >> 4;
            int n21 = (int)((char)wb.sa) >> 4;
            int n22 = (int)((char)(wb.sb << 4)) >> 4;
            int n23 = (int)((char)wb.sb) >> 4;

            acc0 += (float)n16 * x2.s0 + (float)n17 * x2.s1
                  + (float)n18 * x2.s2 + (float)n19 * x2.s3
                  + (float)n20 * x2.s4 + (float)n21 * x2.s5
                  + (float)n22 * x2.s6 + (float)n23 * x2.s7;

            int n24 = (int)((char)(wb.sc << 4)) >> 4;
            int n25 = (int)((char)wb.sc) >> 4;
            int n26 = (int)((char)(wb.sd << 4)) >> 4;
            int n27 = (int)((char)wb.sd) >> 4;
            int n28 = (int)((char)(wb.se << 4)) >> 4;
            int n29 = (int)((char)wb.se) >> 4;
            int n30 = (int)((char)(wb.sf << 4)) >> 4;
            int n31 = (int)((char)wb.sf) >> 4;

            acc1 += (float)n24 * x3.s0 + (float)n25 * x3.s1
                  + (float)n26 * x3.s2 + (float)n27 * x3.s3
                  + (float)n28 * x3.s4 + (float)n29 * x3.s5
                  + (float)n30 * x3.s6 + (float)n31 * x3.s7;
        }

        total_sum += (acc0 + acc1) * scale;
    }

    all_gu[gid] = total_sum;
}

__kernel void silu_mul_all8_batch(
    __global float * restrict all_act,
    __global const float * restrict all_gu,
    int B
) {
    int gid = get_global_id(0);
    if (gid >= B * 8 * 512) return;

    int b = gid / (8 * 512);
    int rem = gid % (8 * 512);
    int k_slot = rem / 512;
    int i = rem % 512;

    size_t gu_base = (size_t)b * (8 * 1024) + k_slot * 1024;
    float g_val = all_gu[gu_base + i];
    float u_val = all_gu[gu_base + 512 + i];

    float silu_g = g_val / (1.0f + exp(-g_val));
    all_act[gid] = silu_g * u_val;
}

__kernel void moe_down_accum_all8_batch(
    __global float * restrict acc,
    __global const uchar * restrict w_bank,
    __global const ushort * restrict s_bank,
    __global const float * restrict all_act,
    __global const uint * restrict top_idx,
    __global const float * restrict top_wt,
    int M,
    int K,
    int B
) {
    int gid = get_global_id(0);
    if (gid >= B * M) return;
    int b = gid / M;
    int m = gid % M;

    int num_groups = K / GROUP_SIZE; // 4
    float total_expert_sum = 0.0f;

    for (int k = 0; k < 8; ++k) {
        uint eid = top_idx[b * 8 + k];
        float wt = top_wt[b * 8 + k];
        size_t row_idx = (size_t)eid * M + m;

        __global const uchar *row_w = w_bank + row_idx * (K / 2);
        __global const ushort *row_s = s_bank + row_idx * num_groups;
        __global const float *x = all_act + (size_t)b * (8 * 512) + k * 512;

        float k_sum = 0.0f;

        for (int g = 0; g < num_groups; ++g) {
            float scale = bf16_to_fp32(row_s[g]);
            __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);
            __global const float *grp_x = x + g * GROUP_SIZE;

            __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;
            __global const float8 *x_vec8 = (__global const float8 *)grp_x;

            float acc0 = 0.0f;
            float acc1 = 0.0f;

            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                uchar16 wb = w_vec16[v];

                float8 x0 = x_vec8[v * 4 + 0];
                float8 x1 = x_vec8[v * 4 + 1];
                float8 x2 = x_vec8[v * 4 + 2];
                float8 x3 = x_vec8[v * 4 + 3];

                int n0  = (int)((char)(wb.s0 << 4)) >> 4;
                int n1  = (int)((char)wb.s0) >> 4;
                int n2  = (int)((char)(wb.s1 << 4)) >> 4;
                int n3  = (int)((char)wb.s1) >> 4;
                int n4  = (int)((char)(wb.s2 << 4)) >> 4;
                int n5  = (int)((char)wb.s2) >> 4;
                int n6  = (int)((char)(wb.s3 << 4)) >> 4;
                int n7  = (int)((char)wb.s3) >> 4;

                acc0 += (float)n0 * x0.s0 + (float)n1 * x0.s1
                      + (float)n2 * x0.s2 + (float)n3 * x0.s3
                      + (float)n4 * x0.s4 + (float)n5 * x0.s5
                      + (float)n6 * x0.s6 + (float)n7 * x0.s7;

                int n8  = (int)((char)(wb.s4 << 4)) >> 4;
                int n9  = (int)((char)wb.s4) >> 4;
                int n10 = (int)((char)(wb.s5 << 4)) >> 4;
                int n11 = (int)((char)wb.s5) >> 4;
                int n12 = (int)((char)(wb.s6 << 4)) >> 4;
                int n13 = (int)((char)wb.s6) >> 4;
                int n14 = (int)((char)(wb.s7 << 4)) >> 4;
                int n15 = (int)((char)wb.s7) >> 4;

                acc1 += (float)n8  * x1.s0 + (float)n9  * x1.s1
                      + (float)n10 * x1.s2 + (float)n11 * x1.s3
                      + (float)n12 * x1.s4 + (float)n13 * x1.s5
                  + (float)n14 * x1.s6 + (float)n15 * x1.s7;

                int n16 = (int)((char)(wb.s8 << 4)) >> 4;
                int n17 = (int)((char)wb.s8) >> 4;
                int n18 = (int)((char)(wb.s9 << 4)) >> 4;
                int n19 = (int)((char)wb.s9) >> 4;
                int n20 = (int)((char)(wb.sa << 4)) >> 4;
                int n21 = (int)((char)wb.sa) >> 4;
                int n22 = (int)((char)(wb.sb << 4)) >> 4;
                int n23 = (int)((char)wb.sb) >> 4;

                acc0 += (float)n16 * x2.s0 + (float)n17 * x2.s1
                      + (float)n18 * x2.s2 + (float)n19 * x2.s3
                      + (float)n20 * x2.s4 + (float)n21 * x2.s5
                      + (float)n22 * x2.s6 + (float)n23 * x2.s7;

                int n24 = (int)((char)(wb.sc << 4)) >> 4;
                int n25 = (int)((char)wb.sc) >> 4;
                int n26 = (int)((char)(wb.sd << 4)) >> 4;
                int n27 = (int)((char)wb.sd) >> 4;
                int n28 = (int)((char)(wb.se << 4)) >> 4;
                int n29 = (int)((char)wb.se) >> 4;
                int n30 = (int)((char)(wb.sf << 4)) >> 4;
                int n31 = (int)((char)wb.sf) >> 4;

                acc1 += (float)n24 * x3.s0 + (float)n25 * x3.s1
                      + (float)n26 * x3.s2 + (float)n27 * x3.s3
                      + (float)n28 * x3.s4 + (float)n29 * x3.s5
                      + (float)n30 * x3.s6 + (float)n31 * x3.s7;
            }

            k_sum += (acc0 + acc1) * scale;
        }

        total_expert_sum += k_sum * wt;
    }

    acc[gid] = total_expert_sum;
}

__kernel void silu_mul_batch(
    __global float * restrict act,
    __global const float * restrict g,
    __global const float * restrict u,
    int count
) {
    int gid = get_global_id(0);
    if (gid >= count) return;
    float g_val = g[gid];
    float u_val = u[gid];
    float silu_g = g_val / (1.0f + exp(-g_val));
    act[gid] = silu_g * u_val;
}

__kernel void block_resadd_moe_batch(
    __global float * restrict x_out,
    __global const float * restrict x_mid,
    __global const float * restrict moe_acc,
    __global const float * restrict sh_down,
    __global const float * restrict sh_gate,
    int B,
    int M
) {
    int gid = get_global_id(0);
    if (gid >= B * M) return;
    int b = gid / M;
    x_out[gid] = x_mid[gid] + moe_acc[gid] + sh_down[gid] * sh_gate[b];
}

__kernel void resadd_batch(
    __global float * restrict out,
    __global const float * restrict a,
    __global const float * restrict b,
    int count
) {
    int gid = get_global_id(0);
    if (gid >= count) return;
    out[gid] = a[gid] + b[gid];
}



