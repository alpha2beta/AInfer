// AInfer Unified SPIR-V Kernel Suite for Qwen3.5-MoE on Intel Arc 140V (Xe2)
// Contains all kernels for both DeltaNet (30 layers) and Full-Attention (10 layers) blocks.

#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define GROUP_SIZE 128
#define HIDDEN_DIM 2048
#define NUM_EXPERTS 256
#define TOP_K 8
#define S_V 128
#define H_V 32
#define H_K 16
#define C_QKV 8192
#define HEAD_DIM 256
#define VOCAB_SIZE 248320
// T9.2: OOB index policy — READS clamp into range (fail-safe, no fault),
// WRITES skip (clamping a write would corrupt a valid slot). Guards are
// no-ops on valid inputs (verified by M4 suite).
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
// 1a. INT4 Symmetric Group-128 Dual-Token GEMV (Speculative Verification)
// Evaluates 2 tokens simultaneously in registers with a single pass over weights.
// =========================================================================
__kernel void int4_gemv_m2(
    __global float * restrict y,              // [2, M] row-major: y[0 * M + m], y[1 * M + m]
    __global const uchar * restrict w_packed, // [M, K / 2]
    __global const ushort * restrict w_scale, // [M, K / 128]
    __global const float * restrict x,        // [2, K] row-major: x[0 * K + k], x[1 * K + k]
    int M,
    int K
) {
    int m = get_global_id(0);
    if (m >= M) return;

    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w = w_packed + (size_t)m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)m * num_groups;

    float total_sum0 = 0.0f;
    float total_sum1 = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        float scale = bf16_to_fp32(row_s[g]);
        __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);
        __global const float *grp_x0 = x + g * GROUP_SIZE;
        __global const float *grp_x1 = x + (size_t)K + g * GROUP_SIZE;

        // GROUP_SIZE = 128 elements = 64 bytes of packed weights
        // Read as 4 x uchar16 (16 bytes = 32 weights per vector load)
        __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;
        __global const float8 *x0_vec8 = (__global const float8 *)grp_x0;
        __global const float8 *x1_vec8 = (__global const float8 *)grp_x1;

        float acc0_0 = 0.0f;
        float acc1_0 = 0.0f;
        float acc0_1 = 0.0f;
        float acc1_1 = 0.0f;

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uchar16 wb = w_vec16[v];

            // 32 weights = 4 float8 chunks
            float8 x0_0 = x0_vec8[v * 4 + 0];
            float8 x1_0 = x0_vec8[v * 4 + 1];
            float8 x2_0 = x0_vec8[v * 4 + 2];
            float8 x3_0 = x0_vec8[v * 4 + 3];

            float8 x0_1 = x1_vec8[v * 4 + 0];
            float8 x1_1 = x1_vec8[v * 4 + 1];
            float8 x2_1 = x1_vec8[v * 4 + 2];
            float8 x3_1 = x1_vec8[v * 4 + 3];

            // Byte 0..3 -> 8 weights for x0
            int n0  = (int)((char)(wb.s0 << 4)) >> 4;
            int n1  = (int)((char)wb.s0) >> 4;
            int n2  = (int)((char)(wb.s1 << 4)) >> 4;
            int n3  = (int)((char)wb.s1) >> 4;
            int n4  = (int)((char)(wb.s2 << 4)) >> 4;
            int n5  = (int)((char)wb.s2) >> 4;
            int n6  = (int)((char)(wb.s3 << 4)) >> 4;
            int n7  = (int)((char)wb.s3) >> 4;

            acc0_0 += (float)n0 * x0_0.s0 + (float)n1 * x0_0.s1
                    + (float)n2 * x0_0.s2 + (float)n3 * x0_0.s3
                    + (float)n4 * x0_0.s4 + (float)n5 * x0_0.s5
                    + (float)n6 * x0_0.s6 + (float)n7 * x0_0.s7;

            acc0_1 += (float)n0 * x0_1.s0 + (float)n1 * x0_1.s1
                    + (float)n2 * x0_1.s2 + (float)n3 * x0_1.s3
                    + (float)n4 * x0_1.s4 + (float)n5 * x0_1.s5
                    + (float)n6 * x0_1.s6 + (float)n7 * x0_1.s7;

            // Byte 4..7 -> 8 weights for x1
            int n8  = (int)((char)(wb.s4 << 4)) >> 4;
            int n9  = (int)((char)wb.s4) >> 4;
            int n10 = (int)((char)(wb.s5 << 4)) >> 4;
            int n11 = (int)((char)wb.s5) >> 4;
            int n12 = (int)((char)(wb.s6 << 4)) >> 4;
            int n13 = (int)((char)wb.s6) >> 4;
            int n14 = (int)((char)(wb.s7 << 4)) >> 4;
            int n15 = (int)((char)wb.s7) >> 4;

            acc1_0 += (float)n8  * x1_0.s0 + (float)n9  * x1_0.s1
                    + (float)n10 * x1_0.s2 + (float)n11 * x1_0.s3
                    + (float)n12 * x1_0.s4 + (float)n13 * x1_0.s5
                    + (float)n14 * x1_0.s6 + (float)n15 * x1_0.s7;

            acc1_1 += (float)n8  * x1_1.s0 + (float)n9  * x1_1.s1
                    + (float)n10 * x1_1.s2 + (float)n11 * x1_1.s3
                    + (float)n12 * x1_1.s4 + (float)n13 * x1_1.s5
                    + (float)n14 * x1_1.s6 + (float)n15 * x1_1.s7;

            // Byte 8..11 -> 8 weights for x2
            int n16 = (int)((char)(wb.s8 << 4)) >> 4;
            int n17 = (int)((char)wb.s8) >> 4;
            int n18 = (int)((char)(wb.s9 << 4)) >> 4;
            int n19 = (int)((char)wb.s9) >> 4;
            int n20 = (int)((char)(wb.sa << 4)) >> 4;
            int n21 = (int)((char)wb.sa) >> 4;
            int n22 = (int)((char)(wb.sb << 4)) >> 4;
            int n23 = (int)((char)wb.sb) >> 4;

            acc0_0 += (float)n16 * x2_0.s0 + (float)n17 * x2_0.s1
                    + (float)n18 * x2_0.s2 + (float)n19 * x2_0.s3
                    + (float)n20 * x2_0.s4 + (float)n21 * x2_0.s5
                    + (float)n22 * x2_0.s6 + (float)n23 * x2_0.s7;

            acc0_1 += (float)n16 * x2_1.s0 + (float)n17 * x2_1.s1
                    + (float)n18 * x2_1.s2 + (float)n19 * x2_1.s3
                    + (float)n20 * x2_1.s4 + (float)n21 * x2_1.s5
                    + (float)n22 * x2_1.s6 + (float)n23 * x2_1.s7;

            // Byte 12..15 -> 8 weights for x3
            int n24 = (int)((char)(wb.sc << 4)) >> 4;
            int n25 = (int)((char)wb.sc) >> 4;
            int n26 = (int)((char)(wb.sd << 4)) >> 4;
            int n27 = (int)((char)wb.sd) >> 4;
            int n28 = (int)((char)(wb.se << 4)) >> 4;
            int n29 = (int)((char)wb.se) >> 4;
            int n30 = (int)((char)(wb.sf << 4)) >> 4;
            int n31 = (int)((char)wb.sf) >> 4;

            acc1_0 += (float)n24 * x3_0.s0 + (float)n25 * x3_0.s1
                    + (float)n26 * x3_0.s2 + (float)n27 * x3_0.s3
                    + (float)n28 * x3_0.s4 + (float)n29 * x3_0.s5
                    + (float)n30 * x3_0.s6 + (float)n31 * x3_0.s7;

            acc1_1 += (float)n24 * x3_1.s0 + (float)n25 * x3_1.s1
                    + (float)n26 * x3_1.s2 + (float)n27 * x3_1.s3
                    + (float)n28 * x3_1.s4 + (float)n29 * x3_1.s5
                    + (float)n30 * x3_1.s6 + (float)n31 * x3_1.s7;
        }

        total_sum0 += (acc0_0 + acc1_0) * scale;
        total_sum1 += (acc0_1 + acc1_1) * scale;
    }

    y[m] = total_sum0;
    y[(size_t)M + m] = total_sum1;
}

// =========================================================================
// 1b. INT4 Symmetric Group-128 Batched Prefill GEMM (DPAS Systolic Xe2)
// =========================================================================
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void int4_gemm_prefill(
    __global float * restrict Y,              // [B, M] row-major: Y[b * M + m]
    __global const uchar * restrict w_packed, // [M, K / 2]
    __global const ushort * restrict w_scale, // [M, K / 128]
    __global const float * restrict X,        // [B, K] row-major: X[b * K + k]
    int M,
    int K,
    int B
) {
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_m = get_group_id(0);
    int grp_b = get_group_id(1);
    int lid = get_sub_group_local_id(); // 0..15

    // Each subgroup computes an M_tile of 16 output channels
    int m_tile_idx = grp_m * num_sg + sg_id;
    int m_base = m_tile_idx * 16;
    if (grp_m * 128 >= M) return;

    int m = m_base + lid;
    int safe_m = (m < M) ? m : 0;
    int num_groups = K / GROUP_SIZE;

    __global const uchar *row_w = w_packed + (size_t)safe_m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)safe_m * num_groups;

    int b_base = grp_b * 32;
    if (b_base >= B) return;

    int cur_B0 = (B - b_base > 0) ? min(8, B - b_base) : 0;
    int cur_B1 = (B - (b_base + 8) > 0) ? min(8, B - (b_base + 8)) : 0;
    int cur_B2 = (B - (b_base + 16) > 0) ? min(8, B - (b_base + 16)) : 0;
    int cur_B3 = (B - (b_base + 24) > 0) ? min(8, B - (b_base + 24)) : 0;

    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    int tid = get_local_id(0); // 0..127
    int tok_idx = tid / 4;      // 0..31
    int k_sub = (tid % 4) * 4;  // 0, 4, 8, 12
    int b_curr = b_base + tok_idx;

    __local half s_x[2][32][16];

    int total_steps = num_groups * 8; // (K / 128) * 8

    // Prefetch step 0 into s_x[0]
    float4 xv0 = (b_curr < B) ? vload4(0, X + (size_t)b_curr * K + k_sub) : (float4)(0.0f);
    s_x[0][tok_idx][k_sub + 0] = (half)xv0.x;
    s_x[0][tok_idx][k_sub + 1] = (half)xv0.y;
    s_x[0][tok_idx][k_sub + 2] = (half)xv0.z;
    s_x[0][tok_idx][k_sub + 3] = (half)xv0.w;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 0; s < total_steps; ++s) {
        int cur_buf = s & 1;
        int next_buf = (s + 1) & 1;
        int g = s / 8;

        // Asynchronously prefetch step s + 1 into s_x[next_buf]
        if (s + 1 < total_steps) {
            int next_k = (s + 1) * 16;
            float4 xv_next = (b_curr < B) ? vload4(0, X + (size_t)b_curr * K + next_k + k_sub) : (float4)(0.0f);
            s_x[next_buf][tok_idx][k_sub + 0] = (half)xv_next.x;
            s_x[next_buf][tok_idx][k_sub + 1] = (half)xv_next.y;
            s_x[next_buf][tok_idx][k_sub + 2] = (half)xv_next.z;
            s_x[next_buf][tok_idx][k_sub + 3] = (half)xv_next.w;
        }

        // 1. Thread lid loads 16 weights (8 bytes) for its row safe_m ONCE
        float s_val = bf16_to_fp32(row_s[g]);
        half s_half = (half)s_val;

        __global const uchar *w_ptr = row_w + (size_t)s * 8;
        uchar8 raw_w = *((__global const uchar8 *)w_ptr);

        // Unpack 16 nibbles to 16 halves and scale ONCE
        half w_deq[16];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uchar byte_val = ((uchar *)&raw_w)[i];
            int n0 = (int)((char)(byte_val << 4)) >> 4;
            int n1 = (int)((char)byte_val) >> 4;
            w_deq[2 * i]     = (half)((float)n0) * s_half;
            w_deq[2 * i + 1] = (half)((float)n1) * s_half;
        }

        int8 b_mat;
        __builtin_memcpy(&b_mat, w_deq, 32);

        // 2. Load activation slices from SLM and issue DPAS
        short8 a_mat0 = (short8)(0);
        #pragma unroll
        for (int bi = 0; bi < 8; ++bi) {
            if (bi < cur_B0) ((short *)&a_mat0)[bi] = as_short(s_x[cur_buf][bi][lid]);
        }
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat0, b_mat, acc0);

        if (cur_B1 > 0) {
            short8 a_mat1 = (short8)(0);
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B1) ((short *)&a_mat1)[bi] = as_short(s_x[cur_buf][8 + bi][lid]);
            }
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat1, b_mat, acc1);
        }

        if (cur_B2 > 0) {
            short8 a_mat2 = (short8)(0);
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B2) ((short *)&a_mat2)[bi] = as_short(s_x[cur_buf][16 + bi][lid]);
            }
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat2, b_mat, acc2);
        }

        if (cur_B3 > 0) {
            short8 a_mat3 = (short8)(0);
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B3) ((short *)&a_mat3)[bi] = as_short(s_x[cur_buf][24 + bi][lid]);
            }
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat3, b_mat, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store outputs: thread lid writes output channel m = m_base + lid for each batch row
    if (m < M) {
        #pragma unroll
        for (int bi = 0; bi < 8; ++bi) {
            if (bi < cur_B0) Y[(size_t)(b_base + bi) * M + m] = ((float *)&acc0)[bi];
        }
        if (cur_B1 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B1) Y[(size_t)(b_base + 8 + bi) * M + m] = ((float *)&acc1)[bi];
            }
        }
        if (cur_B2 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B2) Y[(size_t)(b_base + 16 + bi) * M + m] = ((float *)&acc2)[bi];
            }
        }
        if (cur_B3 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B3) Y[(size_t)(b_base + 24 + bi) * M + m] = ((float *)&acc3)[bi];
            }
        }
    }
}

// =========================================================================
// 1c. INT4 Group-128 Batched Prefill GEMM v2: identical numerics to
// int4_gemm_prefill, but each iteration processes 2 K-slices (32 K
// elements) instead of 1, halving workgroup barriers per launch and
// doubling DPAS issue density. Staging is double-buffered over pairs
// (4KB SLM). Env-gated at runtime (v1 is the default fallback).
// =========================================================================
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void int4_gemm_prefill_v2(
    __global float * restrict Y,              // [B, M] row-major: Y[b * M + m]
    __global const uchar * restrict w_packed, // [M, K / 2]
    __global const ushort * restrict w_scale, // [M, K / 128]
    __global const float * restrict X,        // [B, K] row-major: X[b * K + k]
    int M,
    int K,
    int B
) {
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_m = get_group_id(0);
    int grp_b = get_group_id(1);
    int lid = get_sub_group_local_id(); // 0..15

    int m_tile_idx = grp_m * num_sg + sg_id;
    int m_base = m_tile_idx * 16;
    if (grp_m * 128 >= M) return;

    int m = m_base + lid;
    int safe_m = (m < M) ? m : 0;
    int num_groups = K / GROUP_SIZE;

    __global const uchar *row_w = w_packed + (size_t)safe_m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)safe_m * num_groups;

    int b_base = grp_b * 32;
    if (b_base >= B) return;

    int cur_B0 = (B - b_base > 0) ? min(8, B - b_base) : 0;
    int cur_B1 = (B - (b_base + 8) > 0) ? min(8, B - (b_base + 8)) : 0;
    int cur_B2 = (B - (b_base + 16) > 0) ? min(8, B - (b_base + 16)) : 0;
    int cur_B3 = (B - (b_base + 24) > 0) ? min(8, B - (b_base + 24)) : 0;

    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    int tid = get_local_id(0); // 0..127
    int tok_idx = tid / 4;      // 0..31
    int k_sub = (tid % 4) * 4;  // 0, 4, 8, 12
    int b_curr = b_base + tok_idx;

    __local half s_x[2][2][32][16]; // [buf][slice-in-pair][tok][k]

    int total_steps = num_groups * 8; // (K / 128) * 8, always even
    int total_pairs = total_steps / 2;

    // Prologue: stage pair 0 (slices 0,1) into bufs[0]
    #pragma unroll
    for (int ps = 0; ps < 2; ++ps) {
        float4 xv0 = (b_curr < B) ? vload4(0, X + (size_t)b_curr * K + ps * 16 + k_sub) : (float4)(0.0f);
        s_x[0][ps][tok_idx][k_sub + 0] = (half)xv0.x;
        s_x[0][ps][tok_idx][k_sub + 1] = (half)xv0.y;
        s_x[0][ps][tok_idx][k_sub + 2] = (half)xv0.z;
        s_x[0][ps][tok_idx][k_sub + 3] = (half)xv0.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int pr = 0; pr < total_pairs; ++pr) {
        int cur_buf = pr & 1;
        int next_buf = (pr + 1) & 1;

        // Prefetch pair pr+1 into the idle buffer set (disjoint from the
        // pair under compute, so no hazard without an extra barrier).
        if (pr + 1 < total_pairs) {
            #pragma unroll
            for (int ps = 0; ps < 2; ++ps) {
                int next_k = (pr * 2 + 2 + ps) * 16;
                float4 xv_next = (b_curr < B) ? vload4(0, X + (size_t)b_curr * K + next_k + k_sub) : (float4)(0.0f);
                s_x[next_buf][ps][tok_idx][k_sub + 0] = (half)xv_next.x;
                s_x[next_buf][ps][tok_idx][k_sub + 1] = (half)xv_next.y;
                s_x[next_buf][ps][tok_idx][k_sub + 2] = (half)xv_next.z;
                s_x[next_buf][ps][tok_idx][k_sub + 3] = (half)xv_next.w;
            }
        }

        // Compute both slices of pair pr (8 DPAS issue back-to-back).
        #pragma unroll
        for (int ps = 0; ps < 2; ++ps) {
            int sl = pr * 2 + ps;
            int g = sl / 8;

            float s_val = bf16_to_fp32(row_s[g]);
            half s_half = (half)s_val;

            __global const uchar *w_ptr = row_w + (size_t)sl * 8;
            uchar8 raw_w = *((__global const uchar8 *)w_ptr);

            half w_deq[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uchar byte_val = ((uchar *)&raw_w)[i];
                int n0 = (int)((char)(byte_val << 4)) >> 4;
                int n1 = (int)((char)byte_val) >> 4;
                w_deq[2 * i]     = (half)((float)n0) * s_half;
                w_deq[2 * i + 1] = (half)((float)n1) * s_half;
            }

            int8 b_mat;
            __builtin_memcpy(&b_mat, w_deq, 32);

            short8 a_mat0 = (short8)(0);
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B0) ((short *)&a_mat0)[bi] = as_short(s_x[cur_buf][ps][bi][lid]);
            }
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat0, b_mat, acc0);

            if (cur_B1 > 0) {
                short8 a_mat1 = (short8)(0);
                #pragma unroll
                for (int bi = 0; bi < 8; ++bi) {
                    if (bi < cur_B1) ((short *)&a_mat1)[bi] = as_short(s_x[cur_buf][ps][8 + bi][lid]);
                }
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat1, b_mat, acc1);
            }

            if (cur_B2 > 0) {
                short8 a_mat2 = (short8)(0);
                #pragma unroll
                for (int bi = 0; bi < 8; ++bi) {
                    if (bi < cur_B2) ((short *)&a_mat2)[bi] = as_short(s_x[cur_buf][ps][16 + bi][lid]);
                }
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat2, b_mat, acc2);
            }

            if (cur_B3 > 0) {
                short8 a_mat3 = (short8)(0);
                #pragma unroll
                for (int bi = 0; bi < 8; ++bi) {
                    if (bi < cur_B3) ((short *)&a_mat3)[bi] = as_short(s_x[cur_buf][ps][24 + bi][lid]);
                }
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat3, b_mat, acc3);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store outputs (identical to v1)
    if (m < M) {
        #pragma unroll
        for (int bi = 0; bi < 8; ++bi) {
            if (bi < cur_B0) Y[(size_t)(b_base + bi) * M + m] = ((float *)&acc0)[bi];
        }
        if (cur_B1 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B1) Y[(size_t)(b_base + 8 + bi) * M + m] = ((float *)&acc1)[bi];
            }
        }
        if (cur_B2 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B2) Y[(size_t)(b_base + 16 + bi) * M + m] = ((float *)&acc2)[bi];
            }
        }
        if (cur_B3 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B3) Y[(size_t)(b_base + 24 + bi) * M + m] = ((float *)&acc3)[bi];
            }
        }
    }
}

// =========================================================================
// 2. Deterministic MoE Top-8 Router
// =========================================================================

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
    if (pos >= max_ctx) return; // T9.2: skip OOB cache write (uniform arg)
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
    if (token_id < 0 || token_id >= VOCAB_SIZE) token_id = 0; // T9.2: clamp OOB embed read
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
    if (pos >= max_ctx) return; // T9.2: skip OOB cache write (uniform: no barrier crossed yet)
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
    if (eid >= NUM_EXPERTS) eid = 0; // T9.2: clamp OOB expert read
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
    if (eid >= NUM_EXPERTS) eid = 0; // T9.2: clamp OOB expert read
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

__kernel void argmax_stage2_ptr(
    __global const float * restrict stage1_max_vals,
    __global const uint * restrict stage1_max_indices,
    __global int * restrict out_token,
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
        *out_token = (int)s_idxs[0];
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
        if (eid >= NUM_EXPERTS) continue; // T9.2: skip OOB expert (no contribution)
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
    if (token_id < 0 || token_id >= VOCAB_SIZE) token_id = 0; // T9.2: clamp OOB embed read
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

    __local float s_q[2][S_V];
    __local float s_k[2][S_V];

    int kh = h / 2;
    __global float * S_h = state + (size_t)h * (S_V * S_V);

    // Load entire column j of state matrix into private registers once
    float s_col[S_V];
    #pragma unroll 4
    for (int i = 0; i < S_V; ++i) {
        s_col[i] = S_h[i * S_V + j];
    }

    // Prefetch token 0
    s_q[0][j] = q[kh * S_V + j];
    s_k[0][j] = k[kh * S_V + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int b = 0; b < B; ++b) {
        int buf_cur = b & 1;
        int buf_next = (b + 1) & 1;

        float v_val = v[(size_t)b * C_QKV + 4096 + h * S_V + j];
        float g_val = g[(size_t)b * H_V + h];
        float b_val = beta[(size_t)b * H_V + h];

        float kv_acc = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < S_V; ++i) {
            kv_acc += s_col[i] * s_k[buf_cur][i];
        }
        float kv_j = kv_acc * g_val;
        float delta_j = (v_val - kv_j) * b_val;

        float o_acc = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < S_V; ++i) {
            float s_old = s_col[i];
            float s_new = g_val * s_old + s_k[buf_cur][i] * delta_j;
            s_col[i] = s_new;
            o_acc += s_new * s_q[buf_cur][i];
        }

        out[(size_t)b * (H_V * S_V) + h * S_V + j] = o_acc * SCALE_128;

        if (b + 1 < B) {
            s_q[buf_next][j] = q[(size_t)(b + 1) * (H_K * S_V) + kh * S_V + j];
            s_k[buf_next][j] = k[(size_t)(b + 1) * (H_K * S_V) + kh * S_V + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write final accumulated state back to global memory once
    #pragma unroll 4
    for (int i = 0; i < S_V; ++i) {
        S_h[i * S_V + j] = s_col[i];
    }
}

// =========================================================================
// DeltaNet recurrence v2: identical numerics to deltanet_recurrent_batch,
// but processes tokens in batches of 4 per workgroup barrier instead of
// 1 (q/k for 4 steps staged in SLM, then 4 sequential register-only
// steps). Barrier cost drops from B to B/4 per launch.
// Requires subgroup size 16 only for consistency; no shuffle used.
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

    // Double-buffered 4-step staging in SLM (2x4KB q + 2x4KB k = 16KB,
    // well within 128KB SLM).
    __local float s_q[2][4][S_V];
    __local float s_k[2][4][S_V];

    int kh = h / 2;
    __global float * S_h = state + (size_t)h * (S_V * S_V);

    float s_col[S_V];
    #pragma unroll 2
    for (int i = 0; i < S_V; ++i) {
        s_col[i] = S_h[i * S_V + j];
    }

    // Double-buffered staging: while computing batch N from buf_cur,
    // stage batch N+1 into buf_nxt (disjoint buffers, no hazard), so only
    // ONE barrier per 4 steps instead of one per step.
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
        // Compute current batch (register-only w.r.t. staged SLM).
        #pragma unroll 1
        for (int s = 0; s < 4; ++s) {
            if (s < steps) {
                int b = b0 + s;
                float v_val = v[(size_t)b * C_QKV + 4096 + h * S_V + j];
                float g_val = g[(size_t)b * H_V + h];
                float b_val = beta[(size_t)b * H_V + h];

                // 4-way split accumulators break the single-chain FMA
                // dependency stall (128 sequential FMAs -> 4x32 parallel).
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
            // Stage next batch into the idle buffer (safe: not read since
            // two iterations ago / never), then publish with one barrier.
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
    if (pos >= max_ctx) return; // T9.2: skip OOB cache write (group-uniform)
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

// =========================================================================
// GQA attention prefill v2: identical numerics to gqa_attn_prefill_batch,
// but the per-position dot-product reduction uses in-register subgroup
// butterfly shuffles (0 barriers) plus one SLM exchange per 4 positions,
// instead of a 256-wide SLM tree reduction (~10 barriers) per position.
// Barrier cost drops from ~10/position to 0.25/position.
// Requires subgroup size 16 (same assumption as moe_*_grouped_batch).
// =========================================================================
// Force subgroup size 16: the shuffle butterfly below assumes 16 lanes
// (IGC may otherwise pick SIMD32 for this light kernel).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gqa_attn_prefill_batch_v2(
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
    int lane = get_sub_group_local_id();
    int sg = get_sub_group_id();

    uint pos = (uint)ctrl[1] + (uint)b;
    int kv_h = qh / GQA_GROUP_SIZE;

    // Private q element for this dim (v1 kept it in SLM but each thread
    // only ever reads its own lane — no sharing, no barrier needed).
    float qv = q[(size_t)b * (NUM_Q_HEADS * HEAD_DIM) + qh * HEAD_DIM + tid];

    __local float s_part[64]; // 16 subgroups x 4 positions

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float run_acc = 0.0f;

    uint total_tokens = pos + 1;

    for (uint tb = 0; tb < total_tokens; tb += 4) {
        float part[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            uint t = tb + (uint)u;
            float kval = 0.0f;
            if (t < total_tokens) {
                __global const ushort * k_slot = k_cache + ((size_t)kv_h * max_ctx + t) * HEAD_DIM;
                kval = bf16_to_float(k_slot[tid]);
            }
            part[u] = qv * kval;
        }
        // Subgroup butterfly reduction over the 16 lanes (register-only).
        // XOR pattern keeps every shuffle id in range.
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            float v = part[u];
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 8));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 4));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 2));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 1));
            part[u] = v;
        }
        if (lane == 0) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) s_part[sg * 4 + u] = part[u];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Every thread sums the 16 subgroup partials per position
        // (register-only, no second reduction pass needed).
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) s += s_part[i * 4 + u];
            uint t = tb + (uint)u;
            if (t < total_tokens) {
                __global const ushort * v_slot = v_cache + ((size_t)kv_h * max_ctx + t) * HEAD_DIM;
                float v_val = bf16_to_float(v_slot[tid]);
                float sc = s * ATTN_SCALE;
                if (sc > run_max) {
                    float ed = exp(run_max - sc);
                    run_max = sc;
                    run_sum = run_sum * ed + 1.0f;
                    run_acc = run_acc * ed + v_val;
                } else {
                    float ed = exp(sc - run_max);
                    run_sum += ed;
                    run_acc += ed * v_val;
                }
            }
        }
    }

    float attn_val = run_acc / run_sum;
    float g_val = gate[(size_t)b * (NUM_Q_HEADS * HEAD_DIM) + qh * HEAD_DIM + tid];
    float sig_g = 1.0f / (1.0f + exp(-g_val));

    out[(size_t)b * (NUM_Q_HEADS * HEAD_DIM) + qh * HEAD_DIM + tid] = attn_val * sig_g;
}

// =========================================================================
// =========================================================================
// 1e. INT4 Group-128 Batched Prefill GEMM v4: identical numerics to v2,
// but M_tile=32 rows per subgroup (each lane handles 2 rows) instead of
// 16. The costly a_mat SLM-gather is built ONCE per token-tile and shared
// across both rows' DPAS (8 DPAS per slice-pair instead of 4 for the same
// setup cost). Group x-dim is (M+255)/256 (256 rows/group); the runtime
// sizes it accordingly when v4 is active. Env-gated (v1/v2 fallbacks kept).
// =========================================================================
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void int4_gemm_prefill_v4(
    __global float * restrict Y,              // [B, M] row-major: Y[b * M + m]
    __global const uchar * restrict w_packed, // [M, K / 2]
    __global const ushort * restrict w_scale, // [M, K / 128]
    __global const float * restrict X,        // [B, K] row-major: X[b * K + k]
    int M,
    int K,
    int B
) {
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_m = get_group_id(0);
    int grp_b = get_group_id(1);
    int lid = get_sub_group_local_id(); // 0..15

    // 32 rows per subgroup (256 rows per 128-thread group)
    int m_tile_idx = grp_m * num_sg + sg_id;
    int m_base = m_tile_idx * 32;
    if (grp_m * 256 >= M) return;

    int m0 = m_base + lid;
    int m1 = m_base + 16 + lid;
    int safe_m0 = (m0 < M) ? m0 : 0;
    int safe_m1 = (m1 < M) ? m1 : 0;
    int num_groups = K / GROUP_SIZE;

    __global const uchar *row_w0 = w_packed + (size_t)safe_m0 * (K / 2);
    __global const uchar *row_w1 = w_packed + (size_t)safe_m1 * (K / 2);
    __global const ushort *row_s0 = w_scale + (size_t)safe_m0 * num_groups;
    __global const ushort *row_s1 = w_scale + (size_t)safe_m1 * num_groups;

    int b_base = grp_b * 32;
    if (b_base >= B) return;

    int cur_B0 = (B - b_base > 0) ? min(8, B - b_base) : 0;
    int cur_B1 = (B - (b_base + 8) > 0) ? min(8, B - (b_base + 8)) : 0;
    int cur_B2 = (B - (b_base + 16) > 0) ? min(8, B - (b_base + 16)) : 0;
    int cur_B3 = (B - (b_base + 24) > 0) ? min(8, B - (b_base + 24)) : 0;

    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);
    float8 acc4 = (float8)(0.0f);
    float8 acc5 = (float8)(0.0f);
    float8 acc6 = (float8)(0.0f);
    float8 acc7 = (float8)(0.0f);

    int tid = get_local_id(0); // 0..127
    int tok_idx = tid / 4;      // 0..31
    int k_sub = (tid % 4) * 4;  // 0, 4, 8, 12
    int b_curr = b_base + tok_idx;

    __local half s_x[2][2][32][16]; // [buf][slice-in-pair][tok][k]

    int total_steps = num_groups * 8; // (K / 128) * 8, always even
    int total_pairs = total_steps / 2;

    // Prologue: stage pair 0 (slices 0,1)
    #pragma unroll
    for (int ps = 0; ps < 2; ++ps) {
        float4 xv0 = (b_curr < B) ? vload4(0, X + (size_t)b_curr * K + ps * 16 + k_sub) : (float4)(0.0f);
        s_x[0][ps][tok_idx][k_sub + 0] = (half)xv0.x;
        s_x[0][ps][tok_idx][k_sub + 1] = (half)xv0.y;
        s_x[0][ps][tok_idx][k_sub + 2] = (half)xv0.z;
        s_x[0][ps][tok_idx][k_sub + 3] = (half)xv0.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int pr = 0; pr < total_pairs; ++pr) {
        int cur_buf = pr & 1;
        int next_buf = (pr + 1) & 1;

        if (pr + 1 < total_pairs) {
            #pragma unroll
            for (int ps = 0; ps < 2; ++ps) {
                int next_k = (pr * 2 + 2 + ps) * 16;
                float4 xv_next = (b_curr < B) ? vload4(0, X + (size_t)b_curr * K + next_k + k_sub) : (float4)(0.0f);
                s_x[next_buf][ps][tok_idx][k_sub + 0] = (half)xv_next.x;
                s_x[next_buf][ps][tok_idx][k_sub + 1] = (half)xv_next.y;
                s_x[next_buf][ps][tok_idx][k_sub + 2] = (half)xv_next.z;
                s_x[next_buf][ps][tok_idx][k_sub + 3] = (half)xv_next.w;
            }
        }

        #pragma unroll
        for (int ps = 0; ps < 2; ++ps) {
            int sl = pr * 2 + ps;
            int g = sl / 8;

            // Unpack row 0
            float s_val0 = bf16_to_fp32(row_s0[g]);
            half s_half0 = (half)s_val0;
            __global const uchar *w_ptr0 = row_w0 + (size_t)sl * 8;
            uchar8 raw_w0 = *((__global const uchar8 *)w_ptr0);
            half w_deq0[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uchar byte_val = ((uchar *)&raw_w0)[i];
                int n0 = (int)((char)(byte_val << 4)) >> 4;
                int n1 = (int)((char)byte_val) >> 4;
                w_deq0[2 * i]     = (half)((float)n0) * s_half0;
                w_deq0[2 * i + 1] = (half)((float)n1) * s_half0;
            }
            int8 b_mat0;
            __builtin_memcpy(&b_mat0, w_deq0, 32);

            // Unpack row 1
            float s_val1 = bf16_to_fp32(row_s1[g]);
            half s_half1 = (half)s_val1;
            __global const uchar *w_ptr1 = row_w1 + (size_t)sl * 8;
            uchar8 raw_w1 = *((__global const uchar8 *)w_ptr1);
            half w_deq1[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uchar byte_val = ((uchar *)&raw_w1)[i];
                int n0 = (int)((char)(byte_val << 4)) >> 4;
                int n1 = (int)((char)byte_val) >> 4;
                w_deq1[2 * i]     = (half)((float)n0) * s_half1;
                w_deq1[2 * i + 1] = (half)((float)n1) * s_half1;
            }
            int8 b_mat1;
            __builtin_memcpy(&b_mat1, w_deq1, 32);

            // a_mat built ONCE, shared by both rows' DPAS
            short8 a_mat0 = (short8)(0);
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B0) ((short *)&a_mat0)[bi] = as_short(s_x[cur_buf][ps][bi][lid]);
            }
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat0, b_mat0, acc0);
            acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat0, b_mat1, acc4);

            if (cur_B1 > 0) {
                short8 a_mat1 = (short8)(0);
                #pragma unroll
                for (int bi = 0; bi < 8; ++bi) {
                    if (bi < cur_B1) ((short *)&a_mat1)[bi] = as_short(s_x[cur_buf][ps][8 + bi][lid]);
                }
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat1, b_mat0, acc1);
                acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat1, b_mat1, acc5);
            }

            if (cur_B2 > 0) {
                short8 a_mat2 = (short8)(0);
                #pragma unroll
                for (int bi = 0; bi < 8; ++bi) {
                    if (bi < cur_B2) ((short *)&a_mat2)[bi] = as_short(s_x[cur_buf][ps][16 + bi][lid]);
                }
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat2, b_mat0, acc2);
                acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat2, b_mat1, acc6);
            }

            if (cur_B3 > 0) {
                short8 a_mat3 = (short8)(0);
                #pragma unroll
                for (int bi = 0; bi < 8; ++bi) {
                    if (bi < cur_B3) ((short *)&a_mat3)[bi] = as_short(s_x[cur_buf][ps][24 + bi][lid]);
                }
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat3, b_mat0, acc3);
                acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat3, b_mat1, acc7);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store both rows (each guarded independently)
    if (m0 < M) {
        #pragma unroll
        for (int bi = 0; bi < 8; ++bi) {
            if (bi < cur_B0) Y[(size_t)(b_base + bi) * M + m0] = ((float *)&acc0)[bi];
        }
        if (cur_B1 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B1) Y[(size_t)(b_base + 8 + bi) * M + m0] = ((float *)&acc1)[bi];
            }
        }
        if (cur_B2 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B2) Y[(size_t)(b_base + 16 + bi) * M + m0] = ((float *)&acc2)[bi];
            }
        }
        if (cur_B3 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B3) Y[(size_t)(b_base + 24 + bi) * M + m0] = ((float *)&acc3)[bi];
            }
        }
    }
    if (m1 < M) {
        #pragma unroll
        for (int bi = 0; bi < 8; ++bi) {
            if (bi < cur_B0) Y[(size_t)(b_base + bi) * M + m1] = ((float *)&acc4)[bi];
        }
        if (cur_B1 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B1) Y[(size_t)(b_base + 8 + bi) * M + m1] = ((float *)&acc5)[bi];
            }
        }
        if (cur_B2 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B2) Y[(size_t)(b_base + 16 + bi) * M + m1] = ((float *)&acc6)[bi];
            }
        }
        if (cur_B3 > 0) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B3) Y[(size_t)(b_base + 24 + bi) * M + m1] = ((float *)&acc7)[bi];
            }
        }
    }
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

__kernel void moe_build_expert_bins(
    __global int * restrict expert_counts,    // [256]
    __global int * restrict expert_offsets,   // [256]
    __global int * restrict sorted_tokens,    // [B * 8]
    __global int * restrict sorted_slots,     // [B * 8]
    __global const uint * restrict top_idx,   // [B * 8]
    int B
) {
    int lid = get_local_id(0); // 0..255 (1 workgroup of 256 threads)
    if (get_group_id(0) > 0) return;

    __local int local_counts[256];
    __local int local_offsets[256];

    // 1. Thread lid counts how many times expert lid appears across all (b, k)
    int total_items = B * 8;
    int count = 0;
    for (int idx = 0; idx < total_items; ++idx) {
        if (top_idx[idx] == (uint)lid) {
            count++;
        }
    }
    local_counts[lid] = count;
    expert_counts[lid] = count;
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Parallel prefix sum (exclusive scan) in local memory
    int my_val = count;
    local_offsets[lid] = my_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < 256; offset <<= 1) {
        int temp = 0;
        if (lid >= offset) {
            temp = local_offsets[lid - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        local_offsets[lid] += temp;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int exclusive_offset = (lid == 0) ? 0 : local_offsets[lid - 1];
    expert_offsets[lid] = exclusive_offset;
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. Scatter token and slot indices into sorted order
    int cur = exclusive_offset;
    for (int idx = 0; idx < total_items; ++idx) {
        if (top_idx[idx] == (uint)lid) {
            sorted_tokens[cur] = idx / 8;
            sorted_slots[cur] = idx % 8;
            cur++;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void moe_gateup_grouped_batch(
    __global float * restrict all_gu,
    __global const uchar * restrict w_bank,
    __global const ushort * restrict s_bank,
    __global const float * restrict x,
    __global const int * restrict expert_counts,
    __global const int * restrict expert_offsets,
    __global const int * restrict sorted_tokens,
    __global const int * restrict sorted_slots,
    int K,
    int B
) {
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_id = get_group_id(0);
    int lid = get_sub_group_local_id(); // 0..15

    __local half s_x[2][32][16];
    __local int s_tok[32];
    __local int s_slot[32];

    // Total workgroups = 256 experts * 8 workgroups = 2048 workgroups (128 threads/group)
    int expert_id = grp_id / 8;
    if (expert_id >= NUM_EXPERTS) return;

    int num_tokens = expert_counts[expert_id];
    if (num_tokens <= 0) return;

    int chunk_m = grp_id % 8;
    int m_tile_idx = chunk_m * num_sg + sg_id;
    int m_base = m_tile_idx * 16;
    if (m_base >= 1024) return;

    int m = m_base + lid;
    size_t row_idx = (size_t)expert_id * 1024 + m;
    int num_groups = K / GROUP_SIZE; // 16 for K=2048

    __global const uchar *row_w = w_bank + row_idx * (K / 2);
    __global const ushort *row_s = s_bank + row_idx * num_groups;
    int start_idx = expert_offsets[expert_id];

    // Loop over tokens routed to this expert in macro-tiles of 32 (4 systolic DPAS tiles of 8)
    for (int t_base = 0; t_base < num_tokens; t_base += 32) {
        int cur_T0 = (num_tokens - t_base > 0) ? min(8, num_tokens - t_base) : 0;
        int cur_T1 = (num_tokens - (t_base + 8) > 0) ? min(8, num_tokens - (t_base + 8)) : 0;
        int cur_T2 = (num_tokens - (t_base + 16) > 0) ? min(8, num_tokens - (t_base + 16)) : 0;
        int cur_T3 = (num_tokens - (t_base + 24) > 0) ? min(8, num_tokens - (t_base + 24)) : 0;
        int total_cur_T = cur_T0 + cur_T1 + cur_T2 + cur_T3;

        float8 acc0 = (float8)(0.0f);
        float8 acc1 = (float8)(0.0f);
        float8 acc2 = (float8)(0.0f);
        float8 acc3 = (float8)(0.0f);

        int tid = get_local_id(0); // 0..127
        int tok_idx = tid / 4;      // 0..31
        int k_sub = (tid % 4) * 4;  // 0, 4, 8, 12
        bool valid_tok = (tok_idx < total_cur_T);

        int b_tok = valid_tok ? sorted_tokens[start_idx + t_base + tok_idx] : 0;

        if (tid < 32) {
            s_tok[tid]  = (tid < total_cur_T) ? sorted_tokens[start_idx + t_base + tid] : 0;
            s_slot[tid] = (tid < total_cur_T) ? sorted_slots[start_idx + t_base + tid] : 0;
        }

        int total_steps = num_groups * 8; // (K / 128) * 8

        // Prefetch step 0 into s_x[0] using 128-bit aligned float4 vector loads
        float4 xv0 = valid_tok ? vload4(0, x + (size_t)b_tok * K + k_sub) : (float4)(0.0f);
        s_x[0][tok_idx][k_sub + 0] = (half)xv0.x;
        s_x[0][tok_idx][k_sub + 1] = (half)xv0.y;
        s_x[0][tok_idx][k_sub + 2] = (half)xv0.z;
        s_x[0][tok_idx][k_sub + 3] = (half)xv0.w;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = 0; s < total_steps; ++s) {
            int cur_buf = s & 1;
            int next_buf = (s + 1) & 1;
            int g = s / 8;

            // Asynchronously prefetch step s + 1
            if (s + 1 < total_steps) {
                int next_k = (s + 1) * 16;
                float4 xv_next = valid_tok ? vload4(0, x + (size_t)b_tok * K + next_k + k_sub) : (float4)(0.0f);
                s_x[next_buf][tok_idx][k_sub + 0] = (half)xv_next.x;
                s_x[next_buf][tok_idx][k_sub + 1] = (half)xv_next.y;
                s_x[next_buf][tok_idx][k_sub + 2] = (half)xv_next.z;
                s_x[next_buf][tok_idx][k_sub + 3] = (half)xv_next.w;
            }

            // 1. Thread lid loads 16 weights (8 bytes) for row m ONCE
            float s_val = bf16_to_fp32(row_s[g]);
            half s_half = (half)s_val;

            __global const uchar *w_ptr = row_w + (size_t)s * 8;
            uchar8 raw_w = *((__global const uchar8 *)w_ptr);

            // Unpack 16 nibbles to 16 halves and scale ONCE
            half w_deq[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uchar byte_val = ((uchar *)&raw_w)[i];
                int n0 = (int)((char)(byte_val << 4)) >> 4;
                int n1 = (int)((char)byte_val) >> 4;
                w_deq[2 * i]     = (half)((float)n0) * s_half;
                w_deq[2 * i + 1] = (half)((float)n1) * s_half;
            }

            int8 b_mat;
            __builtin_memcpy(&b_mat, w_deq, 32);

            // 2. Load activation slice from SLM and issue DPAS for up to 4 systolic tiles (32 tokens)
            short8 a_mat0 = (short8)(0);
            #pragma unroll
            for (int ti = 0; ti < 8; ++ti) {
                if (ti < cur_T0) ((short *)&a_mat0)[ti] = as_short(s_x[cur_buf][ti][lid]);
            }
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat0, b_mat, acc0);

            if (cur_T1 > 0) {
                short8 a_mat1 = (short8)(0);
                #pragma unroll
                for (int ti = 0; ti < 8; ++ti) {
                    if (ti < cur_T1) ((short *)&a_mat1)[ti] = as_short(s_x[cur_buf][8 + ti][lid]);
                }
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat1, b_mat, acc1);
            }

            if (cur_T2 > 0) {
                short8 a_mat2 = (short8)(0);
                #pragma unroll
                for (int ti = 0; ti < 8; ++ti) {
                    if (ti < cur_T2) ((short *)&a_mat2)[ti] = as_short(s_x[cur_buf][16 + ti][lid]);
                }
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat2, b_mat, acc2);
            }

            if (cur_T3 > 0) {
                short8 a_mat3 = (short8)(0);
                #pragma unroll
                for (int ti = 0; ti < 8; ++ti) {
                    if (ti < cur_T3) ((short *)&a_mat3)[ti] = as_short(s_x[cur_buf][24 + ti][lid]);
                }
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat3, b_mat, acc3);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store outputs directly into all_gu for each token and slot
        if (m < 1024) {
            #pragma unroll
            for (int ti = 0; ti < 8; ++ti) {
                if (ti < cur_T0) {
                    all_gu[(size_t)s_tok[ti] * (8 * 1024) + s_slot[ti] * 1024 + m] = ((float *)&acc0)[ti];
                }
                if (ti < cur_T1) {
                    all_gu[(size_t)s_tok[8 + ti] * (8 * 1024) + s_slot[8 + ti] * 1024 + m] = ((float *)&acc1)[ti];
                }
                if (ti < cur_T2) {
                    all_gu[(size_t)s_tok[16 + ti] * (8 * 1024) + s_slot[16 + ti] * 1024 + m] = ((float *)&acc2)[ti];
                }
                if (ti < cur_T3) {
                    all_gu[(size_t)s_tok[24 + ti] * (8 * 1024) + s_slot[24 + ti] * 1024 + m] = ((float *)&acc3)[ti];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
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

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void moe_down_grouped_batch(
    __global float * restrict down_out,
    __global const uchar * restrict w_bank,
    __global const ushort * restrict s_bank,
    __global const float * restrict all_act,
    __global const int * restrict expert_counts,
    __global const int * restrict expert_offsets,
    __global const int * restrict sorted_tokens,
    __global const int * restrict sorted_slots,
    int K,
    int B
) {
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_id = get_group_id(0);
    int lid = get_sub_group_local_id(); // 0..15

    __local half s_act[2][32][16];
    __local int s_tok[32];
    __local int s_slot[32];

    // Total workgroups = 256 experts * 16 workgroups = 4096 workgroups (128 threads/group)
    int expert_id = grp_id / 16;
    if (expert_id >= NUM_EXPERTS) return;

    int num_tokens = expert_counts[expert_id];
    if (num_tokens <= 0) return;

    int chunk_m = grp_id % 16; // covers 128 rows
    int m_tile_idx = chunk_m * num_sg + sg_id;
    int m_base = m_tile_idx * 16;
    if (m_base >= 2048) return;

    int m = m_base + lid;
    size_t row_idx = (size_t)expert_id * 2048 + m;
    int num_groups = K / GROUP_SIZE; // 4 for K=512

    __global const uchar *row_w = w_bank + row_idx * (K / 2);
    __global const ushort *row_s = s_bank + row_idx * num_groups;
    int start_idx = expert_offsets[expert_id];

    // Loop over tokens routed to this expert in macro-tiles of 32 (4 systolic DPAS tiles of 8)
    for (int t_base = 0; t_base < num_tokens; t_base += 32) {
        int cur_T0 = (num_tokens - t_base > 0) ? min(8, num_tokens - t_base) : 0;
        int cur_T1 = (num_tokens - (t_base + 8) > 0) ? min(8, num_tokens - (t_base + 8)) : 0;
        int cur_T2 = (num_tokens - (t_base + 16) > 0) ? min(8, num_tokens - (t_base + 16)) : 0;
        int cur_T3 = (num_tokens - (t_base + 24) > 0) ? min(8, num_tokens - (t_base + 24)) : 0;
        int total_cur_T = cur_T0 + cur_T1 + cur_T2 + cur_T3;

        float8 acc0 = (float8)(0.0f);
        float8 acc1 = (float8)(0.0f);
        float8 acc2 = (float8)(0.0f);
        float8 acc3 = (float8)(0.0f);

        int tid = get_local_id(0); // 0..127
        int tok_idx = tid / 4;      // 0..31
        int k_sub = (tid % 4) * 4;  // 0, 4, 8, 12
        bool valid_tok = (tok_idx < total_cur_T);

        int b_tok = valid_tok ? sorted_tokens[start_idx + t_base + tok_idx] : 0;
        int slot  = valid_tok ? sorted_slots[start_idx + t_base + tok_idx] : 0;

        if (tid < 32) {
            s_tok[tid]  = (tid < total_cur_T) ? sorted_tokens[start_idx + t_base + tid] : 0;
            s_slot[tid] = (tid < total_cur_T) ? sorted_slots[start_idx + t_base + tid] : 0;
        }

        int total_steps = num_groups * 8; // (K / 128) * 8

        // Prefetch step 0 into s_act[0] using 128-bit aligned float4 vector loads
        float4 av0 = valid_tok ? vload4(0, all_act + (size_t)b_tok * (8 * 512) + slot * 512 + k_sub) : (float4)(0.0f);
        s_act[0][tok_idx][k_sub + 0] = (half)av0.x;
        s_act[0][tok_idx][k_sub + 1] = (half)av0.y;
        s_act[0][tok_idx][k_sub + 2] = (half)av0.z;
        s_act[0][tok_idx][k_sub + 3] = (half)av0.w;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = 0; s < total_steps; ++s) {
            int cur_buf = s & 1;
            int next_buf = (s + 1) & 1;
            int g = s / 8;

            // Asynchronously prefetch step s + 1
            if (s + 1 < total_steps) {
                int next_k = (s + 1) * 16;
                float4 av_next = valid_tok ? vload4(0, all_act + (size_t)b_tok * (8 * 512) + slot * 512 + next_k + k_sub) : (float4)(0.0f);
                s_act[next_buf][tok_idx][k_sub + 0] = (half)av_next.x;
                s_act[next_buf][tok_idx][k_sub + 1] = (half)av_next.y;
                s_act[next_buf][tok_idx][k_sub + 2] = (half)av_next.z;
                s_act[next_buf][tok_idx][k_sub + 3] = (half)av_next.w;
            }

            // 1. Thread lid loads 16 weights (8 bytes) for row m ONCE
            float s_val = bf16_to_fp32(row_s[g]);
            half s_half = (half)s_val;

            __global const uchar *w_ptr = row_w + (size_t)s * 8;
            uchar8 raw_w = *((__global const uchar8 *)w_ptr);

            // Unpack 16 nibbles to 16 halves and scale ONCE
            half w_deq[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uchar byte_val = ((uchar *)&raw_w)[i];
                int n0 = (int)((char)(byte_val << 4)) >> 4;
                int n1 = (int)((char)byte_val) >> 4;
                w_deq[2 * i]     = (half)((float)n0) * s_half;
                w_deq[2 * i + 1] = (half)((float)n1) * s_half;
            }

            int8 b_mat;
            __builtin_memcpy(&b_mat, w_deq, 32);

            // 2. Load activation slice from SLM and issue DPAS for up to 4 systolic tiles (32 tokens)
            short8 a_mat0 = (short8)(0);
            #pragma unroll
            for (int ti = 0; ti < 8; ++ti) {
                if (ti < cur_T0) ((short *)&a_mat0)[ti] = as_short(s_act[cur_buf][ti][lid]);
            }
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat0, b_mat, acc0);

            if (cur_T1 > 0) {
                short8 a_mat1 = (short8)(0);
                #pragma unroll
                for (int ti = 0; ti < 8; ++ti) {
                    if (ti < cur_T1) ((short *)&a_mat1)[ti] = as_short(s_act[cur_buf][8 + ti][lid]);
                }
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat1, b_mat, acc1);
            }

            if (cur_T2 > 0) {
                short8 a_mat2 = (short8)(0);
                #pragma unroll
                for (int ti = 0; ti < 8; ++ti) {
                    if (ti < cur_T2) ((short *)&a_mat2)[ti] = as_short(s_act[cur_buf][16 + ti][lid]);
                }
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat2, b_mat, acc2);
            }

            if (cur_T3 > 0) {
                short8 a_mat3 = (short8)(0);
                #pragma unroll
                for (int ti = 0; ti < 8; ++ti) {
                    if (ti < cur_T3) ((short *)&a_mat3)[ti] = as_short(s_act[cur_buf][24 + ti][lid]);
                }
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat3, b_mat, acc3);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store outputs directly into down_out for each token and slot
        if (m < 2048) {
            #pragma unroll
            for (int ti = 0; ti < 8; ++ti) {
                if (ti < cur_T0) {
                    down_out[(size_t)s_tok[ti] * (8 * 2048) + s_slot[ti] * 2048 + m] = ((float *)&acc0)[ti];
                }
                if (ti < cur_T1) {
                    down_out[(size_t)s_tok[8 + ti] * (8 * 2048) + s_slot[8 + ti] * 2048 + m] = ((float *)&acc1)[ti];
                }
                if (ti < cur_T2) {
                    down_out[(size_t)s_tok[16 + ti] * (8 * 2048) + s_slot[16 + ti] * 2048 + m] = ((float *)&acc2)[ti];
                }
                if (ti < cur_T3) {
                    down_out[(size_t)s_tok[24 + ti] * (8 * 2048) + s_slot[24 + ti] * 2048 + m] = ((float *)&acc3)[ti];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void moe_accum_down_batch(
    __global float * restrict moe_acc,
    __global const float * restrict down_out,
    __global const float * restrict top_wt,
    int B
) {
    int gid = get_global_id(0);
    if (gid >= B * HIDDEN_DIM) return;
    int b = gid / HIDDEN_DIM;
    int m = gid % HIDDEN_DIM;

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float wt = top_wt[b * 8 + k];
        float val = down_out[(size_t)b * (8 * HIDDEN_DIM) + k * HIDDEN_DIM + m];
        sum += wt * val;
    }
    moe_acc[gid] = sum;
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

// Concat two vectors of length N into Out of length 2*N (MTP embedding + hidden fusion)
__kernel void concat2(
    __global float * restrict out,
    __global const float * restrict a,
    __global const float * restrict b,
    int N
) {
    int gid = get_global_id(0);
    if (gid < N) {
        out[gid] = a[gid];
    } else if (gid < 2 * N) {
        out[gid] = b[gid - N];
    }
}

// =========================================================================
// 9. MTP Dual-Token Verification Kernels (T10.1 Speculative Decoding)
// =========================================================================

__kernel void conv1d_update_silu_m2_spec(
    __global float * restrict qkv_out,
    __global const float * restrict qkv_in,
    __global float * restrict conv_state,
    __global float * restrict conv_snap,
    __global const float * restrict conv_weight
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

    // Token 0 (b = 0)
    float x0 = qkv_in[c];
    float sum0 = s0 * w0 + s1 * w1 + s2 * w2 + x0 * w3;
    float silu0 = sum0 / (1.0f + exp(-sum0));
    qkv_out[c] = silu0;

    // Snapshot intermediate conv state after Token 0
    float snap0 = s1;
    float snap1 = s2;
    float snap2 = x0;
    conv_snap[s_base + 0] = snap0;
    conv_snap[s_base + 1] = snap1;
    conv_snap[s_base + 2] = snap2;

    // Token 1 (b = 1)
    float x1 = qkv_in[(size_t)C_QKV + c];
    float sum1 = snap0 * w0 + snap1 * w1 + snap2 * w2 + x1 * w3;
    float silu1 = sum1 / (1.0f + exp(-sum1));
    qkv_out[(size_t)C_QKV + c] = silu1;

    // Final live state after Token 1 (used if Token 1 accepted)
    conv_state[s_base + 0] = snap1;
    conv_state[s_base + 1] = snap2;
    conv_state[s_base + 2] = x1;
}

__kernel void deltanet_recurrent_m2_spec(
    __global float * restrict out,
    __global float * restrict state,
    __global float * restrict state_snap,
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
    __global float * S_h = state + (size_t)h * (S_V * S_V);
    __global float * S_snap_h = state_snap + (size_t)h * (S_V * S_V);

    // --- Token 0 (b = 0) ---
    s_q[j] = q[(size_t)kh * S_V + j];
    s_k[j] = k[(size_t)kh * S_V + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    float v_val0 = v[(size_t)4096 + h * S_V + j];
    float g_val0 = g[h];
    float b_val0 = beta[h];

    float kv_acc0 = 0.0f;
    for (int i = 0; i < S_V; ++i) {
        kv_acc0 += S_h[i * S_V + j] * s_k[i];
    }
    float kv_j0 = kv_acc0 * g_val0;
    float delta_j0 = (v_val0 - kv_j0) * b_val0;

    float o_acc0 = 0.0f;
    for (int i = 0; i < S_V; ++i) {
        float s_old = S_h[i * S_V + j];
        float s_new = g_val0 * s_old + s_k[i] * delta_j0;
        S_h[i * S_V + j] = s_new;
        S_snap_h[i * S_V + j] = s_new; // Snapshot intermediate recurrent state
        o_acc0 += s_new * s_q[i];
    }
    out[(size_t)h * S_V + j] = o_acc0 * SCALE_128;
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Token 1 (b = 1) ---
    s_q[j] = q[(size_t)(H_K * S_V) + kh * S_V + j];
    s_k[j] = k[(size_t)(H_K * S_V) + kh * S_V + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    float v_val1 = v[(size_t)C_QKV + 4096 + h * S_V + j];
    float g_val1 = g[(size_t)H_V + h];
    float b_val1 = beta[(size_t)H_V + h];

    float kv_acc1 = 0.0f;
    for (int i = 0; i < S_V; ++i) {
        kv_acc1 += S_h[i * S_V + j] * s_k[i];
    }
    float kv_j1 = kv_acc1 * g_val1;
    float delta_j1 = (v_val1 - kv_j1) * b_val1;

    float o_acc1 = 0.0f;
    for (int i = 0; i < S_V; ++i) {
        float s_old = S_h[i * S_V + j];
        float s_new = g_val1 * s_old + s_k[i] * delta_j1;
        S_h[i * S_V + j] = s_new;
        o_acc1 += s_new * s_q[i];
    }
    out[(size_t)(H_V * S_V) + h * S_V + j] = o_acc1 * SCALE_128;
}

__kernel void int4_gemv_m2_lm_head_argmax1(
    __global float * restrict y,              // [2 * M] (optional, can be NULL)
    __global const uchar * restrict w_packed, // [M, K / 2]
    __global const ushort * restrict w_scale, // [M, K / 128]
    __global const float * restrict x,        // [2 * K] (Token 0 at x, Token 1 at x + K)
    __global float * restrict stage1_vals_0,  // [970]
    __global uint * restrict stage1_idxs_0,   // [970]
    __global float * restrict stage1_vals_1,  // [970]
    __global uint * restrict stage1_idxs_1,   // [970]
    int M,
    int K
) {
    int m = get_global_id(0);
    int lid = get_local_id(0);
    int gid = get_group_id(0);

    float total_sum0 = -1e30f;
    float total_sum1 = -1e30f;
    uint my_idx = (m < M) ? (uint)m : 0xFFFFFFFF;

    if (m < M) {
        int num_groups = K / GROUP_SIZE;
        __global const uchar *row_w = w_packed + (size_t)m * (K / 2);
        __global const ushort *row_s = w_scale + (size_t)m * num_groups;

        total_sum0 = 0.0f;
        total_sum1 = 0.0f;

        for (int g = 0; g < num_groups; ++g) {
            float scale = bf16_to_fp32(row_s[g]);
            __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);
            __global const float *grp_x0 = x + g * GROUP_SIZE;
            __global const float *grp_x1 = x + K + g * GROUP_SIZE;

            __global const uchar16 *w_vec16 = (__global const uchar16 *)grp_w;
            __global const float8 *x0_vec8 = (__global const float8 *)grp_x0;
            __global const float8 *x1_vec8 = (__global const float8 *)grp_x1;

            float acc0_0 = 0.0f, acc1_0 = 0.0f;
            float acc0_1 = 0.0f, acc1_1 = 0.0f;

            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                uchar16 wb = w_vec16[v];

                float8 x0_0 = x0_vec8[v * 4 + 0];
                float8 x1_0 = x0_vec8[v * 4 + 1];
                float8 x2_0 = x0_vec8[v * 4 + 2];
                float8 x3_0 = x0_vec8[v * 4 + 3];

                float8 x0_1 = x1_vec8[v * 4 + 0];
                float8 x1_1 = x1_vec8[v * 4 + 1];
                float8 x2_1 = x1_vec8[v * 4 + 2];
                float8 x3_1 = x1_vec8[v * 4 + 3];

                int n0  = (int)((char)(wb.s0 << 4)) >> 4;
                int n1  = (int)((char)wb.s0) >> 4;
                int n2  = (int)((char)(wb.s1 << 4)) >> 4;
                int n3  = (int)((char)wb.s1) >> 4;
                int n4  = (int)((char)(wb.s2 << 4)) >> 4;
                int n5  = (int)((char)wb.s2) >> 4;
                int n6  = (int)((char)(wb.s3 << 4)) >> 4;
                int n7  = (int)((char)wb.s3) >> 4;

                acc0_0 += (float)n0 * x0_0.s0 + (float)n1 * x0_0.s1
                        + (float)n2 * x0_0.s2 + (float)n3 * x0_0.s3
                        + (float)n4 * x0_0.s4 + (float)n5 * x0_0.s5
                        + (float)n6 * x0_0.s6 + (float)n7 * x0_0.s7;

                acc0_1 += (float)n0 * x0_1.s0 + (float)n1 * x0_1.s1
                        + (float)n2 * x0_1.s2 + (float)n3 * x0_1.s3
                        + (float)n4 * x0_1.s4 + (float)n5 * x0_1.s5
                        + (float)n6 * x0_1.s6 + (float)n7 * x0_1.s7;

                int n8  = (int)((char)(wb.s4 << 4)) >> 4;
                int n9  = (int)((char)wb.s4) >> 4;
                int n10 = (int)((char)(wb.s5 << 4)) >> 4;
                int n11 = (int)((char)wb.s5) >> 4;
                int n12 = (int)((char)(wb.s6 << 4)) >> 4;
                int n13 = (int)((char)wb.s6) >> 4;
                int n14 = (int)((char)(wb.s7 << 4)) >> 4;
                int n15 = (int)((char)wb.s7) >> 4;

                acc1_0 += (float)n8  * x1_0.s0 + (float)n9  * x1_0.s1
                        + (float)n10 * x1_0.s2 + (float)n11 * x1_0.s3
                        + (float)n12 * x1_0.s4 + (float)n13 * x1_0.s5
                        + (float)n14 * x1_0.s6 + (float)n15 * x1_0.s7;

                acc1_1 += (float)n8  * x1_1.s0 + (float)n9  * x1_1.s1
                        + (float)n10 * x1_1.s2 + (float)n11 * x1_1.s3
                        + (float)n12 * x1_1.s4 + (float)n13 * x1_1.s5
                        + (float)n14 * x1_1.s6 + (float)n15 * x1_1.s7;

                int n16 = (int)((char)(wb.s8 << 4)) >> 4;
                int n17 = (int)((char)wb.s8) >> 4;
                int n18 = (int)((char)(wb.s9 << 4)) >> 4;
                int n19 = (int)((char)wb.s9) >> 4;
                int n20 = (int)((char)(wb.sa << 4)) >> 4;
                int n21 = (int)((char)wb.sa) >> 4;
                int n22 = (int)((char)(wb.sb << 4)) >> 4;
                int n23 = (int)((char)wb.sb) >> 4;

                acc0_0 += (float)n16 * x2_0.s0 + (float)n17 * x2_0.s1
                        + (float)n18 * x2_0.s2 + (float)n19 * x2_0.s3
                        + (float)n20 * x2_0.s4 + (float)n21 * x2_0.s5
                        + (float)n22 * x2_0.s6 + (float)n23 * x2_0.s7;

                acc0_1 += (float)n16 * x2_1.s0 + (float)n17 * x2_1.s1
                        + (float)n18 * x2_1.s2 + (float)n19 * x2_1.s3
                        + (float)n20 * x2_1.s4 + (float)n21 * x2_1.s5
                        + (float)n22 * x2_1.s6 + (float)n23 * x2_1.s7;

                int n24 = (int)((char)(wb.sc << 4)) >> 4;
                int n25 = (int)((char)wb.sc) >> 4;
                int n26 = (int)((char)(wb.sd << 4)) >> 4;
                int n27 = (int)((char)wb.sd) >> 4;
                int n28 = (int)((char)(wb.se << 4)) >> 4;
                int n29 = (int)((char)wb.se) >> 4;
                int n30 = (int)((char)(wb.sf << 4)) >> 4;
                int n31 = (int)((char)wb.sf) >> 4;

                acc1_0 += (float)n24 * x3_0.s0 + (float)n25 * x3_0.s1
                        + (float)n26 * x3_0.s2 + (float)n27 * x3_0.s3
                        + (float)n28 * x3_0.s4 + (float)n29 * x3_0.s5
                        + (float)n30 * x3_0.s6 + (float)n31 * x3_0.s7;

                acc1_1 += (float)n24 * x3_1.s0 + (float)n25 * x3_1.s1
                        + (float)n26 * x3_1.s2 + (float)n27 * x3_1.s3
                        + (float)n28 * x3_1.s4 + (float)n29 * x3_1.s5
                        + (float)n30 * x3_1.s6 + (float)n31 * x3_1.s7;
            }

            total_sum0 += (acc0_0 + acc1_0) * scale;
            total_sum1 += (acc0_1 + acc1_1) * scale;
        }

        if (y != NULL) {
            y[m] = total_sum0;
            y[(size_t)M + m] = total_sum1;
        }
    }

    __local float s_val0[256];
    __local uint s_idx0[256];
    __local float s_val1[256];
    __local uint s_idx1[256];

    s_val0[lid] = total_sum0;
    s_idx0[lid] = my_idx;
    s_val1[lid] = total_sum1;
    s_idx1[lid] = my_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            if (s_val0[lid + s] > s_val0[lid] || (s_val0[lid + s] == s_val0[lid] && s_idx0[lid + s] < s_idx0[lid])) {
                s_val0[lid] = s_val0[lid + s];
                s_idx0[lid] = s_idx0[lid + s];
            }
            if (s_val1[lid + s] > s_val1[lid] || (s_val1[lid + s] == s_val1[lid] && s_idx1[lid + s] < s_idx1[lid])) {
                s_val1[lid] = s_val1[lid + s];
                s_idx1[lid] = s_idx1[lid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        stage1_vals_0[gid] = s_val0[0];
        stage1_idxs_0[gid] = s_idx0[0];
        stage1_vals_1[gid] = s_val1[0];
        stage1_idxs_1[gid] = s_idx1[0];
    }
}





