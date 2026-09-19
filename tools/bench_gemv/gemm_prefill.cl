// INT4 Symmetric Group-128 Batched Prefill GEMM Kernel for Intel Arc 140V (Xe2)
// Computes Y[B, M] = W[M, K] * X[B, K]^T
// Reuses unpacked INT4 weights across batch B to minimize memory bus traffic.

#define GROUP_SIZE 128

static inline float bf16_to_fp32(ushort b) {
    uint u = ((uint)b) << 16;
    return as_float(u);
}

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

    // Up to 32 batch tokens accumulated in private registers
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

            // Unpack 32 weights from 16 bytes into private registers
            float4 w0, w1, w2, w3, w4, w5, w6, w7;

            // Byte 0..3 -> 8 weights
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

            // Byte 4..7 -> 8 weights
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

            // Byte 8..11 -> 8 weights
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

            // Byte 12..15 -> 8 weights
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

            // Reuse the 8 weight float4 registers across all batch items b
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
