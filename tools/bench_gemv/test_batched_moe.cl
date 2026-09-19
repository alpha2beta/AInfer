#define GROUP_SIZE 128

static inline float bf16_to_fp32(ushort b) {
    uint u = ((uint)b) << 16;
    return as_float(u);
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
