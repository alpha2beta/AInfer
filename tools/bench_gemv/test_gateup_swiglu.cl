#define GROUP_SIZE 128

static inline float bf16_to_fp32(ushort b) {
    uint u = ((uint)b) << 16;
    return as_float(u);
}

__kernel void moe_gateup_swiglu(
    __global float * restrict act,             // [M = 512]
    __global const uchar * restrict w_bank,    // [256, 1024, K/2]
    __global const ushort * restrict s_bank,   // [256, 1024, K/128]
    __global const float * restrict x,         // [K = 2048]
    __global const uint * restrict top_idx,    // [8]
    int k_slot,                                // 0..7
    int M,                                     // 512
    int K                                      // 2048
) {
    int m = get_global_id(0);
    if (m >= M) return;

    uint eid = top_idx[k_slot];
    size_t row_g = (size_t)eid * 1024 + m;
    size_t row_u = (size_t)eid * 1024 + 512 + m;
    int num_groups = K / GROUP_SIZE;

    __global const uchar *g_w = w_bank + row_g * (K / 2);
    __global const ushort *g_s = s_bank + row_g * num_groups;
    __global const uchar *u_w = w_bank + row_u * (K / 2);
    __global const ushort *u_s = s_bank + row_u * num_groups;

    float sum_g = 0.0f;
    float sum_u = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        float sc_g = bf16_to_fp32(g_s[g]);
        float sc_u = bf16_to_fp32(u_s[g]);

        __global const uchar16 *gw16 = (__global const uchar16 *)(g_w + g * (GROUP_SIZE / 2));
        __global const uchar16 *uw16 = (__global const uchar16 *)(u_w + g * (GROUP_SIZE / 2));
        __global const float8 *xv8 = (__global const float8 *)(x + g * GROUP_SIZE);

        float g_acc0 = 0.0f, g_acc1 = 0.0f;
        float u_acc0 = 0.0f, u_acc1 = 0.0f;

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uchar16 g_wb = gw16[v];
            uchar16 u_wb = uw16[v];

            float8 x0 = xv8[v * 4 + 0];
            float8 x1 = xv8[v * 4 + 1];
            float8 x2 = xv8[v * 4 + 2];
            float8 x3 = xv8[v * 4 + 3];

            // Byte 0..3 for x0
            int gn0 = (int)((char)(g_wb.s0 << 4)) >> 4;
            int gn1 = (int)((char)g_wb.s0) >> 4;
            int gn2 = (int)((char)(g_wb.s1 << 4)) >> 4;
            int gn3 = (int)((char)g_wb.s1) >> 4;
            int gn4 = (int)((char)(g_wb.s2 << 4)) >> 4;
            int gn5 = (int)((char)g_wb.s2) >> 4;
            int gn6 = (int)((char)(g_wb.s3 << 4)) >> 4;
            int gn7 = (int)((char)g_wb.s3) >> 4;

            int un0 = (int)((char)(u_wb.s0 << 4)) >> 4;
            int un1 = (int)((char)u_wb.s0) >> 4;
            int un2 = (int)((char)(u_wb.s1 << 4)) >> 4;
            int un3 = (int)((char)u_wb.s1) >> 4;
            int un4 = (int)((char)(u_wb.s2 << 4)) >> 4;
            int un5 = (int)((char)u_wb.s2) >> 4;
            int un6 = (int)((char)(u_wb.s3 << 4)) >> 4;
            int un7 = (int)((char)u_wb.s3) >> 4;

            g_acc0 += (float)gn0 * x0.s0 + (float)gn1 * x0.s1
                    + (float)gn2 * x0.s2 + (float)gn3 * x0.s3
                    + (float)gn4 * x0.s4 + (float)gn5 * x0.s5
                    + (float)gn6 * x0.s6 + (float)gn7 * x0.s7;

            u_acc0 += (float)un0 * x0.s0 + (float)un1 * x0.s1
                    + (float)un2 * x0.s2 + (float)un3 * x0.s3
                    + (float)un4 * x0.s4 + (float)un5 * x0.s5
                    + (float)un6 * x0.s6 + (float)un7 * x0.s7;

            // Byte 4..7 for x1
            int gn8  = (int)((char)(g_wb.s4 << 4)) >> 4;
            int gn9  = (int)((char)g_wb.s4) >> 4;
            int gn10 = (int)((char)(g_wb.s5 << 4)) >> 4;
            int gn11 = (int)((char)g_wb.s5) >> 4;
            int gn12 = (int)((char)(g_wb.s6 << 4)) >> 4;
            int gn13 = (int)((char)g_wb.s6) >> 4;
            int gn14 = (int)((char)(g_wb.s7 << 4)) >> 4;
            int gn15 = (int)((char)g_wb.s7) >> 4;

            int un8  = (int)((char)(u_wb.s4 << 4)) >> 4;
            int un9  = (int)((char)u_wb.s4) >> 4;
            int un10 = (int)((char)(u_wb.s5 << 4)) >> 4;
            int un11 = (int)((char)u_wb.s5) >> 4;
            int un12 = (int)((char)(u_wb.s6 << 4)) >> 4;
            int un13 = (int)((char)u_wb.s6) >> 4;
            int un14 = (int)((char)(u_wb.s7 << 4)) >> 4;
            int un15 = (int)((char)u_wb.s7) >> 4;

            g_acc1 += (float)gn8  * x1.s0 + (float)gn9  * x1.s1
                    + (float)gn10 * x1.s2 + (float)gn11 * x1.s3
                    + (float)gn12 * x1.s4 + (float)gn13 * x1.s5
                    + (float)gn14 * x1.s6 + (float)gn15 * x1.s7;

            u_acc1 += (float)un8  * x1.s0 + (float)un9  * x1.s1
                    + (float)un10 * x1.s2 + (float)un11 * x1.s3
                    + (float)un12 * x1.s4 + (float)un13 * x1.s5
                    + (float)un14 * x1.s6 + (float)un15 * x1.s7;

            // Byte 8..11 for x2
            int gn16 = (int)((char)(g_wb.s8 << 4)) >> 4;
            int gn17 = (int)((char)g_wb.s8) >> 4;
            int gn18 = (int)((char)(g_wb.s9 << 4)) >> 4;
            int gn19 = (int)((char)g_wb.s9) >> 4;
            int gn20 = (int)((char)(g_wb.sa << 4)) >> 4;
            int gn21 = (int)((char)g_wb.sa) >> 4;
            int gn22 = (int)((char)(g_wb.sb << 4)) >> 4;
            int gn23 = (int)((char)g_wb.sb) >> 4;

            int un16 = (int)((char)(u_wb.s8 << 4)) >> 4;
            int un17 = (int)((char)u_wb.s8) >> 4;
            int un18 = (int)((char)(u_wb.s9 << 4)) >> 4;
            int un19 = (int)((char)u_wb.s9) >> 4;
            int un20 = (int)((char)(u_wb.sa << 4)) >> 4;
            int un21 = (int)((char)u_wb.sa) >> 4;
            int un22 = (int)((char)(u_wb.sb << 4)) >> 4;
            int un23 = (int)((char)u_wb.sb) >> 4;

            g_acc0 += (float)gn16 * x2.s0 + (float)gn17 * x2.s1
                    + (float)gn18 * x2.s2 + (float)gn19 * x2.s3
                    + (float)gn20 * x2.s4 + (float)gn21 * x2.s5
                    + (float)gn22 * x2.s6 + (float)gn23 * x2.s7;

            u_acc0 += (float)un16 * x2.s0 + (float)un17 * x2.s1
                    + (float)un18 * x2.s2 + (float)un19 * x2.s3
                    + (float)un20 * x2.s4 + (float)un21 * x2.s5
                    + (float)un22 * x2.s6 + (float)un23 * x2.s7;

            // Byte 12..15 for x3
            int gn24 = (int)((char)(g_wb.sc << 4)) >> 4;
            int gn25 = (int)((char)g_wb.sc) >> 4;
            int gn26 = (int)((char)(g_wb.sd << 4)) >> 4;
            int gn27 = (int)((char)g_wb.sd) >> 4;
            int gn28 = (int)((char)(g_wb.se << 4)) >> 4;
            int gn29 = (int)((char)g_wb.se) >> 4;
            int gn30 = (int)((char)(g_wb.sf << 4)) >> 4;
            int gn31 = (int)((char)g_wb.sf) >> 4;

            int un24 = (int)((char)(u_wb.sc << 4)) >> 4;
            int un25 = (int)((char)u_wb.sc) >> 4;
            int un26 = (int)((char)(u_wb.sd << 4)) >> 4;
            int un27 = (int)((char)u_wb.sd) >> 4;
            int un28 = (int)((char)(u_wb.se << 4)) >> 4;
            int un29 = (int)((char)u_wb.se) >> 4;
            int un30 = (int)((char)(u_wb.sf << 4)) >> 4;
            int un31 = (int)((char)u_wb.sf) >> 4;

            g_acc1 += (float)gn24 * x3.s0 + (float)gn25 * x3.s1
                    + (float)gn26 * x3.s2 + (float)gn27 * x3.s3
                    + (float)gn28 * x3.s4 + (float)gn29 * x3.s5
                    + (float)gn30 * x3.s6 + (float)gn31 * x3.s7;

            u_acc1 += (float)un24 * x3.s0 + (float)un25 * x3.s1
                    + (float)un26 * x3.s2 + (float)un27 * x3.s3
                    + (float)un28 * x3.s4 + (float)un29 * x3.s5
                    + (float)un30 * x3.s6 + (float)un31 * x3.s7;
        }

        sum_g += (g_acc0 + g_acc1) * sc_g;
        sum_u += (u_acc0 + u_acc1) * sc_u;
    }

    float sig_g = 1.0f / (1.0f + exp(-sum_g));
    act[m] = (sum_g * sig_g) * sum_u;
}
