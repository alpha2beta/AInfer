#define GROUP_SIZE 128

static inline float bf16_to_fp32(ushort b) {
    uint u = ((uint)b) << 16;
    return as_float(u);
}

__kernel void int4_gemv_m1(
    __global float * restrict y,              // [M]
    __global const uchar * restrict w_packed, // [M, K / 2]
    __global const ushort * restrict w_scale, // [M, K / 128]
    __global const float * restrict x,        // [K]
    int M,
    int K
) {
    int gid = get_global_id(0);
    int m0 = gid * 2;
    int m1 = m0 + 1;
    if (m0 >= M) return;

    int num_groups = K / GROUP_SIZE;
    __global const uchar *row0_w = w_packed + (size_t)m0 * (K / 2);
    __global const ushort *row0_s = w_scale + (size_t)m0 * num_groups;

    bool has_m1 = (m1 < M);
    __global const uchar *row1_w = has_m1 ? (w_packed + (size_t)m1 * (K / 2)) : row0_w;
    __global const ushort *row1_s = has_m1 ? (w_scale + (size_t)m1 * num_groups) : row0_s;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        float scale0 = bf16_to_fp32(row0_s[g]);
        float scale1 = has_m1 ? bf16_to_fp32(row1_s[g]) : 0.0f;

        __global const uchar16 *w0_vec16 = (__global const uchar16 *)(row0_w + g * (GROUP_SIZE / 2));
        __global const uchar16 *w1_vec16 = (__global const uchar16 *)(row1_w + g * (GROUP_SIZE / 2));
        __global const float8 *x_vec8 = (__global const float8 *)(x + g * GROUP_SIZE);

        float acc0_0 = 0.0f, acc0_1 = 0.0f;
        float acc1_0 = 0.0f, acc1_1 = 0.0f;

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uchar16 wb0 = w0_vec16[v];
            uchar16 wb1 = w1_vec16[v];

            float8 x0 = x_vec8[v * 4 + 0];
            float8 x1 = x_vec8[v * 4 + 1];
            float8 x2 = x_vec8[v * 4 + 2];
            float8 x3 = x_vec8[v * 4 + 3];

            // Byte 0..3 for x0
            int a0  = (int)((char)(wb0.s0 << 4)) >> 4;
            int a1  = (int)((char)wb0.s0) >> 4;
            int a2  = (int)((char)(wb0.s1 << 4)) >> 4;
            int a3  = (int)((char)wb0.s1) >> 4;
            int a4  = (int)((char)(wb0.s2 << 4)) >> 4;
            int a5  = (int)((char)wb0.s2) >> 4;
            int a6  = (int)((char)(wb0.s3 << 4)) >> 4;
            int a7  = (int)((char)wb0.s3) >> 4;

            acc0_0 += (float)a0 * x0.s0 + (float)a1 * x0.s1
                    + (float)a2 * x0.s2 + (float)a3 * x0.s3
                    + (float)a4 * x0.s4 + (float)a5 * x0.s5
                    + (float)a6 * x0.s6 + (float)a7 * x0.s7;

            int b0  = (int)((char)(wb1.s0 << 4)) >> 4;
            int b1  = (int)((char)wb1.s0) >> 4;
            int b2  = (int)((char)(wb1.s1 << 4)) >> 4;
            int b3  = (int)((char)wb1.s1) >> 4;
            int b4  = (int)((char)(wb1.s2 << 4)) >> 4;
            int b5  = (int)((char)wb1.s2) >> 4;
            int b6  = (int)((char)(wb1.s3 << 4)) >> 4;
            int b7  = (int)((char)wb1.s3) >> 4;

            acc1_0 += (float)b0 * x0.s0 + (float)b1 * x0.s1
                    + (float)b2 * x0.s2 + (float)b3 * x0.s3
                    + (float)b4 * x0.s4 + (float)b5 * x0.s5
                    + (float)b6 * x0.s6 + (float)b7 * x0.s7;

            // Byte 4..7 for x1
            int a8  = (int)((char)(wb0.s4 << 4)) >> 4;
            int a9  = (int)((char)wb0.s4) >> 4;
            int a10 = (int)((char)(wb0.s5 << 4)) >> 4;
            int a11 = (int)((char)wb0.s5) >> 4;
            int a12 = (int)((char)(wb0.s6 << 4)) >> 4;
            int a13 = (int)((char)wb0.s6) >> 4;
            int a14 = (int)((char)(wb0.s7 << 4)) >> 4;
            int a15 = (int)((char)wb0.s7) >> 4;

            acc0_1 += (float)a8 * x1.s0 + (float)a9 * x1.s1
                    + (float)a10 * x1.s2 + (float)a11 * x1.s3
                    + (float)a12 * x1.s4 + (float)a13 * x1.s5
                    + (float)a14 * x1.s6 + (float)a15 * x1.s7;

            int b8  = (int)((char)(wb1.s4 << 4)) >> 4;
            int b9  = (int)((char)wb1.s4) >> 4;
            int b10 = (int)((char)(wb1.s5 << 4)) >> 4;
            int b11 = (int)((char)wb1.s5) >> 4;
            int b12 = (int)((char)(wb1.s6 << 4)) >> 4;
            int b13 = (int)((char)wb1.s6) >> 4;
            int b14 = (int)((char)(wb1.s7 << 4)) >> 4;
            int b15 = (int)((char)wb1.s7) >> 4;

            acc1_1 += (float)b8 * x1.s0 + (float)b9 * x1.s1
                    + (float)b10 * x1.s2 + (float)b11 * x1.s3
                    + (float)b12 * x1.s4 + (float)b13 * x1.s5
                    + (float)b14 * x1.s6 + (float)b15 * x1.s7;

            // Byte 8..11 for x2
            int a16 = (int)((char)(wb0.s8 << 4)) >> 4;
            int a17 = (int)((char)wb0.s8) >> 4;
            int a18 = (int)((char)(wb0.s9 << 4)) >> 4;
            int a19 = (int)((char)wb0.s9) >> 4;
            int a20 = (int)((char)(wb0.sa << 4)) >> 4;
            int a21 = (int)((char)wb0.sa) >> 4;
            int a22 = (int)((char)(wb0.sb << 4)) >> 4;
            int a23 = (int)((char)wb0.sb) >> 4;

            acc0_0 += (float)a16 * x2.s0 + (float)a17 * x2.s1
                    + (float)a18 * x2.s2 + (float)a19 * x2.s3
                    + (float)a20 * x2.s4 + (float)a21 * x2.s5
                    + (float)a22 * x2.s6 + (float)a23 * x2.s7;

            int b16 = (int)((char)(wb1.s8 << 4)) >> 4;
            int b17 = (int)((char)wb1.s8) >> 4;
            int b18 = (int)((char)(wb1.s9 << 4)) >> 4;
            int b19 = (int)((char)wb1.s9) >> 4;
            int b20 = (int)((char)(wb1.sa << 4)) >> 4;
            int b21 = (int)((char)wb1.sa) >> 4;
            int b22 = (int)((char)(wb1.sb << 4)) >> 4;
            int b23 = (int)((char)wb1.sb) >> 4;

            acc1_0 += (float)b16 * x2.s0 + (float)b17 * x2.s1
                    + (float)b18 * x2.s2 + (float)b19 * x2.s3
                    + (float)b20 * x2.s4 + (float)b21 * x2.s5
                    + (float)b22 * x2.s6 + (float)b23 * x2.s7;

            // Byte 12..15 for x3
            int a24 = (int)((char)(wb0.sc << 4)) >> 4;
            int a25 = (int)((char)wb0.sc) >> 4;
            int a26 = (int)((char)(wb0.sd << 4)) >> 4;
            int a27 = (int)((char)wb0.sd) >> 4;
            int a28 = (int)((char)(wb0.se << 4)) >> 4;
            int a29 = (int)((char)wb0.se) >> 4;
            int a30 = (int)((char)(wb0.sf << 4)) >> 4;
            int a31 = (int)((char)wb0.sf) >> 4;

            acc0_1 += (float)a24 * x3.s0 + (float)a25 * x3.s1
                    + (float)a26 * x3.s2 + (float)a27 * x3.s3
                    + (float)a28 * x3.s4 + (float)a29 * x3.s5
                    + (float)a30 * x3.s6 + (float)a31 * x3.s7;

            int b24 = (int)((char)(wb1.sc << 4)) >> 4;
            int b25 = (int)((char)wb1.sc) >> 4;
            int b26 = (int)((char)(wb1.sd << 4)) >> 4;
            int b27 = (int)((char)wb1.sd) >> 4;
            int b28 = (int)((char)(wb1.se << 4)) >> 4;
            int b29 = (int)((char)wb1.se) >> 4;
            int b30 = (int)((char)(wb1.sf << 4)) >> 4;
            int b31 = (int)((char)wb1.sf) >> 4;

            acc1_1 += (float)b24 * x3.s0 + (float)b25 * x3.s1
                    + (float)b26 * x3.s2 + (float)b27 * x3.s3
                    + (float)b28 * x3.s4 + (float)b29 * x3.s5
                    + (float)b30 * x3.s6 + (float)b31 * x3.s7;
        }

        sum0 += (acc0_0 + acc0_1) * scale0;
        if (has_m1) sum1 += (acc1_0 + acc1_1) * scale1;
    }

    y[m0] = sum0;
    if (has_m1) y[m1] = sum1;
}
