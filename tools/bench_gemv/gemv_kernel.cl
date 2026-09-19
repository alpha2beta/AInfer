// INT4 Symmetric Group-128 GEMV Kernel for Intel Arc 140V (Xe2)
// Computes y = W * x, where W is INT4 group-128 with BF16 scales, x is FP32.
// Matrix dimensions: M rows, K columns (K is multiple of 128).

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
    int m = get_global_id(0);
    if (m >= M) return;

    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w = w_packed + (size_t)m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)m * num_groups;

    float total_sum = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        float scale = bf16_to_fp32(row_s[g]);
        __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2); // 64 bytes
        __global const float *grp_x = x + g * GROUP_SIZE;           // 128 floats

        float grp_acc = 0.0f;

        // Process 4 bytes (8 weights) at a time
        __global const uchar4 *w_vec = (__global const uchar4 *)grp_w;
        __global const float8 *x_vec = (__global const float8 *)grp_x;

        for (int i = 0; i < 16; ++i) {
            uchar4 wb = w_vec[i];
            float8 xv = x_vec[i];

            // Unpack 4 bytes into 8 signed integers in [-8, 7]
            int n0 = (wb.x & 0x0F); if (n0 >= 8) n0 -= 16;
            int n1 = (wb.x >> 4);   if (n1 >= 8) n1 -= 16;
            int n2 = (wb.y & 0x0F); if (n2 >= 8) n2 -= 16;
            int n3 = (wb.y >> 4);   if (n3 >= 8) n3 -= 16;
            int n4 = (wb.z & 0x0F); if (n4 >= 8) n4 -= 16;
            int n5 = (wb.z >> 4);   if (n5 >= 8) n5 -= 16;
            int n6 = (wb.w & 0x0F); if (n6 >= 8) n6 -= 16;
            int n7 = (wb.w >> 4);   if (n7 >= 8) n7 -= 16;

            grp_acc += (float)n0 * xv.s0 + (float)n1 * xv.s1
                     + (float)n2 * xv.s2 + (float)n3 * xv.s3
                     + (float)n4 * xv.s4 + (float)n5 * xv.s5
                     + (float)n6 * xv.s6 + (float)n7 * xv.s7;
        }

        total_sum += grp_acc * scale;
    }

    y[m] = total_sum;
}
