// test_gemv_m2.cl — Experimental DPAS Dual-Token GEMV kernels for Intel Arc 140V (Xe2)

#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define GROUP_SIZE 128

static inline float bf16_to_fp32(ushort b) {
    uint u = ((uint)b) << 16;
    return as_float(u);
}

// -----------------------------------------------------------------------------
// Baseline (Current trunk): 1 thread per row, scalar accumulation
// -----------------------------------------------------------------------------
__kernel void int4_gemv_m2_baseline(
    __global float * restrict y,
    __global const uchar * restrict w_packed,
    __global const ushort * restrict w_scale,
    __global const float * restrict x,
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

    y[m] = total_sum0;
    y[(size_t)M + m] = total_sum1;
}

// -----------------------------------------------------------------------------
// Variant 1: DPAS k16 (16 M-rows per subgroup, 16 K-depth per step)
// -----------------------------------------------------------------------------
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void int4_gemv_m2_dpas_k16(
    __global float * restrict y,
    __global const uchar * restrict w_packed,
    __global const ushort * restrict w_scale,
    __global const float * restrict x,
    int M,
    int K
) {
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_m = get_group_id(0);
    int lid = get_sub_group_local_id(); // 0..15

    int m_tile_idx = grp_m * num_sg + sg_id;
    int m_base = m_tile_idx * 16;
    int m = m_base + lid;
    int safe_m = (m < M) ? m : 0;

    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w = w_packed + (size_t)safe_m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)safe_m * num_groups;

    float8 acc = (float8)(0.0f);
    int total_steps = K / 16;

    for (int s = 0; s < total_steps; ++s) {
        int k = s * 16;
        int g = s / 8;

        float x0 = x[k + lid];
        float x1 = x[(size_t)K + k + lid];

        short8 a_mat = (short8)(0);
        ((short *)&a_mat)[0] = as_short((half)x0);
        ((short *)&a_mat)[1] = as_short((half)x1);

        float s_val = bf16_to_fp32(row_s[g]);
        half s_half = (half)s_val;

        __global const uchar *w_ptr = row_w + (size_t)s * 8;
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

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_mat, b_mat, acc);
    }

    if (m < M) {
        y[m] = acc.s0;
        y[(size_t)M + m] = acc.s1;
    }
}

// -----------------------------------------------------------------------------
// Variant 2: DPAS k32 (16 M-rows per subgroup, 32 K-depth per step, 16B weight load)
// -----------------------------------------------------------------------------
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void int4_gemv_m2_dpas_k32(
    __global float * restrict y,
    __global const uchar * restrict w_packed,
    __global const ushort * restrict w_scale,
    __global const float * restrict x,
    int M,
    int K
) {
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_m = get_group_id(0);
    int lid = get_sub_group_local_id(); // 0..15

    int m_tile_idx = grp_m * num_sg + sg_id;
    int m_base = m_tile_idx * 16;
    int m = m_base + lid;
    int safe_m = (m < M) ? m : 0;

    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w = w_packed + (size_t)safe_m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)safe_m * num_groups;

    float8 acc = (float8)(0.0f);
    int total_pairs = K / 32;

    for (int pr = 0; pr < total_pairs; ++pr) {
        int k = pr * 32;
        int g = pr / 4; // 32 * 4 = 128 = GROUP_SIZE

        float x0_0 = x[k + lid];
        float x0_1 = x[k + 16 + lid];
        float x1_0 = x[(size_t)K + k + lid];
        float x1_1 = x[(size_t)K + k + 16 + lid];

        short8 a_mat0 = (short8)(0);
        ((short *)&a_mat0)[0] = as_short((half)x0_0);
        ((short *)&a_mat0)[1] = as_short((half)x1_0);

        short8 a_mat1 = (short8)(0);
        ((short *)&a_mat1)[0] = as_short((half)x0_1);
        ((short *)&a_mat1)[1] = as_short((half)x1_1);

        float s_val = bf16_to_fp32(row_s[g]);
        half s_half = (half)s_val;

        __global const uchar *w_ptr = row_w + (size_t)pr * 16;
        uchar16 raw_w = *((__global const uchar16 *)w_ptr);

        half w_deq0[16];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uchar byte_val = ((uchar *)&raw_w)[i];
            int n0 = (int)((char)(byte_val << 4)) >> 4;
            int n1 = (int)((char)byte_val) >> 4;
            w_deq0[2 * i]     = (half)((float)n0) * s_half;
            w_deq0[2 * i + 1] = (half)((float)n1) * s_half;
        }

        half w_deq1[16];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uchar byte_val = ((uchar *)&raw_w)[8 + i];
            int n0 = (int)((char)(byte_val << 4)) >> 4;
            int n1 = (int)((char)byte_val) >> 4;
            w_deq1[2 * i]     = (half)((float)n0) * s_half;
            w_deq1[2 * i + 1] = (half)((float)n1) * s_half;
        }

        int8 b_mat0, b_mat1;
        __builtin_memcpy(&b_mat0, w_deq0, 32);
        __builtin_memcpy(&b_mat1, w_deq1, 32);

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_mat0, b_mat0, acc);
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_mat1, b_mat1, acc);
    }

    if (m < M) {
        y[m] = acc.s0;
        y[(size_t)M + m] = acc.s1;
    }
}

// -----------------------------------------------------------------------------
// Variant 3: DPAS m32k32 (32 M-rows per subgroup, 32 K-depth per step)
// Reuses activation tiles a_mat0, a_mat1 across 2 rows of M (m0, m1)
// -----------------------------------------------------------------------------
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void int4_gemv_m2_dpas_m32k32(
    __global float * restrict y,
    __global const uchar * restrict w_packed,
    __global const ushort * restrict w_scale,
    __global const float * restrict x,
    int M,
    int K
) {
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_m = get_group_id(0);
    int lid = get_sub_group_local_id(); // 0..15

    int m_tile_idx = grp_m * num_sg + sg_id;
    int m_base = m_tile_idx * 32;
    int m0 = m_base + lid;
    int m1 = m_base + 16 + lid;

    if (m0 >= M) return;

    int safe_m0 = m0;
    int safe_m1 = (m1 < M) ? m1 : 0;

    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w0 = w_packed + (size_t)safe_m0 * (K / 2);
    __global const ushort *row_s0 = w_scale + (size_t)safe_m0 * num_groups;

    __global const uchar *row_w1 = w_packed + (size_t)safe_m1 * (K / 2);
    __global const ushort *row_s1 = w_scale + (size_t)safe_m1 * num_groups;

    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    int total_pairs = K / 32;

    for (int pr = 0; pr < total_pairs; ++pr) {
        int k = pr * 32;
        int g = pr / 4; // 32 * 4 = 128 = GROUP_SIZE

        // Load activations ONCE for both rows
        float x0_0 = x[k + lid];
        float x0_1 = x[k + 16 + lid];
        float x1_0 = x[(size_t)K + k + lid];
        float x1_1 = x[(size_t)K + k + 16 + lid];

        short8 a_mat0 = (short8)(0);
        ((short *)&a_mat0)[0] = as_short((half)x0_0);
        ((short *)&a_mat0)[1] = as_short((half)x1_0);

        short8 a_mat1 = (short8)(0);
        ((short *)&a_mat1)[0] = as_short((half)x0_1);
        ((short *)&a_mat1)[1] = as_short((half)x1_1);

        // Row 0 weights
        float s_val0 = bf16_to_fp32(row_s0[g]);
        half s_half0 = (half)s_val0;
        uchar16 raw_w0 = *((__global const uchar16 *)(row_w0 + (size_t)pr * 16));

        half w_deq0_0[16], w_deq0_1[16];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uchar b0 = ((uchar *)&raw_w0)[i];
            uchar b1 = ((uchar *)&raw_w0)[8 + i];
            w_deq0_0[2 * i]     = (half)((float)((int)((char)(b0 << 4)) >> 4)) * s_half0;
            w_deq0_0[2 * i + 1] = (half)((float)((int)((char)b0) >> 4)) * s_half0;
            w_deq0_1[2 * i]     = (half)((float)((int)((char)(b1 << 4)) >> 4)) * s_half0;
            w_deq0_1[2 * i + 1] = (half)((float)((int)((char)b1) >> 4)) * s_half0;
        }

        int8 b_mat0_0, b_mat0_1;
        __builtin_memcpy(&b_mat0_0, w_deq0_0, 32);
        __builtin_memcpy(&b_mat0_1, w_deq0_1, 32);

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat0, b_mat0_0, acc0);
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat1, b_mat0_1, acc0);

        // Row 1 weights (if m1 < M)
        if (m1 < M) {
            float s_val1 = bf16_to_fp32(row_s1[g]);
            half s_half1 = (half)s_val1;
            uchar16 raw_w1 = *((__global const uchar16 *)(row_w1 + (size_t)pr * 16));

            half w_deq1_0[16], w_deq1_1[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uchar b0 = ((uchar *)&raw_w1)[i];
                uchar b1 = ((uchar *)&raw_w1)[8 + i];
                w_deq1_0[2 * i]     = (half)((float)((int)((char)(b0 << 4)) >> 4)) * s_half1;
                w_deq1_0[2 * i + 1] = (half)((float)((int)((char)b0) >> 4)) * s_half1;
                w_deq1_1[2 * i]     = (half)((float)((int)((char)(b1 << 4)) >> 4)) * s_half1;
                w_deq1_1[2 * i + 1] = (half)((float)((int)((char)b1) >> 4)) * s_half1;
            }

            int8 b_mat1_0, b_mat1_1;
            __builtin_memcpy(&b_mat1_0, w_deq1_0, 32);
            __builtin_memcpy(&b_mat1_1, w_deq1_1, 32);

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat0, b_mat1_0, acc1);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_mat1, b_mat1_1, acc1);
        }
    }

    if (m0 < M) {
        y[m0] = acc0.s0;
        y[(size_t)M + m0] = acc0.s1;
    }
    if (m1 < M) {
        y[m1] = acc1.s0;
        y[(size_t)M + m1] = acc1.s1;
    }
}

// -----------------------------------------------------------------------------
// Variant 4: DPAS Opt (native M=2 systolic mad, group-level FP32 scaling, 32 K/step)
// -----------------------------------------------------------------------------
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void int4_gemv_m2_dpas_opt(
    __global float * restrict y,
    __global const uchar * restrict w_packed,
    __global const ushort * restrict w_scale,
    __global const float * restrict x,
    int M,
    int K
) {
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_m = get_group_id(0);
    int lid = get_sub_group_local_id(); // 0..15

    int m_tile_idx = grp_m * num_sg + sg_id;
    int m_base = m_tile_idx * 16;
    int m = m_base + lid;
    int safe_m = (m < M) ? m : 0;

    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w = w_packed + (size_t)safe_m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)safe_m * num_groups;

    float2 total_acc = (float2)(0.0f);

    for (int g = 0; g < num_groups; ++g) {
        float2 grp_acc = (float2)(0.0f);
        __global const uchar16 *grp_w = (__global const uchar16 *)(row_w + g * (GROUP_SIZE / 2));
        int k_grp = g * GROUP_SIZE;

        #pragma unroll
        for (int pr = 0; pr < 4; ++pr) {
            int k = k_grp + pr * 32;

            float x0_0 = x[k + lid];
            float x0_1 = x[k + 16 + lid];
            float x1_0 = x[(size_t)K + k + lid];
            float x1_1 = x[(size_t)K + k + 16 + lid];

            short2 a0;
            a0.s0 = as_short((half)x0_0);
            a0.s1 = as_short((half)x1_0);

            short2 a1;
            a1.s0 = as_short((half)x0_1);
            a1.s1 = as_short((half)x1_1);

            uchar16 raw_w = grp_w[pr];

            half w_deq0[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uchar b = ((uchar *)&raw_w)[i];
                int n0 = (int)((char)(b << 4)) >> 4;
                int n1 = (int)((char)b) >> 4;
                w_deq0[2 * i]     = (half)n0;
                w_deq0[2 * i + 1] = (half)n1;
            }

            half w_deq1[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uchar b = ((uchar *)&raw_w)[8 + i];
                int n0 = (int)((char)(b << 4)) >> 4;
                int n1 = (int)((char)b) >> 4;
                w_deq1[2 * i]     = (half)n0;
                w_deq1[2 * i + 1] = (half)n1;
            }

            int8 b0, b1;
            __builtin_memcpy(&b0, w_deq0, 32);
            __builtin_memcpy(&b1, w_deq1, 32);

            grp_acc = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, grp_acc);
            grp_acc = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, grp_acc);
        }

        float scale = bf16_to_fp32(row_s[g]);
        total_acc += grp_acc * scale;
    }

    if (m < M) {
        y[m] = total_acc.s0;
        y[(size_t)M + m] = total_acc.s1;
    }
}

// -----------------------------------------------------------------------------
// Variant 5: DPAS SLM (SLM staged activations, native M=2 systolic mad, 32 K/step)
// -----------------------------------------------------------------------------
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void int4_gemv_m2_dpas_slm(
    __global float * restrict y,
    __global const uchar * restrict w_packed,
    __global const ushort * restrict w_scale,
    __global const float * restrict x,
    int M,
    int K
) {
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_m = get_group_id(0);
    int lid = get_sub_group_local_id(); // 0..15
    int tid = get_local_id(0);          // 0..127

    int m_tile_idx = grp_m * num_sg + sg_id;
    int m_base = m_tile_idx * 16;
    int m = m_base + lid;
    int safe_m = (m < M) ? m : 0;

    int num_groups = K / GROUP_SIZE;
    __global const uchar *row_w = w_packed + (size_t)safe_m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)safe_m * num_groups;

    __local half s_x0[2][GROUP_SIZE];
    __local half s_x1[2][GROUP_SIZE];

    float2 total_acc = (float2)(0.0f);

    // Prologue: prefetch group 0 into s_x[0]
    int k0 = tid;
    s_x0[0][tid] = (half)x[k0];
    s_x1[0][tid] = (half)x[(size_t)K + k0];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int g = 0; g < num_groups; ++g) {
        int cur_buf = g & 1;
        int next_buf = (g + 1) & 1;

        // Prefetch next group into idle buffer
        if (g + 1 < num_groups) {
            int k_next = (g + 1) * GROUP_SIZE + tid;
            s_x0[next_buf][tid] = (half)x[k_next];
            s_x1[next_buf][tid] = (half)x[(size_t)K + k_next];
        }

        float2 grp_acc = (float2)(0.0f);
        __global const uchar16 *grp_w = (__global const uchar16 *)(row_w + g * (GROUP_SIZE / 2));

        #pragma unroll
        for (int pr = 0; pr < 4; ++pr) {
            int k_sub = pr * 32 + lid;

            short2 a0;
            a0.s0 = as_short(s_x0[cur_buf][k_sub]);
            a0.s1 = as_short(s_x1[cur_buf][k_sub]);

            short2 a1;
            a1.s0 = as_short(s_x0[cur_buf][k_sub + 16]);
            a1.s1 = as_short(s_x1[cur_buf][k_sub + 16]);

            uchar16 raw_w = grp_w[pr];

            half w_deq0[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uchar b = ((uchar *)&raw_w)[i];
                int n0 = (int)((char)(b << 4)) >> 4;
                int n1 = (int)((char)b) >> 4;
                w_deq0[2 * i]     = (half)n0;
                w_deq0[2 * i + 1] = (half)n1;
            }

            half w_deq1[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uchar b = ((uchar *)&raw_w)[8 + i];
                int n0 = (int)((char)(b << 4)) >> 4;
                int n1 = (int)((char)b) >> 4;
                w_deq1[2 * i]     = (half)n0;
                w_deq1[2 * i + 1] = (half)n1;
            }

            int8 b0, b1;
            __builtin_memcpy(&b0, w_deq0, 32);
            __builtin_memcpy(&b1, w_deq1, 32);

            grp_acc = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, grp_acc);
            grp_acc = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, grp_acc);
        }

        float scale = bf16_to_fp32(row_s[g]);
        total_acc += grp_acc * scale;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (m < M) {
        y[m] = total_acc.s0;
        y[(size_t)M + m] = total_acc.s1;
    }
}

// -----------------------------------------------------------------------------
// Variant 6: LM Head DPAS Argmax1 (M=2 DPAS systolic evaluation + workgroup reduction)
// -----------------------------------------------------------------------------
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void int4_gemv_m2_lm_head_dpas_argmax1(
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
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_m = get_group_id(0);
    int lid = get_sub_group_local_id(); // 0..15
    int tid = get_local_id(0);          // 0..255

    int m_tile_idx = grp_m * num_sg + sg_id;
    int m_base = m_tile_idx * 16;
    int m = m_base + lid;

    float total_sum0 = -1e30f;
    float total_sum1 = -1e30f;
    uint my_idx = (m < M) ? (uint)m : 0xFFFFFFFF;

    if (m < M) {
        int num_groups = K / GROUP_SIZE;
        __global const uchar *row_w = w_packed + (size_t)m * (K / 2);
        __global const ushort *row_s = w_scale + (size_t)m * num_groups;

        float2 total_acc = (float2)(0.0f);

        for (int g = 0; g < num_groups; ++g) {
            float2 grp_acc = (float2)(0.0f);
            __global const uchar16 *grp_w = (__global const uchar16 *)(row_w + g * (GROUP_SIZE / 2));
            int k_grp = g * GROUP_SIZE;

            #pragma unroll
            for (int pr = 0; pr < 4; ++pr) {
                int k = k_grp + pr * 32;

                float x0_0 = x[k + lid];
                float x0_1 = x[k + 16 + lid];
                float x1_0 = x[(size_t)K + k + lid];
                float x1_1 = x[(size_t)K + k + 16 + lid];

                short2 a0;
                a0.s0 = as_short((half)x0_0);
                a0.s1 = as_short((half)x1_0);

                short2 a1;
                a1.s0 = as_short((half)x0_1);
                a1.s1 = as_short((half)x1_1);

                uchar16 raw_w = grp_w[pr];

                half w_deq0[16];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uchar b = ((uchar *)&raw_w)[i];
                    int n0 = (int)((char)(b << 4)) >> 4;
                    int n1 = (int)((char)b) >> 4;
                    w_deq0[2 * i]     = (half)n0;
                    w_deq0[2 * i + 1] = (half)n1;
                }

                half w_deq1[16];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uchar b = ((uchar *)&raw_w)[8 + i];
                    int n0 = (int)((char)(b << 4)) >> 4;
                    int n1 = (int)((char)b) >> 4;
                    w_deq1[2 * i]     = (half)n0;
                    w_deq1[2 * i + 1] = (half)n1;
                }

                int8 b0, b1;
                __builtin_memcpy(&b0, w_deq0, 32);
                __builtin_memcpy(&b1, w_deq1, 32);

                grp_acc = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, grp_acc);
                grp_acc = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, grp_acc);
            }

            float scale = bf16_to_fp32(row_s[g]);
            total_acc += grp_acc * scale;
        }

        total_sum0 = total_acc.s0;
        total_sum1 = total_acc.s1;

        if (y != NULL) {
            y[m] = total_sum0;
            y[(size_t)M + m] = total_sum1;
        }
    }

    __local float s_val0[256];
    __local uint s_idx0[256];
    __local float s_val1[256];
    __local uint s_idx1[256];

    s_val0[tid] = total_sum0;
    s_idx0[tid] = my_idx;
    s_val1[tid] = total_sum1;
    s_idx1[tid] = my_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val0[tid + s] > s_val0[tid] || (s_val0[tid + s] == s_val0[tid] && s_idx0[tid + s] < s_idx0[tid])) {
                s_val0[tid] = s_val0[tid + s];
                s_idx0[tid] = s_idx0[tid + s];
            }
            if (s_val1[tid + s] > s_val1[tid] || (s_val1[tid + s] == s_val1[tid] && s_idx1[tid + s] < s_idx1[tid])) {
                s_val1[tid] = s_val1[tid + s];
                s_idx1[tid] = s_idx1[tid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        stage1_vals_0[grp_m] = s_val0[0];
        stage1_idxs_0[grp_m] = s_idx0[0];
        stage1_vals_1[grp_m] = s_val1[0];
        stage1_idxs_1[grp_m] = s_idx1[0];
    }
}



