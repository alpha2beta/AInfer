// DPAS INT4 GEMM Prototype for Intel Arc 140V (Xe2)
// Uses intel_sub_group_f16_f16_matrix_mad_k16 for hardware systolic matrix acceleration.
// Computes Y[B, M] = X[B, K] * W[M, K]^T
// Tile size per subgroup (16 threads): M_tile = 16, B_tile = 8, K_step = 16.

#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define GROUP_SIZE 128

static inline float bf16_to_fp32(ushort b) {
    uint u = ((uint)b) << 16;
    return as_float(u);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpas_int4_gemm_m16_b8(
    __global float * restrict Y,              // [B, M] row-major
    __global const uchar * restrict w_packed, // [M, K / 2]
    __global const ushort * restrict w_scale, // [M, K / 128]
    __global const float * restrict X,        // [B, K] row-major
    int M,
    int K,
    int B
) {
    int sg_id = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int grp_id = get_group_id(0);
    int lid = get_sub_group_local_id(); // 0..15

    // Each subgroup computes an M_tile of 16 output channels
    int m_tile_idx = grp_id * num_sg + sg_id;
    int m_base = m_tile_idx * 16;
    if (m_base >= M) return;

    int m = m_base + lid; // Output row for this thread
    int num_groups = K / GROUP_SIZE;

    __global const uchar *row_w = w_packed + (size_t)m * (K / 2);
    __global const ushort *row_s = w_scale + (size_t)m * num_groups;

    // Loop over batch tiles of size 8
    for (int b_base = 0; b_base < B; b_base += 8) {
        int cur_B = (B - b_base < 8) ? (B - b_base) : 8;

        float8 acc = (float8)(0.0f);

        for (int g = 0; g < num_groups; ++g) {
            float s_val = bf16_to_fp32(row_s[g]);
            half s_half = (half)s_val;

            __global const uchar *grp_w = row_w + g * (GROUP_SIZE / 2);

            // In group-128, there are 8 steps of K=16
            for (int step = 0; step < 8; ++step) {
                int k_base = g * GROUP_SIZE + step * 16;

                // 1. Thread lid loads 16 weights (8 bytes) for its row m
                __global const uchar *w_ptr = grp_w + step * 8;
                uchar8 raw_w = *((__global const uchar8 *)w_ptr);

                // Unpack 16 nibbles to 16 halves and scale
                half w_deq[16];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uchar byte_val = ((uchar *)&raw_w)[i];
                    int n0 = (int)((char)(byte_val << 4)) >> 4;
                    int n1 = (int)((char)byte_val) >> 4;
                    w_deq[2 * i]     = (half)((float)n0) * s_half;
                    w_deq[2 * i + 1] = (half)((float)n1) * s_half;
                }

                // Pack 16 halves (32 bytes) into int8 b
                int8 b_mat;
                __builtin_memcpy(&b_mat, w_deq, 32);

                // 2. Load activation slice for each batch row
                // Thread lid loads X[b, k_base + lid] as half
                short8 a_mat = (short8)(0);
                #pragma unroll
                for (int bi = 0; bi < 8; ++bi) {
                    if (bi < cur_B) {
                        float x_val = X[(size_t)(b_base + bi) * K + k_base + lid];
                        half x_half = (half)x_val;
                        ((short *)&a_mat)[bi] = as_short(x_half);
                    }
                }

                // 3. Hardware DPAS operation (8x16x16: 8 rows of A x 16 cols of B with FP32 accumulation)
                acc = intel_sub_group_f16_f16_matrix_mad_k16(a_mat, b_mat, acc);
            }
        }

        // Store outputs: thread lid writes its column m = m_base + lid for each batch row
        if (m < M) {
            #pragma unroll
            for (int bi = 0; bi < 8; ++bi) {
                if (bi < cur_B) {
                    float val = ((float *)&acc)[bi];
                    Y[(size_t)(b_base + bi) * M + m] = val;
                }
            }
        }
    }
}
