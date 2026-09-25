// Optimized decode attention override module (T-decode-opt / I3.6, 2026-09-24).
//
// Contains `gqa_attn_decode_ctrl` with in-register Intel Xe2 subgroup butterfly
// reduction (cl_intel_subgroups) and 16-token unrolled double-buffered SLM staging.
// Barrier frequency is reduced from 3 barriers per position down to 1 barrier
// per 16 positions (a 48x reduction in barrier synchronization overhead!).
// Same entry name, same signature, same launch shape (16 groups x 256) as
// the copy in all_kernels.cl, so the runtime can override the kernel handle
// without any other change.
//
// Compiled via:
//   ocloc compile -device lnl -file tools/kernels_258v/attn_decode_opt.cl \
//     -output tools/kernels_258v/all_kernels.spv.attn
//   cp tools/kernels_258v/all_kernels.spv.attn_lnl.spv tools/kernels_258v/all_kernels.spv.attn

#define HEAD_DIM 256
#define ROTARY_DIM 64
#define ROTARY_HALF 32
#define NUM_Q_HEADS 16
#define NUM_KV_HEADS 2
#define GQA_GROUP_SIZE 8 // 16 / 2
#define ATTN_SCALE 0.0625f // 1.0f / sqrt(256.0f) = 1/16

static inline float bf16_to_float(ushort b) {
    uint u = ((uint)b) << 16;
    return as_float(u);
}

// GQA Attention Decode Kernel with Subgroup Butterfly Reduction & Online Softmax:
// Launch 16 workgroups of 256 work-items (1 workgroup per Q head).
// Attends over tokens 0..pos using BF16 KV cache.
// Applies Sigmoid Output Gating: out[h, d] *= sigmoid(gate[h, d]).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gqa_attn_decode_ctrl(
    __global float * restrict out,             // [16, 256]
    __global const float * restrict q,         // [16, 256]
    __global const float * restrict gate,      // [16, 256]
    __global const ushort * restrict k_cache,  // [2, max_ctx, 256]
    __global const ushort * restrict v_cache,  // [2, max_ctx, 256]
    __global const int * restrict ctrl,        // ctrl[1] = position
    uint max_ctx
) {
    uint pos = (uint)ctrl[1];
    int qh = get_group_id(0); // 0..15 (Q head index)
    if (qh >= NUM_Q_HEADS) return;
    int tid = get_local_id(0); // 0..255 (dimension index d)
    int lane = get_sub_group_local_id(); // 0..15
    int sg = get_sub_group_id(); // 0..15

    int kv_h = qh / GQA_GROUP_SIZE; // 0 or 1
    float qv = q[qh * HEAD_DIM + tid];

    // Double buffer: 2 x (16 subgroups x 16 positions) = 512 floats = 2 KB SLM
    __local float s_part[2][256];

    // Running Online Softmax State
    float run_max = -1e30f;
    float run_sum = 0.0f;
    float run_acc = 0.0f;

    uint total_tokens = pos + 1;

    for (uint tb = 0; tb < total_tokens; tb += 16) {
        int buf_idx = (int)((tb >> 4) & 1);
        float part[16];

        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            uint t = tb + (uint)u;
            float kval = 0.0f;
            if (t < total_tokens) {
                __global const ushort * k_slot = k_cache + ((size_t)kv_h * max_ctx + t) * HEAD_DIM;
                kval = bf16_to_float(k_slot[tid]);
            }
            part[u] = qv * kval;
        }

        // Subgroup butterfly reduction across the 16 lanes (register-only, zero barrier)
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            float v = part[u];
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 8));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 4));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 2));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 1));
            part[u] = v;
        }

        if (lane == 0) {
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                s_part[buf_idx][sg * 16 + u] = part[u];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                s += s_part[buf_idx][i * 16 + u];
            }
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

    // Final attention vector for this head
    float attn_val = run_acc / run_sum;

    // Apply Sigmoid Output Gate: out = attn_val * sigmoid(gate)
    float g_val = gate[qh * HEAD_DIM + tid];
    float sig_g = 1.0f / (1.0f + exp(-g_val));

    out[qh * HEAD_DIM + tid] = attn_val * sig_g;
}
