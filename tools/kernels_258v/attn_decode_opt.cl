// Optimized decode attention override module (T-decode-opt, 2026-09-22).
//
// Contains ONLY `gqa_attn_decode_ctrl` with 3 barriers per position instead
// of 10 (8-thread 4-chain dot reduction + 1-float SLM broadcast).
// Same entry name, same signature, same launch shape (16 groups x 256) as
// the copy in all_kernels.cl, so the runtime can override the kernel handle
// without any other change.
//
// Rationale: all_kernels.cl also needs Intel-ESIMD extensions
// (cl_intel_subgroups, joint-matrix) that upstream clang cannot compile, and
// no Intel compiler is installed on this box — so this upstream-clang-clean
// subset is built separately:
//
//   clang -target spirv64 -x cl -cl-std=CL2.0 -O2 \
//     -c tools/kernels_258v/attn_decode_opt.cl \
//     -o tools/kernels_258v/all_kernels.spv.attn
//
// The runtime loads `<spv>.attn` as an optional override module (warns and
// keeps the bundled kernel if absent). all_kernels.cl / attention.cl carry
// the same rewrite as source-of-truth for the next Intel-toolchain rebuild.

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

// GQA Attention Decode Kernel with Online Softmax:
// Launch 16 workgroups of 256 work-items (1 workgroup per Q head).
// Attends over tokens 0..pos using BF16 KV cache.
// Applies Sigmoid Output Gating: out[h, d] *= sigmoid(gate[h, d]).
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

    int kv_h = qh / GQA_GROUP_SIZE; // 0 or 1

    __local float s_red[HEAD_DIM];
    __local float s_score[1];

    float qv = q[qh * HEAD_DIM + tid];

    // Running Online Softmax State
    float run_max = -1e30f;
    float run_sum = 0.0f;
    float run_acc = 0.0f;

    uint total_tokens = pos + 1;

    for (uint t = 0; t < total_tokens; ++t) {
        __global const ushort * k_slot = k_cache + (kv_h * max_ctx + t) * HEAD_DIM;
        float k_val = bf16_to_float(k_slot[tid]);

        // Dot product Q[tid] * K[tid]
        s_red[tid] = qv * k_val;
        barrier(CLK_LOCAL_MEM_FENCE);

        // 8 threads x 32 partials with 4 independent accumulator chains
        // (no long dependency chain), thread 0 combines 8 and broadcasts.
        // v2: v1's single 256-deep serial chain was SLOWER than the tree.
        if (tid < 8) {
            __local float *base = s_red + tid * 32;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                a0 += base[i * 4 + 0];
                a1 += base[i * 4 + 1];
                a2 += base[i * 4 + 2];
                a3 += base[i * 4 + 3];
            }
            s_red[tid] = (a0 + a1) + (a2 + a3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid == 0) {
            float s = s_red[0] + s_red[1] + s_red[2] + s_red[3]
                    + s_red[4] + s_red[5] + s_red[6] + s_red[7];
            s_score[0] = s * ATTN_SCALE;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float score = s_score[0];

        // Load V token
        __global const ushort * v_slot = v_cache + (kv_h * max_ctx + t) * HEAD_DIM;
        float v_val = bf16_to_float(v_slot[tid]);

        // Online softmax update
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

    // Final attention vector for this head
    float attn_val = run_acc / run_sum;

    // Apply Sigmoid Output Gate: out = attn_val * sigmoid(gate)
    float g_val = gate[qh * HEAD_DIM + tid];
    float sig_g = 1.0f / (1.0f + exp(-g_val));

    out[qh * HEAD_DIM + tid] = attn_val * sig_g;
}
