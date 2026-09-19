// Full Attention (GQA) and RoPE Kernels for Qwen3.5-MoE on Intel Arc 140V (Xe2)
// N_q = 16, N_kv = 2, D = 256, Rotary_dim = 64, Theta = 1e7, BF16 KV cache.

#define HEAD_DIM 256
#define ROTARY_DIM 64
#define ROTARY_HALF 32
#define NUM_Q_HEADS 16
#define NUM_KV_HEADS 2
#define GQA_GROUP_SIZE 8 // 16 / 2
#define ROPE_THETA 10000000.0f // 1e7
#define ATTN_SCALE 0.0625f // 1.0f / sqrt(256.0f) = 1/16

// BF16 <-> FP32 conversion helpers
static inline ushort float_to_bf16(float x) {
    uint u = as_uint(x);
    // Round to nearest even
    return (ushort)((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

static inline float bf16_to_float(ushort b) {
    uint u = ((uint)b) << 16;
    return as_float(u);
}

// 1. RoPE and KV Cache Append Kernel:
// - Applies rotary embedding to Q (16 heads) and K (2 heads) in-place.
// - Uses SLM to buffer heads for completely race-free parallel rotation.
// - Converts rotated K and V to BF16 and appends to KV cache at slot `pos`.
__kernel void rope_and_kv_append_bf16(
    __global float * restrict q,           // [16, 256]
    __global float * restrict k,           // [2, 256]
    __global const float * restrict v,     // [2, 256]
    __global ushort * restrict k_cache,    // [2, max_ctx, 256]
    __global ushort * restrict v_cache,    // [2, max_ctx, 256]
    uint pos,
    uint max_ctx
) {
    int gid = get_global_id(0); // 0..255 (launch with 1 workgroup of 256 work-items)
    __local float s_buf[HEAD_DIM];

    // Precompute rotation angles for rotary dimensions 0..31
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

    // 1. Q RoPE across all 16 heads
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

    // 2. K RoPE and Cache Write across 2 heads
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

        // Store to BF16 KV cache
        __global ushort * k_slot = k_cache + (h * max_ctx + pos) * HEAD_DIM;
        __global ushort * v_slot = v_cache + (h * max_ctx + pos) * HEAD_DIM;
        k_slot[gid] = float_to_bf16(rot_val);
        v_slot[gid] = float_to_bf16(v[h * HEAD_DIM + gid]);
    }
}

// 2. GQA Attention Decode Kernel with Online Softmax:
// Launch 16 workgroups of 256 work-items (1 workgroup per Q head).
// Attends over tokens 0..pos using BF16 KV cache.
// Applies Sigmoid Output Gating: out[h, d] *= sigmoid(gate[h, d]).
__kernel void gqa_attn_decode_bf16(
    __global float * restrict out,             // [16, 256]
    __global const float * restrict q,         // [16, 256]
    __global const float * restrict gate,      // [16, 256]
    __global const ushort * restrict k_cache,  // [2, max_ctx, 256]
    __global const ushort * restrict v_cache,  // [2, max_ctx, 256]
    uint pos,                                  // attends over 0..pos (total pos+1 tokens)
    uint max_ctx
) {
    int qh = get_group_id(0); // 0..15 (Q head index)
    if (qh >= NUM_Q_HEADS) return;
    int tid = get_local_id(0); // 0..255 (dimension index d)

    int kv_h = qh / GQA_GROUP_SIZE; // 0 or 1

    __local float s_q[HEAD_DIM];
    __local float s_red[HEAD_DIM];

    // Load query head into SLM
    s_q[tid] = q[qh * HEAD_DIM + tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Running Online Softmax State
    float run_max = -1e30f;
    float run_sum = 0.0f;
    float run_acc = 0.0f;

    uint total_tokens = pos + 1;

    for (uint t = 0; t < total_tokens; ++t) {
        __global const ushort * k_slot = k_cache + (kv_h * max_ctx + t) * HEAD_DIM;
        float k_val = bf16_to_float(k_slot[tid]);

        // Dot product Q[tid] * K[tid]
        s_red[tid] = s_q[tid] * k_val;
        barrier(CLK_LOCAL_MEM_FENCE);

        // SLM reduction for dot product
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) {
                s_red[tid] += s_red[tid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float score = s_red[0] * ATTN_SCALE;
        barrier(CLK_LOCAL_MEM_FENCE);

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
