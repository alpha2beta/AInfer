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

// Baseline: 3 barriers per position
__kernel void gqa_attn_decode_baseline(
    __global float * restrict out,
    __global const float * restrict q,
    __global const float * restrict gate,
    __global const ushort * restrict k_cache,
    __global const ushort * restrict v_cache,
    uint pos,
    uint max_ctx
) {
    int qh = get_group_id(0);
    if (qh >= NUM_Q_HEADS) return;
    int tid = get_local_id(0);

    int kv_h = qh / GQA_GROUP_SIZE;

    __local float s_red[HEAD_DIM];
    __local float s_score[1];

    float qv = q[qh * HEAD_DIM + tid];

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float run_acc = 0.0f;

    uint total_tokens = pos + 1;

    for (uint t = 0; t < total_tokens; ++t) {
        __global const ushort * k_slot = k_cache + (kv_h * max_ctx + t) * HEAD_DIM;
        float k_val = bf16_to_float(k_slot[tid]);

        s_red[tid] = qv * k_val;
        barrier(CLK_LOCAL_MEM_FENCE);

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

        __global const ushort * v_slot = v_cache + (kv_h * max_ctx + t) * HEAD_DIM;
        float v_val = bf16_to_float(v_slot[tid]);

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

    float attn_val = run_acc / run_sum;
    float g_val = gate[qh * HEAD_DIM + tid];
    float sig_g = 1.0f / (1.0f + exp(-g_val));

    out[qh * HEAD_DIM + tid] = attn_val * sig_g;
}

// Subgroup 4-token unroll with double-buffered SLM (1 barrier per 4 tokens)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gqa_attn_decode_sg_u4(
    __global float * restrict out,
    __global const float * restrict q,
    __global const float * restrict gate,
    __global const ushort * restrict k_cache,
    __global const ushort * restrict v_cache,
    uint pos,
    uint max_ctx
) {
    int qh = get_group_id(0);
    if (qh >= NUM_Q_HEADS) return;
    int tid = get_local_id(0);
    int lane = get_sub_group_local_id();
    int sg = get_sub_group_id();

    int kv_h = qh / GQA_GROUP_SIZE;
    float qv = q[qh * HEAD_DIM + tid];

    __local float s_part[2][64]; // Double buffer: 2 x (16 subgroups x 4 positions)

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float run_acc = 0.0f;

    uint total_tokens = pos + 1;

    for (uint tb = 0; tb < total_tokens; tb += 4) {
        int buf_idx = (int)((tb >> 2) & 1);
        float part[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            uint t = tb + (uint)u;
            float kval = 0.0f;
            if (t < total_tokens) {
                __global const ushort * k_slot = k_cache + ((size_t)kv_h * max_ctx + t) * HEAD_DIM;
                kval = bf16_to_float(k_slot[tid]);
            }
            part[u] = qv * kval;
        }

        // Subgroup butterfly reduction across the 16 lanes (register-only)
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            float v = part[u];
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 8));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 4));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 2));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 1));
            part[u] = v;
        }

        if (lane == 0) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                s_part[buf_idx][sg * 4 + u] = part[u];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                s += s_part[buf_idx][i * 4 + u];
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

    float attn_val = run_acc / run_sum;
    float g_val = gate[qh * HEAD_DIM + tid];
    float sig_g = 1.0f / (1.0f + exp(-g_val));

    out[qh * HEAD_DIM + tid] = attn_val * sig_g;
}

// Subgroup 8-token unroll with double-buffered SLM (1 barrier per 8 tokens)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gqa_attn_decode_sg_u8(
    __global float * restrict out,
    __global const float * restrict q,
    __global const float * restrict gate,
    __global const ushort * restrict k_cache,
    __global const ushort * restrict v_cache,
    uint pos,
    uint max_ctx
) {
    int qh = get_group_id(0);
    if (qh >= NUM_Q_HEADS) return;
    int tid = get_local_id(0);
    int lane = get_sub_group_local_id();
    int sg = get_sub_group_id();

    int kv_h = qh / GQA_GROUP_SIZE;
    float qv = q[qh * HEAD_DIM + tid];

    __local float s_part[2][128]; // Double buffer: 2 x (16 subgroups x 8 positions)

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float run_acc = 0.0f;

    uint total_tokens = pos + 1;

    for (uint tb = 0; tb < total_tokens; tb += 8) {
        int buf_idx = (int)((tb >> 3) & 1);
        float part[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            uint t = tb + (uint)u;
            float kval = 0.0f;
            if (t < total_tokens) {
                __global const ushort * k_slot = k_cache + ((size_t)kv_h * max_ctx + t) * HEAD_DIM;
                kval = bf16_to_float(k_slot[tid]);
            }
            part[u] = qv * kval;
        }

        // Subgroup butterfly reduction across the 16 lanes (register-only)
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            float v = part[u];
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 8));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 4));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 2));
            v += intel_sub_group_shuffle(v, (uint)(lane ^ 1));
            part[u] = v;
        }

        if (lane == 0) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                s_part[buf_idx][sg * 8 + u] = part[u];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                s += s_part[buf_idx][i * 8 + u];
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

    float attn_val = run_acc / run_sum;
    float g_val = gate[qh * HEAD_DIM + tid];
    float sig_g = 1.0f / (1.0f + exp(-g_val));

    out[qh * HEAD_DIM + tid] = attn_val * sig_g;
}

// Subgroup 16-token unroll with double-buffered SLM (1 barrier per 16 tokens)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gqa_attn_decode_sg_u16(
    __global float * restrict out,
    __global const float * restrict q,
    __global const float * restrict gate,
    __global const ushort * restrict k_cache,
    __global const ushort * restrict v_cache,
    uint pos,
    uint max_ctx
) {
    int qh = get_group_id(0);
    if (qh >= NUM_Q_HEADS) return;
    int tid = get_local_id(0);
    int lane = get_sub_group_local_id();
    int sg = get_sub_group_id();

    int kv_h = qh / GQA_GROUP_SIZE;
    float qv = q[qh * HEAD_DIM + tid];

    __local float s_part[2][256]; // Double buffer: 2 x (16 subgroups x 16 positions)

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

        // Subgroup butterfly reduction across the 16 lanes (register-only)
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

    float attn_val = run_acc / run_sum;
    float g_val = gate[qh * HEAD_DIM + tid];
    float sig_g = 1.0f / (1.0f + exp(-g_val));

    out[qh * HEAD_DIM + tid] = attn_val * sig_g;
}
