// Elementwise & Reduction Kernels for Qwen3.5-MoE on Intel Arc 140V (Xe2)
// Hidden_size=2048, expert_dim=512, vocab_size=248320, head_dim=256, ssm_dim=128.

#define HIDDEN_DIM 2048
#define EPS 1e-6f

// 1. RMSNorm with (1 + w) zero-centered weights for hidden_dim=2048:
// y[i] = x[i] * rsqrt(mean(x^2) + eps) * (1.0f + w[i])
__kernel void rmsnorm_2048(
    __global float * restrict y,
    __global const float * restrict x,
    __global const float * restrict w
) {
    int lid = get_local_id(0); // 0..255 (launch with 1 workgroup of 256 work-items)
    __local float local_ss[256];

    // Each work-item sums 2048 / 256 = 8 elements
    float sum_sq = 0.0f;
    for (int j = lid; j < HIDDEN_DIM; j += 256) {
        float val = x[j];
        sum_sq += val * val;
    }
    local_ss[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction in SLM
    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            local_ss[lid] += local_ss[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_rms = rsqrt(local_ss[0] / (float)HIDDEN_DIM + EPS);

    // Normalize and scale with (1 + w)
    for (int j = lid; j < HIDDEN_DIM; j += 256) {
        y[j] = x[j] * inv_rms * (1.0f + w[j]);
    }
}

// 2. RMSNorm for Q/K projection heads (dim=256, num_heads=16 for Q, 2 for KV)
// Launch with num_heads workgroups of 64 or 256 work-items
__kernel void rmsnorm_head_256(
    __global float * restrict y,
    __global const float * restrict x,
    __global const float * restrict w,
    int num_heads
) {
    int h = get_group_id(0);
    if (h >= num_heads) return;
    int lid = get_local_id(0); // 0..63 (launch with 64 work-items per head)

    __local float local_ss[64];
    __global const float * x_head = x + h * 256;
    __global float * y_head = y + h * 256;

    float sum_sq = 0.0f;
    for (int j = lid; j < 256; j += 64) {
        float val = x_head[j];
        sum_sq += val * val;
    }
    local_ss[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 32; s > 0; s >>= 1) {
        if (lid < s) {
            local_ss[lid] += local_ss[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_rms = rsqrt(local_ss[0] / 256.0f + EPS);

    for (int j = lid; j < 256; j += 64) {
        y_head[j] = x_head[j] * inv_rms * (1.0f + w[j]);
    }
}

// 3. RMSNorm for SSM value heads (dim=128, num_heads=32, standard w[i] scaling)
__kernel void rmsnorm_head_128(
    __global float * restrict y,
    __global const float * restrict x,
    __global const float * restrict w,
    int num_heads
) {
    int h = get_group_id(0);
    if (h >= num_heads) return;
    int lid = get_local_id(0); // 0..31 (launch with 32 work-items per head)

    __local float local_ss[32];
    __global const float * x_head = x + h * 128;
    __global float * y_head = y + h * 128;

    float sum_sq = 0.0f;
    for (int j = lid; j < 128; j += 32) {
        float val = x_head[j];
        sum_sq += val * val;
    }
    local_ss[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 16; s > 0; s >>= 1) {
        if (lid < s) {
            local_ss[lid] += local_ss[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_rms = rsqrt(local_ss[0] / 128.0f + EPS);

    for (int j = lid; j < 128; j += 32) {
        y_head[j] = x_head[j] * inv_rms * w[j];
    }
}

// 4. SwiGLU: z[i] = silu(g[i]) * u[i] = (g[i] / (1 + exp(-g[i]))) * u[i]
__kernel void silu_mul_512(
    __global float * restrict z,
    __global const float * restrict g,
    __global const float * restrict u
) {
    int idx = get_global_id(0); // 0..511
    if (idx >= 512) return;
    float g_val = g[idx];
    float u_val = u[idx];
    float silu_g = g_val / (1.0f + exp(-g_val));
    z[idx] = silu_g * u_val;
}

// 5. Residual Add: y[i] = x1[i] + x2[i]
__kernel void residual_add_2048(
    __global float * restrict y,
    __global const float * restrict x1,
    __global const float * restrict x2
) {
    int idx = get_global_id(0);
    if (idx >= HIDDEN_DIM) return;
    y[idx] = x1[idx] + x2[idx];
}

// 6. MoE Expert Output Accumulator:
// out[i] += expert_out[i] * weight
__kernel void moe_accumulate_expert(
    __global float * restrict out,
    __global const float * restrict expert_out,
    float weight
) {
    int idx = get_global_id(0);
    if (idx >= HIDDEN_DIM) return;
    out[idx] += expert_out[idx] * weight;
}

// 7. MoE Add Shared Expert with sigmoid gate:
// out[i] += shared_out[i] * sigmoid(shared_gate_val)
__kernel void moe_add_shared_expert(
    __global float * restrict out,
    __global const float * restrict shared_out,
    float shared_gate_val
) {
    int idx = get_global_id(0);
    if (idx >= HIDDEN_DIM) return;
    float g = 1.0f / (1.0f + exp(-shared_gate_val));
    out[idx] += shared_out[idx] * g;
}

// 8. Vocabulary Argmax for Vocab=248320:
// Splits 248320 elements across 970 workgroups of 256 work-items.
// Group passes partial (max_val, max_idx) to stage 2.
__kernel void argmax_stage1(
    __global const float * restrict logits,
    __global float * restrict group_max_val,
    __global uint * restrict group_max_idx,
    uint N
) {
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int idx = gid * 256 + lid;

    __local float s_val[256];
    __local uint s_idx[256];

    if (idx < N) {
        s_val[lid] = logits[idx];
        s_idx[lid] = idx;
    } else {
        s_val[lid] = -1e30f;
        s_idx[lid] = 0xFFFFFFFF;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            if (s_val[lid + s] > s_val[lid] || (s_val[lid + s] == s_val[lid] && s_idx[lid + s] < s_idx[lid])) {
                s_val[lid] = s_val[lid + s];
                s_idx[lid] = s_idx[lid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        group_max_val[gid] = s_val[0];
        group_max_idx[gid] = s_idx[0];
    }
}

// Argmax Stage 2: Reduces up to 1024 partials into the final token ID
__kernel void argmax_stage2(
    __global const float * restrict group_max_val,
    __global const uint * restrict group_max_idx,
    __global uint * restrict out_token_id,
    uint num_groups
) {
    int lid = get_local_id(0); // 0..255 or 0..1023
    if (lid == 0) {
        float best_val = group_max_val[0];
        uint best_id = group_max_idx[0];
        for (uint i = 1; i < num_groups; ++i) {
            float v = group_max_val[i];
            uint id = group_max_idx[i];
            if (v > best_val || (v == best_val && id < best_id)) {
                best_val = v;
                best_id = id;
            }
        }
        out_token_id[0] = best_id;
    }
}
