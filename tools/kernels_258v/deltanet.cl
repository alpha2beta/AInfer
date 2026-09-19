// DeltaNet Linear Attention Kernels for Qwen3.5-MoE on Intel Arc 140V (Xe2)
// S_V = 128, H_v = 32, H_k = 16, C_qkv = 8192, conv_kernel = 4.

#define S_V 128
#define H_V 32
#define H_K 16
#define C_QKV 8192
#define SCALE_128 0.08838834764831845f // 1.0f / sqrt(128.0f)
#define EPS 1e-6f

// 1. Depthwise 1D Convolution with kernel size 4 and SiLU activation:
// Updates conv state buffer [C_QKV, 3] in place and emits convolved QKV [8192].
__kernel void conv1d_update_silu(
    __global float * restrict qkv_out,
    __global const float * restrict qkv_in,
    __global float * restrict conv_state,
    __global const float * restrict conv_weight
) {
    int c = get_global_id(0); // 0..8191
    if (c >= C_QKV) return;

    int s_base = c * 3;
    float s0 = conv_state[s_base + 0];
    float s1 = conv_state[s_base + 1];
    float s2 = conv_state[s_base + 2];
    float x  = qkv_in[c];

    int w_base = c * 4;
    float w0 = conv_weight[w_base + 0];
    float w1 = conv_weight[w_base + 1];
    float w2 = conv_weight[w_base + 2];
    float w3 = conv_weight[w_base + 3];

    float sum = s0 * w0 + s1 * w1 + s2 * w2 + x * w3;
    float silu_val = sum / (1.0f + exp(-sum));
    qkv_out[c] = silu_val;

    // Shift state history
    conv_state[s_base + 0] = s1;
    conv_state[s_base + 1] = s2;
    conv_state[s_base + 2] = x;
}

// 2. Head L2-Normalization for Q and K:
// Normalizes num_heads (16 for Q, 16 for K) heads of 128 elements each:
// y[h, i] = x[h, i] / max(sqrt(sum(x^2)), eps)
__kernel void head_l2_norm_128(
    __global float * restrict y,
    __global const float * restrict x,
    int num_heads
) {
    int h = get_group_id(0);
    if (h >= num_heads) return;
    int lid = get_local_id(0); // 0..127

    __local float local_ss[128];
    __global const float * x_head = x + h * S_V;
    __global float * y_head = y + h * S_V;

    float val = x_head[lid];
    local_ss[lid] = val * val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction in SLM
    for (int s = 64; s > 0; s >>= 1) {
        if (lid < s) {
            local_ss[lid] += local_ss[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_l2 = 1.0f / fmax(sqrt(local_ss[0]), EPS);
    y_head[lid] = val * inv_l2;
}

// 3. Gate Precomputation for DeltaNet:
// Computes decay g and step size beta for each of the 32 value heads.
__kernel void deltanet_gate_prep(
    __global float * restrict g_out,
    __global float * restrict beta_out,
    __global const float * restrict a_in,
    __global const float * restrict b_in,
    __global const float * restrict dt_bias,
    __global const float * restrict A_log
) {
    int h = get_global_id(0); // 0..31
    if (h >= H_V) return;

    float a_val = a_in[h] + dt_bias[h];
    // softplus(a_val) = log(1 + exp(a_val))
    float dt = (a_val > 20.0f) ? a_val : log(1.0f + exp(a_val));
    float gate = -exp(A_log[h]) * dt;
    g_out[h] = exp(gate);
    beta_out[h] = 1.0f / (1.0f + exp(-b_in[h])); // sigmoid
}

// 4. DeltaNet Recurrent Decode Step:
// 32 value heads. Launch 32 workgroups of 128 work-items.
// Each work-item j (0..127) manages column j of head h with fully coalesced row loads.
__kernel void deltanet_recurrent_decode(
    __global float * restrict out,
    __global float * restrict state, // [32, 128, 128] FP32
    __global const float * restrict q, // [16, 128]
    __global const float * restrict k, // [16, 128]
    __global const float * restrict v, // [32, 128]
    __global const float * restrict g, // [32]
    __global const float * restrict beta // [32]
) {
    int h = get_group_id(0); // 0..31
    if (h >= H_V) return;
    int j = get_local_id(0); // 0..127 (column index)

    __local float s_q[S_V];
    __local float s_k[S_V];

    int kh = h / 2; // Key/Query head mapping (16 K heads -> 32 V heads)

    // Cooperatively load Q and K into SLM
    s_q[j] = q[kh * S_V + j];
    s_k[j] = k[kh * S_V + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    float v_val = v[h * S_V + j];
    float g_val = g[h];
    float b_val = beta[h];

    __global float * S_h = state + h * (S_V * S_V);

    // Step 1: Compute kv_j = sum_i S_h[i, j] * k[i] * g
    float kv_acc = 0.0f;
    for (int i = 0; i < S_V; ++i) {
        kv_acc += S_h[i * S_V + j] * s_k[i];
    }
    float kv_j = kv_acc * g_val;

    // Step 2: Compute delta_j = (v_j - kv_j) * beta
    float delta_j = (v_val - kv_j) * b_val;

    // Step 3: Update state matrix column in-place and accumulate attention output o_j
    // S_new[i, j] = g * S[i, j] + k[i] * delta_j
    // o_j = sum_i S_new[i, j] * q[i]
    float o_acc = 0.0f;
    for (int i = 0; i < S_V; ++i) {
        float s_old = S_h[i * S_V + j];
        float s_new = g_val * s_old + s_k[i] * delta_j;
        S_h[i * S_V + j] = s_new;
        o_acc += s_new * s_q[i];
    }

    // Step 4: Write scaled attention output
    out[h * S_V + j] = o_acc * SCALE_128;
}

// 5. DeltaNet Head RMSNorm and SiLU Gating:
// For each of the 32 heads:
// 1. RMSNorm(out[h, 0..127], ssm_norm_w[0..127]) with standard w scaling
// 2. Multiply by silu(z[h, 0..127])
__kernel void deltanet_head_norm_silu_z(
    __global float * restrict final_out,
    __global const float * restrict attn_out,
    __global const float * restrict z_gate,
    __global const float * restrict ssm_norm_w
) {
    int h = get_group_id(0); // 0..31
    if (h >= H_V) return;
    int j = get_local_id(0); // 0..127

    __local float local_ss[128];
    __global const float * x_head = attn_out + h * S_V;
    __global const float * z_head = z_gate + h * S_V;
    __global float * y_head = final_out + h * S_V;

    float val = x_head[j];
    local_ss[j] = val * val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 64; s > 0; s >>= 1) {
        if (j < s) {
            local_ss[j] += local_ss[j + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_rms = rsqrt(local_ss[0] / 128.0f + EPS);

    // Apply standard w scaling
    float norm_val = val * inv_rms * ssm_norm_w[j];

    // Multiply by silu(z)
    float zv = z_head[j];
    float silu_z = zv / (1.0f + exp(-zv));
    y_head[j] = norm_val * silu_z;
}
