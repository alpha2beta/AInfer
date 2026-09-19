// Deterministic Top-k Router Kernel for Qwen3.5-MoE on Intel Arc 140V (Xe2)
// Computes:
// 1. Logits: z_i = dot(x[0..2047], W_gate[i, 0..2047]) for i = 0..255
// 2. Softmax: p_i = exp(z_i - max(z)) / sum(exp(z - max(z)))
// 3. Top-k selection (k=8) with deterministic tie-breaking (smaller expert index wins)
// 4. Renormalization: w_m = p_m / sum_{j in top8}(p_j)
// 5. Shared expert gate: s = 1.0 / (1.0 + exp(-dot(x, W_shared)))

#define HIDDEN_DIM 2048
#define NUM_EXPERTS 256
#define TOP_K 8

__kernel void moe_topk_router(
    __global const float * restrict x,           // [HIDDEN_DIM]
    __global const float * restrict w_gate,      // [NUM_EXPERTS, HIDDEN_DIM]
    __global const float * restrict w_shared,    // [HIDDEN_DIM] (optional, can be NULL)
    __global uint * restrict out_topk_indices,   // [TOP_K]
    __global float * restrict out_topk_weights,  // [TOP_K]
    __global float * restrict out_shared_gate    // [1]
) {
    int lid = get_local_id(0); // 0 .. 255
    __local float local_logits[NUM_EXPERTS];
    __local float local_probs[NUM_EXPERTS];
    __local float local_max;
    __local float local_sum;

    // 1. Compute dot product for row lid
    __global const float4 *x_vec = (__global const float4 *)x;
    __global const float4 *w_vec = (__global const float4 *)(w_gate + lid * HIDDEN_DIM);

    float sum = 0.0f;
    for (int j = 0; j < HIDDEN_DIM / 4; ++j) {
        float4 xv = x_vec[j];
        float4 wv = w_vec[j];
        sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }
    local_logits[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Work-item 0 computes max logit and softmax
    if (lid == 0) {
        float m = local_logits[0];
        for (int i = 1; i < NUM_EXPERTS; ++i) {
            if (local_logits[i] > m) {
                m = local_logits[i];
            }
        }
        local_max = m;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. Compute exp(z - max)
    float exp_val = exp(local_logits[lid] - local_max);
    local_probs[lid] = exp_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float s = 0.0f;
        for (int i = 0; i < NUM_EXPERTS; ++i) {
            s += local_probs[i];
        }
        local_sum = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4. Normalize to probabilities
    local_probs[lid] = local_probs[lid] / local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // 5. Deterministic top-8 selection on work-item 0
    if (lid == 0) {
        float best_prob[TOP_K];
        uint best_idx[TOP_K];

        for (int k = 0; k < TOP_K; ++k) {
            best_prob[k] = -1.0f;
            best_idx[k] = 0xFFFFFFFF;
        }

        for (uint i = 0; i < NUM_EXPERTS; ++i) {
            float p = local_probs[i];
            // Check if p enters top-k
            if (p > best_prob[TOP_K - 1]) {
                // Find insertion position
                int pos = TOP_K - 1;
                while (pos > 0 && (p > best_prob[pos - 1] || (p == best_prob[pos - 1] && i < best_idx[pos - 1]))) {
                    pos--;
                }
                // Shift elements down
                for (int j = TOP_K - 1; j > pos; --j) {
                    best_prob[j] = best_prob[j - 1];
                    best_idx[j] = best_idx[j - 1];
                }
                best_prob[pos] = p;
                best_idx[pos] = i;
            }
        }

        // Renormalize top-8 probabilities to sum to 1.0
        float top_sum = 0.0f;
        for (int k = 0; k < TOP_K; ++k) {
            top_sum += best_prob[k];
        }
        float inv_top_sum = (top_sum > 0.0f) ? (1.0f / top_sum) : 1.0f;

        for (int k = 0; k < TOP_K; ++k) {
            out_topk_indices[k] = best_idx[k];
            out_topk_weights[k] = best_prob[k] * inv_top_sum;
        }

        // Compute shared expert gate if provided
        if (w_shared != 0 && out_shared_gate != 0) {
            __global const float4 *ws_vec = (__global const float4 *)w_shared;
            float sh_sum = 0.0f;
            for (int j = 0; j < HIDDEN_DIM / 4; ++j) {
                float4 xv = x_vec[j];
                float4 wv = ws_vec[j];
                sh_sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
            }
            // Sigmoid: 1 / (1 + exp(-x))
            out_shared_gate[0] = 1.0f / (1.0f + exp(-sh_sum));
        }
    }
}
