#define private public
#include "runtime_258v.h"
#undef private

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ainfer;

int main(int argc, char **argv) {
  const char *binfer = (argc > 1) ? argv[1] : "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv = (argc > 2) ? argv[2] : "tools/kernels_258v/all_kernels.spv";

  AInferRuntime258V rt;
  if (!rt.init(binfer, spv, 2048)) {
    std::fprintf(stderr, "rt.init failed\n");
    return 1;
  }

  // Pre-fill prompt with 1 token so d_x_post_ is filled with real layer 0 input
  // Let's inspect layer 0
  auto &lb = rt.layers_[0];
  printf("Testing Layer 0 MoE Parity...\n");

  // Create an immediate command list for custom testing
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t cmd = nullptr;
  CHECK_L0(zeCommandListCreateImmediate(rt.ctx_, rt.dev_, &qdesc, &cmd));

  // Initialize d_x_post_ with some non-zero values (e.g. 1.0f or random)
  std::vector<float> h_x(HIDDEN_DIM);
  for (int i = 0; i < HIDDEN_DIM; ++i) {
    h_x[i] = std::sin((float)(i + 1) * 0.1f);
  }
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd, rt.d_x_post_, h_x.data(), HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));

  // 1. Run Router
  uint32_t *d_top_idx_ptr = rt.d_ctrl_->top_idx;
  float *d_top_wt_ptr = rt.d_ctrl_->top_wt;
  float *d_sh_gate_ptr = &rt.d_ctrl_->sh_gate_val;
  ze_group_count_t gcnt_router{1, 1, 1};

  CHECK_L0(zeKernelSetArgumentValue(rt.k_router_, 0, sizeof(void *), &rt.d_x_post_));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_router_, 1, sizeof(void *), &lb.router_w));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_router_, 2, sizeof(void *), &lb.shared_gate_w));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_router_, 3, sizeof(void *), &d_top_idx_ptr));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_router_, 4, sizeof(void *), &d_top_wt_ptr));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_router_, 5, sizeof(void *), &d_sh_gate_ptr));
  CHECK_L0(zeKernelSetGroupSize(rt.k_router_, 256, 1, 1));
  CHECK_L0(zeCommandListAppendLaunchKernel(cmd, rt.k_router_, &gcnt_router, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(cmd, nullptr, 0, nullptr));

  // Read back router top_idx and top_wt
  printf("Router results:\n");
  for (int k = 0; k < TOP_K; ++k) {
    printf("  Expert %d: idx = %u, wt = %f\n", k, rt.d_ctrl_->top_idx[k], rt.d_ctrl_->top_wt[k]);
  }

  // 2. Run Serial MoE
  // Clear d_moe_acc_
  std::vector<float> zeros(HIDDEN_DIM, 0.0f);
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd, rt.d_moe_acc_, zeros.data(), HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));

  int M_gu = 2 * EXP_INTER_DIM, K_gu = HIDDEN_DIM;
  int M_dn = HIDDEN_DIM, K_dn = EXP_INTER_DIM;
  ze_group_count_t gcnt_silu{(uint32_t)((EXP_INTER_DIM + 255) / 256), 1, 1};

  std::vector<std::vector<float>> serial_gu(TOP_K, std::vector<float>(M_gu));
  std::vector<std::vector<float>> serial_act(TOP_K, std::vector<float>(EXP_INTER_DIM));

  for (int k = 0; k < TOP_K; ++k) {
    // Gate_up GEMV
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gemv_, 0, sizeof(void *), &rt.d_exp_gu_));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gemv_, 1, sizeof(void *), &lb.exp_gu_w));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gemv_, 2, sizeof(void *), &lb.exp_gu_s));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gemv_, 3, sizeof(void *), &rt.d_x_post_));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gemv_, 4, sizeof(void *), &d_top_idx_ptr));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gemv_, 5, sizeof(int), &k));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gemv_, 6, sizeof(int), &M_gu));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gemv_, 7, sizeof(int), &K_gu));
    CHECK_L0(zeKernelSetGroupSize(rt.k_exp_gemv_, 256, 1, 1));
    ze_group_count_t gc_gu{(uint32_t)((M_gu + 255) / 256), 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd, rt.k_exp_gemv_, &gc_gu, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd, nullptr, 0, nullptr));

    CHECK_L0(zeCommandListAppendMemoryCopy(cmd, serial_gu[k].data(), rt.d_exp_gu_, M_gu * sizeof(float), nullptr, 0, nullptr));

    // SwiGLU 512
    float *d_exp_gu_up = rt.d_exp_gu_ + EXP_INTER_DIM;
    CHECK_L0(zeKernelSetArgumentValue(rt.k_silu512_, 0, sizeof(void *), &rt.d_exp_act_));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_silu512_, 1, sizeof(void *), &rt.d_exp_gu_));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_silu512_, 2, sizeof(void *), &d_exp_gu_up));
    CHECK_L0(zeKernelSetGroupSize(rt.k_silu512_, 256, 1, 1));
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd, rt.k_silu512_, &gcnt_silu, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd, nullptr, 0, nullptr));

    CHECK_L0(zeCommandListAppendMemoryCopy(cmd, serial_act[k].data(), rt.d_exp_act_, EXP_INTER_DIM * sizeof(float), nullptr, 0, nullptr));

    // Down GEMV + Accum
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_, 0, sizeof(void *), &rt.d_moe_acc_));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_, 1, sizeof(void *), &lb.exp_dn_w));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_, 2, sizeof(void *), &lb.exp_dn_s));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_, 3, sizeof(void *), &rt.d_exp_act_));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_, 4, sizeof(void *), &d_top_idx_ptr));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_, 5, sizeof(void *), &d_top_wt_ptr));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_, 6, sizeof(int), &k));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_, 7, sizeof(int), &M_dn));
    CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_, 8, sizeof(int), &K_dn));
    CHECK_L0(zeKernelSetGroupSize(rt.k_exp_dn_accum_, 256, 1, 1));
    ze_group_count_t gc_dn{(uint32_t)((M_dn + 255) / 256), 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd, rt.k_exp_dn_accum_, &gc_dn, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd, nullptr, 0, nullptr));
  }

  std::vector<float> serial_moe_acc(HIDDEN_DIM);
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd, serial_moe_acc.data(), rt.d_moe_acc_, HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));

  // 3. Run Batched MoE
  // Allocate temporary output buffer for batched moe acc
  ze_device_mem_alloc_desc_t mdesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void *d_moe_acc_batched = nullptr;
  CHECK_L0(zeMemAllocDevice(rt.ctx_, &mdesc, HIDDEN_DIM * sizeof(float), 64, rt.dev_, &d_moe_acc_batched));

  // Batched Gate_up GEMV [8, 1024]
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gu_all_, 0, sizeof(void *), &rt.d_exp_gu_all_));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gu_all_, 1, sizeof(void *), &lb.exp_gu_w));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gu_all_, 2, sizeof(void *), &lb.exp_gu_s));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gu_all_, 3, sizeof(void *), &rt.d_x_post_));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gu_all_, 4, sizeof(void *), &d_top_idx_ptr));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_gu_all_, 5, sizeof(int), &K_gu));
  CHECK_L0(zeKernelSetGroupSize(rt.k_exp_gu_all_, 256, 1, 1));
  ze_group_count_t gc_gu_all{(uint32_t)((TOP_K * M_gu + 255) / 256), 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(cmd, rt.k_exp_gu_all_, &gc_gu_all, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(cmd, nullptr, 0, nullptr));

  std::vector<float> batched_gu(TOP_K * M_gu);
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd, batched_gu.data(), rt.d_exp_gu_all_, TOP_K * M_gu * sizeof(float), nullptr, 0, nullptr));

  // Batched SwiGLU [8, 512]
  CHECK_L0(zeKernelSetArgumentValue(rt.k_silu_all_, 0, sizeof(void *), &rt.d_exp_act_all_));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_silu_all_, 1, sizeof(void *), &rt.d_exp_gu_all_));
  CHECK_L0(zeKernelSetGroupSize(rt.k_silu_all_, 256, 1, 1));
  ze_group_count_t gc_silu_all{(uint32_t)((TOP_K * EXP_INTER_DIM + 255) / 256), 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(cmd, rt.k_silu_all_, &gc_silu_all, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(cmd, nullptr, 0, nullptr));

  std::vector<float> batched_act(TOP_K * EXP_INTER_DIM);
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd, batched_act.data(), rt.d_exp_act_all_, TOP_K * EXP_INTER_DIM * sizeof(float), nullptr, 0, nullptr));

  // Batched Down GEMV + Accum [2048]
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_all_, 0, sizeof(void *), &d_moe_acc_batched));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_all_, 1, sizeof(void *), &lb.exp_dn_w));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_all_, 2, sizeof(void *), &lb.exp_dn_s));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_all_, 3, sizeof(void *), &rt.d_exp_act_all_));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_all_, 4, sizeof(void *), &d_top_idx_ptr));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_all_, 5, sizeof(void *), &d_top_wt_ptr));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_all_, 6, sizeof(int), &M_dn));
  CHECK_L0(zeKernelSetArgumentValue(rt.k_exp_dn_accum_all_, 7, sizeof(int), &K_dn));
  CHECK_L0(zeKernelSetGroupSize(rt.k_exp_dn_accum_all_, 256, 1, 1));
  ze_group_count_t gc_dn_all{(uint32_t)((M_dn + 255) / 256), 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(cmd, rt.k_exp_dn_accum_all_, &gc_dn_all, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(cmd, nullptr, 0, nullptr));

  std::vector<float> batched_moe_acc(HIDDEN_DIM);
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd, batched_moe_acc.data(), d_moe_acc_batched, HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));

  // Compare Gate_up
  printf("\n--- Gate_Up Parity Check ---\n");
  for (int k = 0; k < TOP_K; ++k) {
    float max_diff = 0.0f;
    for (int m = 0; m < M_gu; ++m) {
      float diff = std::abs(serial_gu[k][m] - batched_gu[k * M_gu + m]);
      if (diff > max_diff) max_diff = diff;
    }
    printf("Expert %d (eid %u) Gate_Up max_diff: %e\n", k, rt.d_ctrl_->top_idx[k], max_diff);
  }

  // Compare SwiGLU Act
  printf("\n--- SwiGLU Act Parity Check ---\n");
  for (int k = 0; k < TOP_K; ++k) {
    float max_diff = 0.0f;
    for (int m = 0; m < EXP_INTER_DIM; ++m) {
      float diff = std::abs(serial_act[k][m] - batched_act[k * EXP_INTER_DIM + m]);
      if (diff > max_diff) max_diff = diff;
    }
    printf("Expert %d (eid %u) SwiGLU Act max_diff: %e\n", k, rt.d_ctrl_->top_idx[k], max_diff);
  }

  // Compare Moe Acc
  printf("\n--- Final Moe Acc Parity Check ---\n");
  float max_diff_acc = 0.0f;
  float max_rel_acc = 0.0f;
  for (int m = 0; m < HIDDEN_DIM; ++m) {
    float diff = std::abs(serial_moe_acc[m] - batched_moe_acc[m]);
    float rel = diff / (std::abs(serial_moe_acc[m]) + 1e-6f);
    if (diff > max_diff_acc) max_diff_acc = diff;
    if (rel > max_rel_acc) max_rel_acc = rel;
  }
  printf("Moe Acc: max_diff = %e, max_rel = %e\n", max_diff_acc, max_rel_acc);
  printf("Sample 0: serial = %f, batched = %f\n", serial_moe_acc[0], batched_moe_acc[0]);
  printf("Sample 1: serial = %f, batched = %f\n", serial_moe_acc[1], batched_moe_acc[1]);
  printf("Sample 2: serial = %f, batched = %f\n", serial_moe_acc[2], batched_moe_acc[2]);

  zeMemFree(rt.ctx_, d_moe_acc_batched);
  zeCommandListDestroy(cmd);
  return 0;
}
