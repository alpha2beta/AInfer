#include <level_zero/ze_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

#define CHECK(expr)                                                            \
  do {                                                                         \
    ze_result_t _r = (expr);                                                   \
    if (_r != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "L0 error %d at %s:%d (%s)\n", (int)_r, __FILE__,   \
                   __LINE__, #expr);                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static inline float bf16_to_fp32(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

int main() {
  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  uint32_t nDrv = 0;
  CHECK(zeDriverGet(&nDrv, nullptr));
  std::vector<ze_driver_handle_t> drvs(nDrv);
  CHECK(zeDriverGet(&nDrv, drvs.data()));

  ze_device_handle_t dev = nullptr;
  ze_driver_handle_t drv = drvs[0];
  uint32_t nv = 0;
  CHECK(zeDeviceGet(drv, &nv, nullptr));
  std::vector<ze_device_handle_t> vs(nv);
  CHECK(zeDeviceGet(drv, &nv, vs.data()));
  dev = vs[0];

  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(drv, &cdesc, &ctx));

  // Read SPIR-V
  std::ifstream spv_f("tools/bench_gemv/test_batched_moe.spv", std::ios::binary);
  if (!spv_f) {
    std::fprintf(stderr, "cannot open test_batched_moe.spv\n");
    return 1;
  }
  spv_f.seekg(0, std::ios::end);
  size_t spv_size = spv_f.tellg();
  spv_f.seekg(0, std::ios::beg);
  std::vector<uint8_t> spv(spv_size);
  spv_f.read((char *)spv.data(), spv_size);

  ze_module_desc_t mdesc = {ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr, ZE_MODULE_FORMAT_IL_SPIRV,
                            spv_size, spv.data(), nullptr, nullptr};
  ze_module_handle_t mod = nullptr;
  CHECK(zeModuleCreate(ctx, dev, &mdesc, &mod, nullptr));

  ze_kernel_desc_t kdesc_gu = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "moe_gateup_all8_ctrl"};
  ze_kernel_handle_t k_gu = nullptr;
  CHECK(zeKernelCreate(mod, &kdesc_gu, &k_gu));

  ze_kernel_desc_t kdesc_silu = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "silu_mul_all8"};
  ze_kernel_handle_t k_silu = nullptr;
  CHECK(zeKernelCreate(mod, &kdesc_silu, &k_silu));

  ze_kernel_desc_t kdesc_dn = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "moe_down_accum_all8_ctrl"};
  ze_kernel_handle_t k_dn = nullptr;
  CHECK(zeKernelCreate(mod, &kdesc_dn, &k_dn));

  const int HIDDEN = 2048;
  const int INTER = 512;
  const int GU_M = 1024;
  const int TOP_K = 8;
  const int NUM_EXP = 16; // subset for microbench

  // Sizes
  size_t gu_w_bytes = (size_t)NUM_EXP * GU_M * (HIDDEN / 2);
  size_t gu_s_bytes = (size_t)NUM_EXP * GU_M * (HIDDEN / 128) * sizeof(uint16_t);
  size_t dn_w_bytes = (size_t)NUM_EXP * HIDDEN * (INTER / 2);
  size_t dn_s_bytes = (size_t)NUM_EXP * HIDDEN * (INTER / 128) * sizeof(uint16_t);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis_byte(0, 255);
  std::uniform_int_distribution<int> dis_scale(0x3800, 0x3F00);
  std::normal_distribution<float> dis_x(0.0f, 1.0f);

  std::vector<uint8_t> h_gu_w(gu_w_bytes);
  std::vector<uint16_t> h_gu_s(NUM_EXP * GU_M * (HIDDEN / 128));
  std::vector<uint8_t> h_dn_w(dn_w_bytes);
  std::vector<uint16_t> h_dn_s(NUM_EXP * HIDDEN * (INTER / 128));
  std::vector<float> h_x(HIDDEN);
  std::vector<uint32_t> h_top_idx = {0, 2, 4, 6, 8, 10, 12, 14};
  std::vector<float> h_top_wt = {0.25f, 0.20f, 0.15f, 0.12f, 0.10f, 0.08f, 0.06f, 0.04f};

  for (auto &b : h_gu_w) b = (uint8_t)dis_byte(gen);
  for (auto &s : h_gu_s) s = (uint16_t)dis_scale(gen);
  for (auto &b : h_dn_w) b = (uint8_t)dis_byte(gen);
  for (auto &s : h_dn_s) s = (uint16_t)dis_scale(gen);
  for (auto &x : h_x) x = dis_x(gen);

  // CPU Golden
  std::vector<float> h_golden(HIDDEN, 0.0f);
  for (int k = 0; k < TOP_K; ++k) {
    uint32_t eid = h_top_idx[k];
    float wt = h_top_wt[k];

    // Gate & Up
    std::vector<float> act(INTER, 0.0f);
    for (int m = 0; m < INTER; ++m) {
      // Gate
      float sum_g = 0.0f;
      for (int g = 0; g < HIDDEN / 128; ++g) {
        float scale = bf16_to_fp32(h_gu_s[(eid * GU_M + m) * (HIDDEN / 128) + g]);
        float grp_acc = 0.0f;
        for (int i = 0; i < 64; ++i) {
          uint8_t wb = h_gu_w[(eid * GU_M + m) * (HIDDEN / 2) + g * 64 + i];
          int8_t n0 = (wb & 0x0F); if (n0 >= 8) n0 -= 16;
          int8_t n1 = (wb >> 4);   if (n1 >= 8) n1 -= 16;
          grp_acc += (float)n0 * h_x[g * 128 + 2 * i] + (float)n1 * h_x[g * 128 + 2 * i + 1];
        }
        sum_g += grp_acc * scale;
      }

      // Up
      float sum_u = 0.0f;
      int m_up = m + 512;
      for (int g = 0; g < HIDDEN / 128; ++g) {
        float scale = bf16_to_fp32(h_gu_s[(eid * GU_M + m_up) * (HIDDEN / 128) + g]);
        float grp_acc = 0.0f;
        for (int i = 0; i < 64; ++i) {
          uint8_t wb = h_gu_w[(eid * GU_M + m_up) * (HIDDEN / 2) + g * 64 + i];
          int8_t n0 = (wb & 0x0F); if (n0 >= 8) n0 -= 16;
          int8_t n1 = (wb >> 4);   if (n1 >= 8) n1 -= 16;
          grp_acc += (float)n0 * h_x[g * 128 + 2 * i] + (float)n1 * h_x[g * 128 + 2 * i + 1];
        }
        sum_u += grp_acc * scale;
      }

      float silu_g = sum_g / (1.0f + std::exp(-sum_g));
      act[m] = silu_g * sum_u;
    }

    // Down & Accum
    for (int m = 0; m < HIDDEN; ++m) {
      float sum_dn = 0.0f;
      for (int g = 0; g < INTER / 128; ++g) {
        float scale = bf16_to_fp32(h_dn_s[(eid * HIDDEN + m) * (INTER / 128) + g]);
        float grp_acc = 0.0f;
        for (int i = 0; i < 64; ++i) {
          uint8_t wb = h_dn_w[(eid * HIDDEN + m) * (INTER / 2) + g * 64 + i];
          int8_t n0 = (wb & 0x0F); if (n0 >= 8) n0 -= 16;
          int8_t n1 = (wb >> 4);   if (n1 >= 8) n1 -= 16;
          grp_acc += (float)n0 * act[g * 128 + 2 * i] + (float)n1 * act[g * 128 + 2 * i + 1];
        }
        sum_dn += grp_acc * scale;
      }
      h_golden[m] += sum_dn * wt;
    }
  }

  // Device Buffers
  ze_device_mem_alloc_desc_t memDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void *d_gu_w = nullptr, *d_gu_s = nullptr, *d_dn_w = nullptr, *d_dn_s = nullptr;
  void *d_x = nullptr, *d_top_idx = nullptr, *d_top_wt = nullptr;
  void *d_all_gu = nullptr, *d_all_act = nullptr, *d_acc = nullptr;

  CHECK(zeMemAllocDevice(ctx, &memDesc, gu_w_bytes, 64, dev, &d_gu_w));
  CHECK(zeMemAllocDevice(ctx, &memDesc, gu_s_bytes, 64, dev, &d_gu_s));
  CHECK(zeMemAllocDevice(ctx, &memDesc, dn_w_bytes, 64, dev, &d_dn_w));
  CHECK(zeMemAllocDevice(ctx, &memDesc, dn_s_bytes, 64, dev, &d_dn_s));
  CHECK(zeMemAllocDevice(ctx, &memDesc, HIDDEN * sizeof(float), 64, dev, &d_x));
  CHECK(zeMemAllocDevice(ctx, &memDesc, TOP_K * sizeof(uint32_t), 64, dev, &d_top_idx));
  CHECK(zeMemAllocDevice(ctx, &memDesc, TOP_K * sizeof(float), 64, dev, &d_top_wt));
  CHECK(zeMemAllocDevice(ctx, &memDesc, TOP_K * GU_M * sizeof(float), 64, dev, &d_all_gu));
  CHECK(zeMemAllocDevice(ctx, &memDesc, TOP_K * INTER * sizeof(float), 64, dev, &d_all_act));
  CHECK(zeMemAllocDevice(ctx, &memDesc, HIDDEN * sizeof(float), 64, dev, &d_acc));

  // Copy inputs
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t init_list = nullptr;
  CHECK(zeCommandListCreateImmediate(ctx, dev, &qdesc, &init_list));
  CHECK(zeCommandListAppendMemoryCopy(init_list, d_gu_w, h_gu_w.data(), gu_w_bytes, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(init_list, d_gu_s, h_gu_s.data(), gu_s_bytes, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(init_list, d_dn_w, h_dn_w.data(), dn_w_bytes, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(init_list, d_dn_s, h_dn_s.data(), dn_s_bytes, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(init_list, d_x, h_x.data(), HIDDEN * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(init_list, d_top_idx, h_top_idx.data(), TOP_K * sizeof(uint32_t), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(init_list, d_top_wt, h_top_wt.data(), TOP_K * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListDestroy(init_list));

  // Set Kernel Arguments
  int K_gu = HIDDEN;
  CHECK(zeKernelSetArgumentValue(k_gu, 0, sizeof(void *), &d_all_gu));
  CHECK(zeKernelSetArgumentValue(k_gu, 1, sizeof(void *), &d_gu_w));
  CHECK(zeKernelSetArgumentValue(k_gu, 2, sizeof(void *), &d_gu_s));
  CHECK(zeKernelSetArgumentValue(k_gu, 3, sizeof(void *), &d_x));
  CHECK(zeKernelSetArgumentValue(k_gu, 4, sizeof(void *), &d_top_idx));
  CHECK(zeKernelSetArgumentValue(k_gu, 5, sizeof(int), &K_gu));
  CHECK(zeKernelSetGroupSize(k_gu, 256, 1, 1));
  ze_group_count_t gcnt_gu{(TOP_K * GU_M + 255) / 256, 1, 1}; // 32 groups

  CHECK(zeKernelSetArgumentValue(k_silu, 0, sizeof(void *), &d_all_act));
  CHECK(zeKernelSetArgumentValue(k_silu, 1, sizeof(void *), &d_all_gu));
  CHECK(zeKernelSetGroupSize(k_silu, 256, 1, 1));
  ze_group_count_t gcnt_silu{(TOP_K * INTER + 255) / 256, 1, 1}; // 16 groups

  int M_dn = HIDDEN;
  int K_dn = INTER;
  CHECK(zeKernelSetArgumentValue(k_dn, 0, sizeof(void *), &d_acc));
  CHECK(zeKernelSetArgumentValue(k_dn, 1, sizeof(void *), &d_dn_w));
  CHECK(zeKernelSetArgumentValue(k_dn, 2, sizeof(void *), &d_dn_s));
  CHECK(zeKernelSetArgumentValue(k_dn, 3, sizeof(void *), &d_all_act));
  CHECK(zeKernelSetArgumentValue(k_dn, 4, sizeof(void *), &d_top_idx));
  CHECK(zeKernelSetArgumentValue(k_dn, 5, sizeof(void *), &d_top_wt));
  CHECK(zeKernelSetArgumentValue(k_dn, 6, sizeof(int), &M_dn));
  CHECK(zeKernelSetArgumentValue(k_dn, 7, sizeof(int), &K_dn));
  CHECK(zeKernelSetGroupSize(k_dn, 256, 1, 1));
  ze_group_count_t gcnt_dn{(M_dn + 255) / 256, 1, 1}; // 8 groups

  // Record a dedicated command list
  ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
  ze_command_list_handle_t cmd = nullptr;
  CHECK(zeCommandListCreate(ctx, dev, &ldesc, &cmd));

  CHECK(zeCommandListAppendLaunchKernel(cmd, k_gu, &gcnt_gu, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(cmd, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendLaunchKernel(cmd, k_silu, &gcnt_silu, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(cmd, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendLaunchKernel(cmd, k_dn, &gcnt_dn, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(cmd, nullptr, 0, nullptr));

  CHECK(zeCommandListClose(cmd));

  // Command queue & fence for execution
  ze_command_queue_handle_t queue = nullptr;
  ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
                                ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandQueueCreate(ctx, dev, &qd, &queue));
  ze_fence_desc_t fd = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  ze_fence_handle_t fence = nullptr;
  CHECK(zeFenceCreate(queue, &fd, &fence));

  // Execute once
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cmd, fence));
  CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
  CHECK(zeFenceReset(fence));

  // Read back result & verify correctness
  std::vector<float> h_out(HIDDEN, 0.0f);
  ze_command_list_handle_t r_list = nullptr;
  CHECK(zeCommandListCreateImmediate(ctx, dev, &qdesc, &r_list));
  CHECK(zeCommandListAppendMemoryCopy(r_list, h_out.data(), d_acc, HIDDEN * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListDestroy(r_list));

  float max_diff = 0.0f;
  float max_rel = 0.0f;
  for (int m = 0; m < HIDDEN; ++m) {
    float diff = std::abs(h_out[m] - h_golden[m]);
    float rel = diff / (std::abs(h_golden[m]) + 1e-6f);
    if (diff > max_diff) max_diff = diff;
    if (rel > max_rel) max_rel = rel;
  }
  std::printf("Sample [0]: GPU = %f, CPU = %f (diff = %e)\n", h_out[0], h_golden[0], std::abs(h_out[0] - h_golden[0]));
  std::printf("Sample [1]: GPU = %f, CPU = %f (diff = %e)\n", h_out[1], h_golden[1], std::abs(h_out[1] - h_golden[1]));
  std::printf("Correctness: max_abs_diff = %e, max_rel_diff = %e [%s]\n",
              max_diff, max_rel, (max_rel < 1e-3f ? "PASS" : "FAIL"));

  // Benchmark steady state
  const int ITERS = 500;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < ITERS; ++i) {
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cmd, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
  }
  auto t1 = std::chrono::steady_clock::now();
  double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  double us_per_layer = total_us / ITERS;

  std::printf("Batched 8-Expert MoE Block Latency: %5.2f us (All 8 experts combined!)\n", us_per_layer);

  return (max_diff < 1e-4f) ? 0 : 1;
}
