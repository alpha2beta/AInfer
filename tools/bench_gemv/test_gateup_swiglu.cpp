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
  std::ifstream spv_f("tools/bench_gemv/test_gateup_swiglu.spv", std::ios::binary);
  if (!spv_f) {
    std::fprintf(stderr, "cannot open test_gateup_swiglu.spv\n");
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

  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "moe_gateup_swiglu"};
  ze_kernel_handle_t kern = nullptr;
  CHECK(zeKernelCreate(mod, &kdesc, &kern));

  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t list = nullptr;
  CHECK(zeCommandListCreateImmediate(ctx, dev, &qdesc, &list));

  const int M = 512;
  const int K = 2048;
  const int groups = K / 128;

  // w_bank: 1024 rows of K/2 bytes
  size_t w_bytes = (size_t)1024 * (K / 2);
  size_t s_bytes = (size_t)1024 * groups * sizeof(uint16_t);
  size_t x_bytes = (size_t)K * sizeof(float);
  size_t act_bytes = (size_t)M * sizeof(float);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis_byte(0, 255);
  std::uniform_int_distribution<int> dis_scale(0x3800, 0x3F00);
  std::normal_distribution<float> dis_x(0.0f, 1.0f);

  std::vector<uint8_t> h_w(w_bytes);
  std::vector<uint16_t> h_s(1024 * groups);
  std::vector<float> h_x(K);
  std::vector<float> h_golden(M, 0.0f);
  std::vector<float> h_out(M, 0.0f);

  for (auto &b : h_w) b = (uint8_t)dis_byte(gen);
  for (auto &s : h_s) s = (uint16_t)dis_scale(gen);
  for (auto &x : h_x) x = dis_x(gen);

  // Compute CPU Golden
  for (int m = 0; m < M; ++m) {
    // Gate
    float sum_g = 0.0f;
    for (int g = 0; g < groups; ++g) {
      float scale = bf16_to_fp32(h_s[m * groups + g]);
      float grp_acc = 0.0f;
      for (int i = 0; i < 64; ++i) {
        uint8_t wb = h_w[m * (K / 2) + g * 64 + i];
        int8_t n0 = (wb & 0x0F); if (n0 >= 8) n0 -= 16;
        int8_t n1 = (wb >> 4);   if (n1 >= 8) n1 -= 16;
        grp_acc += (float)n0 * h_x[g * 128 + 2 * i] + (float)n1 * h_x[g * 128 + 2 * i + 1];
      }
      sum_g += grp_acc * scale;
    }

    // Up
    float sum_u = 0.0f;
    int m_up = m + 512;
    for (int g = 0; g < groups; ++g) {
      float scale = bf16_to_fp32(h_s[m_up * groups + g]);
      float grp_acc = 0.0f;
      for (int i = 0; i < 64; ++i) {
        uint8_t wb = h_w[m_up * (K / 2) + g * 64 + i];
        int8_t n0 = (wb & 0x0F); if (n0 >= 8) n0 -= 16;
        int8_t n1 = (wb >> 4);   if (n1 >= 8) n1 -= 16;
        grp_acc += (float)n0 * h_x[g * 128 + 2 * i] + (float)n1 * h_x[g * 128 + 2 * i + 1];
      }
      sum_u += grp_acc * scale;
    }

    float silu_g = sum_g / (1.0f + std::exp(-sum_g));
    h_golden[m] = silu_g * sum_u;
  }

  // GPU Memory
  ze_device_mem_alloc_desc_t memDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void *d_act = nullptr, *d_w = nullptr, *d_s = nullptr, *d_x = nullptr;
  uint32_t top_idx_val = 0;
  void *d_top_idx = nullptr;

  CHECK(zeMemAllocDevice(ctx, &memDesc, act_bytes, 64, dev, &d_act));
  CHECK(zeMemAllocDevice(ctx, &memDesc, w_bytes, 64, dev, &d_w));
  CHECK(zeMemAllocDevice(ctx, &memDesc, s_bytes, 64, dev, &d_s));
  CHECK(zeMemAllocDevice(ctx, &memDesc, x_bytes, 64, dev, &d_x));
  CHECK(zeMemAllocDevice(ctx, &memDesc, sizeof(uint32_t), 64, dev, &d_top_idx));

  CHECK(zeCommandListAppendMemoryCopy(list, d_w, h_w.data(), w_bytes, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_s, h_s.data(), s_bytes, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_x, h_x.data(), x_bytes, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_top_idx, &top_idx_val, sizeof(uint32_t), nullptr, 0, nullptr));

  int k_slot = 0;
  int M_val = M;
  int K_val = K;

  CHECK(zeKernelSetArgumentValue(kern, 0, sizeof(void *), &d_act));
  CHECK(zeKernelSetArgumentValue(kern, 1, sizeof(void *), &d_w));
  CHECK(zeKernelSetArgumentValue(kern, 2, sizeof(void *), &d_s));
  CHECK(zeKernelSetArgumentValue(kern, 3, sizeof(void *), &d_x));
  CHECK(zeKernelSetArgumentValue(kern, 4, sizeof(void *), &d_top_idx));
  CHECK(zeKernelSetArgumentValue(kern, 5, sizeof(int), &k_slot));
  CHECK(zeKernelSetArgumentValue(kern, 6, sizeof(int), &M_val));
  CHECK(zeKernelSetArgumentValue(kern, 7, sizeof(int), &K_val));

  uint32_t group_size = 256;
  CHECK(zeKernelSetGroupSize(kern, group_size, 1, 1));
  ze_group_count_t gcnt = {(uint32_t)((M + group_size - 1) / group_size), 1, 1};

  // Warmup
  for (int i = 0; i < 20; ++i) {
    CHECK(zeCommandListAppendLaunchKernel(list, kern, &gcnt, nullptr, 0, nullptr));
  }

  // Verify correctness
  CHECK(zeCommandListAppendMemoryCopy(list, h_out.data(), d_act, act_bytes, nullptr, 0, nullptr));

  float max_diff = 0.0f;
  for (int m = 0; m < M; ++m) {
    float diff = std::abs(h_out[m] - h_golden[m]);
    if (diff > max_diff) max_diff = diff;
  }
  std::printf("Correctness Check: max_diff = %e [%s]\n", max_diff, (max_diff < 1e-4f ? "PASS" : "FAIL"));

  // Benchmark latency
  const int ITERS = 500;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < ITERS; ++i) {
    CHECK(zeCommandListAppendLaunchKernel(list, kern, &gcnt, nullptr, 0, nullptr));
  }
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
  double bytes = (double)(w_bytes + s_bytes + x_bytes + act_bytes);
  double bw = (bytes / 1e9) / (us * 1e-6);

  std::printf("Fused GateUp+SwiGLU: Latency = %5.2f us | Bandwidth = %5.2f GB/s\n", us, bw);

  return (max_diff < 1e-4f) ? 0 : 1;
}
