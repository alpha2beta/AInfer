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

// CPU reference for INT4 symmetric GEMM with batch B
void cpu_gemm_reference(float *Y, const uint8_t *w_packed, const uint16_t *w_scale,
                         const float *X, int M, int K, int B) {
  int num_groups = K / 128;
  for (int m = 0; m < M; ++m) {
    const uint8_t *row_w = w_packed + (size_t)m * (K / 2);
    const uint16_t *row_s = w_scale + (size_t)m * num_groups;

    std::vector<float> total(B, 0.0f);

    for (int g = 0; g < num_groups; ++g) {
      float scale = bf16_to_fp32(row_s[g]);
      const uint8_t *grp_w = row_w + g * 64;

      std::vector<float> grp_acc(B, 0.0f);
      for (int i = 0; i < 64; ++i) {
        uint8_t wb = grp_w[i];
        int8_t n0 = (wb & 0x0F); if (n0 >= 8) n0 -= 16;
        int8_t n1 = (wb >> 4);   if (n1 >= 8) n1 -= 16;

        for (int b = 0; b < B; ++b) {
          const float *tok_x = X + (size_t)b * K + g * 128;
          grp_acc[b] += (float)n0 * tok_x[2 * i] + (float)n1 * tok_x[2 * i + 1];
        }
      }

      for (int b = 0; b < B; ++b) {
        total[b] += grp_acc[b] * scale;
      }
    }

    for (int b = 0; b < B; ++b) {
      Y[(size_t)b * M + m] = total[b];
    }
  }
}

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1] : "tools/bench_gemv/gemm_prefill.spv";
  const int M = 2048;
  const int K = 2048;

  std::printf("=================================================================\n");
  std::printf("--- AInfer Batched Prefill GEMM Evaluation on Arc 140V (Xe2) ---\n");
  std::printf("=================================================================\n");
  std::printf("  Matrix Shape:   M = %d, K = %d\n", M, K);
  std::printf("  SPIR-V Kernel:  %s\n", spv_path);
  std::printf("-----------------------------------------------------------------\n");

  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  uint32_t drv_cnt = 0;
  CHECK(zeDriverGet(&drv_cnt, nullptr));
  if (drv_cnt == 0) return 1;
  std::vector<ze_driver_handle_t> drvs(drv_cnt);
  CHECK(zeDriverGet(&drv_cnt, drvs.data()));

  ze_driver_handle_t drv = drvs[0];
  uint32_t dev_cnt = 0;
  CHECK(zeDeviceGet(drv, &dev_cnt, nullptr));
  if (dev_cnt == 0) return 1;
  std::vector<ze_device_handle_t> devs(dev_cnt);
  CHECK(zeDeviceGet(drv, &dev_cnt, devs.data()));
  ze_device_handle_t dev = devs[0];

  ze_device_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  CHECK(zeDeviceGetProperties(dev, &props));
  std::printf("  Device:         %s\n", props.name);

  ze_context_desc_t cdesc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t ctx = nullptr;
  CHECK(zeContextCreate(drv, &cdesc, &ctx));

  ze_command_queue_desc_t qdesc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
                                ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue = nullptr;
  CHECK(zeCommandQueueCreate(ctx, dev, &qdesc, &queue));

  // Load SPIR-V
  std::ifstream spv_file(spv_path, std::ios::binary | std::ios::ate);
  if (!spv_file) {
    std::fprintf(stderr, "Cannot open %s\n", spv_path);
    return 1;
  }
  size_t spv_size = spv_file.tellg();
  spv_file.seekg(0);
  std::vector<uint8_t> spv_bytes(spv_size);
  spv_file.read((char *)spv_bytes.data(), spv_size);

  ze_module_desc_t mdesc{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
                         ZE_MODULE_FORMAT_IL_SPIRV, spv_size, spv_bytes.data(),
                         "-cl-std=CL2.0", nullptr};
  ze_module_handle_t mod = nullptr;
  CHECK(zeModuleCreate(ctx, dev, &mdesc, &mod, nullptr));

  ze_kernel_desc_t kdesc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "int4_gemm_prefill"};
  ze_kernel_handle_t kernel = nullptr;
  CHECK(zeKernelCreate(mod, &kdesc, &kernel));

  // Allocate host & device buffers
  size_t w_packed_bytes = (size_t)M * (K / 2);
  size_t w_scale_bytes = (size_t)M * (K / 128) * sizeof(uint16_t);
  const int MAX_B = 32;
  size_t x_bytes = (size_t)MAX_B * K * sizeof(float);
  size_t y_bytes = (size_t)MAX_B * M * sizeof(float);

  std::vector<uint8_t> h_w_packed(w_packed_bytes);
  std::vector<uint16_t> h_w_scale(w_scale_bytes / 2);
  std::vector<float> h_x(MAX_B * K);
  std::vector<float> h_y_ref(MAX_B * M);
  std::vector<float> h_y_dev(MAX_B * M);

  std::mt19937 rng(42);
  for (size_t i = 0; i < w_packed_bytes; ++i) h_w_packed[i] = rng() & 0xFF;
  for (size_t i = 0; i < h_w_scale.size(); ++i) {
    float s = 0.01f + 0.001f * (rng() % 50);
    uint32_t u;
    std::memcpy(&u, &s, 4);
    h_w_scale[i] = (uint16_t)(u >> 16);
  }
  for (size_t i = 0; i < h_x.size(); ++i) {
    h_x[i] = ((int)(rng() % 200) - 100) / 100.0f;
  }

  void *d_w_packed = nullptr, *d_w_scale = nullptr, *d_x = nullptr, *d_y = nullptr;
  ze_device_mem_alloc_desc_t m_dev{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  CHECK(zeMemAllocDevice(ctx, &m_dev, w_packed_bytes, 64, dev, &d_w_packed));
  CHECK(zeMemAllocDevice(ctx, &m_dev, w_scale_bytes, 64, dev, &d_w_scale));
  CHECK(zeMemAllocDevice(ctx, &m_dev, x_bytes, 64, dev, &d_x));
  CHECK(zeMemAllocDevice(ctx, &m_dev, y_bytes, 64, dev, &d_y));

  ze_command_list_desc_t ldesc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
  ze_command_list_handle_t cmd = nullptr;
  CHECK(zeCommandListCreate(ctx, dev, &ldesc, &cmd));

  CHECK(zeCommandListAppendMemoryCopy(cmd, d_w_packed, h_w_packed.data(), w_packed_bytes, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(cmd, d_w_scale, h_w_scale.data(), w_scale_bytes, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(cmd, d_x, h_x.data(), x_bytes, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(cmd));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cmd, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
  CHECK(zeCommandListDestroy(cmd));

  std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
  double baseline_us = 0.0;

  std::printf("\n%-8s | %-12s | %-14s | %-12s | %-12s | %-10s\n",
              "Batch B", "Latency (us)", "Lat/tok (us)", "Speedup", "Parity Diff", "Status");
  std::printf("---------|--------------|----------------|--------------|--------------|-----------\n");

  for (int B : batch_sizes) {
    // 1. Compute CPU reference for parity check
    cpu_gemm_reference(h_y_ref.data(), h_w_packed.data(), h_w_scale.data(), h_x.data(), M, K, B);

    // 2. Setup kernel arguments
    CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &d_y));
    CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &d_w_packed));
    CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(void *), &d_w_scale));
    CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(void *), &d_x));
    CHECK(zeKernelSetArgumentValue(kernel, 4, sizeof(int), &M));
    CHECK(zeKernelSetArgumentValue(kernel, 5, sizeof(int), &K));
    CHECK(zeKernelSetArgumentValue(kernel, 6, sizeof(int), &B));

    CHECK(zeKernelSetGroupSize(kernel, 256, 1, 1));
    ze_group_count_t gcnt{(uint32_t)((M + 255) / 256), 1, 1};

    // 3. Record command list
    ze_command_list_handle_t cmd_exec = nullptr;
    CHECK(zeCommandListCreate(ctx, dev, &ldesc, &cmd_exec));
    CHECK(zeCommandListAppendLaunchKernel(cmd_exec, kernel, &gcnt, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(cmd_exec));

    // 4. Warmup
    for (int w = 0; w < 5; ++w) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cmd_exec, nullptr));
      CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    }

    // 5. Timed iterations
    const int ITERS = 50;
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cmd_exec, nullptr));
      CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    }
    auto t1 = std::chrono::steady_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
    double lat_per_tok_us = avg_us / B;

    if (B == 1) baseline_us = avg_us;
    double speedup = (baseline_us * B) / avg_us;

    // 6. Check parity
    ze_command_list_handle_t cmd_read = nullptr;
    CHECK(zeCommandListCreate(ctx, dev, &ldesc, &cmd_read));
    CHECK(zeCommandListAppendMemoryCopy(cmd_read, h_y_dev.data(), d_y, (size_t)B * M * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListClose(cmd_read));
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cmd_read, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    CHECK(zeCommandListDestroy(cmd_read));

    float max_diff = 0.0f;
    for (size_t i = 0; i < (size_t)B * M; ++i) {
      float diff = std::fabs(h_y_dev[i] - h_y_ref[i]);
      if (diff > max_diff) max_diff = diff;
    }

    bool pass = (max_diff < 1e-4);
    std::printf("B = %-4d | %8.2f us  | %8.2f us/tok | %8.2fx     | %10.2e   | %s\n",
                B, avg_us, lat_per_tok_us, speedup, max_diff, pass ? "PASS" : "FAIL");

    CHECK(zeCommandListDestroy(cmd_exec));
  }

  std::printf("=================================================================\n");

  zeKernelDestroy(kernel);
  zeModuleDestroy(mod);
  zeMemFree(ctx, d_w_packed);
  zeMemFree(ctx, d_w_scale);
  zeMemFree(ctx, d_x);
  zeMemFree(ctx, d_y);
  zeCommandQueueDestroy(queue);
  zeContextDestroy(ctx);

  return 0;
}
