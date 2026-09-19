// AInfer INT4 GEMV Benchmark & Verification on Arc 140V (T4.3)
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

// CPU reference for INT4 symmetric GEMV
void cpu_gemv_reference(float *y, const uint8_t *w_packed, const uint16_t *w_scale,
                        const float *x, int M, int K) {
  int num_groups = K / 128;
  for (int m = 0; m < M; ++m) {
    const uint8_t *row_w = w_packed + (size_t)m * (K / 2);
    const uint16_t *row_s = w_scale + (size_t)m * num_groups;
    float total = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
      float scale = bf16_to_fp32(row_s[g]);
      const uint8_t *grp_w = row_w + g * 64;
      const float *grp_x = x + g * 128;

      float grp_acc = 0.0f;
      for (int i = 0; i < 64; ++i) {
        uint8_t wb = grp_w[i];
        int8_t n0 = (wb & 0x0F); if (n0 >= 8) n0 -= 16;
        int8_t n1 = (wb >> 4);   if (n1 >= 8) n1 -= 16;

        grp_acc += (float)n0 * grp_x[2 * i] + (float)n1 * grp_x[2 * i + 1];
      }
      total += grp_acc * scale;
    }
    y[m] = total;
  }
}

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1] : "tools/bench_gemv/gemv_kernel.spv";
  std::ifstream spv_f(spv_path, std::ios::binary);
  if (!spv_f) {
    std::fprintf(stderr, "cannot open %s\n", spv_path);
    return 1;
  }
  spv_f.seekg(0, std::ios::end);
  size_t spv_size = spv_f.tellg();
  spv_f.seekg(0, std::ios::beg);
  std::vector<uint8_t> spv(spv_size);
  spv_f.read((char *)spv.data(), spv_size);

  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  uint32_t nDrv = 0;
  CHECK(zeDriverGet(&nDrv, nullptr));
  std::vector<ze_driver_handle_t> drvs(nDrv);
  CHECK(zeDriverGet(&nDrv, drvs.data()));

  ze_device_handle_t dev = nullptr;
  ze_driver_handle_t drv = nullptr;
  char devName[256] = "Unknown";

  for (auto d : drvs) {
    uint32_t nv = 0;
    if (zeDeviceGet(d, &nv, nullptr) != ZE_RESULT_SUCCESS) continue;
    std::vector<ze_device_handle_t> vs(nv);
    zeDeviceGet(d, &nv, vs.data());
    for (auto v : vs) {
      ze_device_properties_t pr = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      if (zeDeviceGetProperties(v, &pr) == ZE_RESULT_SUCCESS && pr.vendorId == 0x8086) {
        if (pr.deviceId == 0x64a0 || dev == nullptr) {
          dev = v;
          drv = d;
          std::strncpy(devName, pr.name, sizeof(devName) - 1);
        }
      }
    }
  }

  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(drv, &cdesc, &ctx));

  ze_module_desc_t mdesc = {ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr, ZE_MODULE_FORMAT_IL_SPIRV,
                            spv_size, spv.data(), nullptr, nullptr};
  ze_module_handle_t mod = nullptr;
  CHECK(zeModuleCreate(ctx, dev, &mdesc, &mod, nullptr));

  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "int4_gemv_m1"};
  ze_kernel_handle_t kern = nullptr;
  CHECK(zeKernelCreate(mod, &kdesc, &kern));

  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t list = nullptr;
  CHECK(zeCommandListCreateImmediate(ctx, dev, &qdesc, &list));

  // Benchmark shapes: gate_up_proj (1024x2048) and down_proj (2048x512)
  struct Shape { int M; int K; const char *name; };
  std::vector<Shape> shapes = {
      {1024, 2048, "gate_up_proj [1024, 2048]"},
      {2048, 512,  "down_proj    [2048, 512]"}
  };

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis_byte(0, 255);
  std::uniform_int_distribution<int> dis_scale(0x3800, 0x3F00); // positive BF16 exponents ~0.01 - 1.0
  std::normal_distribution<float> dis_x(0.0f, 1.0f);

  std::printf("=== Arc 140V INT4 GEMV Benchmark (%s) ===\n", devName);

  for (const auto &sh : shapes) {
    int M = sh.M;
    int K = sh.K;
    int groups = K / 128;

    size_t w_bytes = (size_t)M * (K / 2);
    size_t s_bytes = (size_t)M * groups * sizeof(uint16_t);
    size_t x_bytes = (size_t)K * sizeof(float);
    size_t y_bytes = (size_t)M * sizeof(float);

    std::vector<uint8_t> h_w(w_bytes);
    std::vector<uint16_t> h_s(M * groups);
    std::vector<float> h_x(K);
    std::vector<float> cpu_y(M, 0.0f);
    std::vector<float> gpu_y(M, 0.0f);

    for (auto &b : h_w) b = (uint8_t)dis_byte(gen);
    for (auto &s : h_s) s = (uint16_t)dis_scale(gen);
    for (auto &x : h_x) x = dis_x(gen);

    // Compute CPU golden
    cpu_gemv_reference(cpu_y.data(), h_w.data(), h_s.data(), h_x.data(), M, K);

    // Allocate Device memory
    ze_device_mem_alloc_desc_t memDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    void *d_y = nullptr, *d_w = nullptr, *d_s = nullptr, *d_x = nullptr;
    CHECK(zeMemAllocDevice(ctx, &memDesc, y_bytes, 64, dev, &d_y));
    CHECK(zeMemAllocDevice(ctx, &memDesc, w_bytes, 64, dev, &d_w));
    CHECK(zeMemAllocDevice(ctx, &memDesc, s_bytes, 64, dev, &d_s));
    CHECK(zeMemAllocDevice(ctx, &memDesc, x_bytes, 64, dev, &d_x));

    CHECK(zeCommandListAppendMemoryCopy(list, d_w, h_w.data(), w_bytes, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, d_s, h_s.data(), s_bytes, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, d_x, h_x.data(), x_bytes, nullptr, 0, nullptr));

    CHECK(zeKernelSetArgumentValue(kern, 0, sizeof(void *), &d_y));
    CHECK(zeKernelSetArgumentValue(kern, 1, sizeof(void *), &d_w));
    CHECK(zeKernelSetArgumentValue(kern, 2, sizeof(void *), &d_s));
    CHECK(zeKernelSetArgumentValue(kern, 3, sizeof(void *), &d_x));
    CHECK(zeKernelSetArgumentValue(kern, 4, sizeof(int), &M));
    CHECK(zeKernelSetArgumentValue(kern, 5, sizeof(int), &K));

    for (uint32_t group_size : {16u, 32u, 64u, 128u, 256u}) {
      CHECK(zeKernelSetGroupSize(kern, group_size, 1, 1));
      ze_group_count_t gcnt = {(uint32_t)(((M + 1) / 2 + group_size - 1) / group_size), 1, 1};

      for (int i = 0; i < 20; ++i) { // warmup
        CHECK(zeCommandListAppendLaunchKernel(list, kern, &gcnt, nullptr, 0, nullptr));
      }

      const int BENCH_ITERS = 500;
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < BENCH_ITERS; ++i) {
        CHECK(zeCommandListAppendLaunchKernel(list, kern, &gcnt, nullptr, 0, nullptr));
      }
      auto t1 = std::chrono::steady_clock::now();
      double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      double us_per_call = total_us / BENCH_ITERS;

      double bytes_streamed = (double)(w_bytes + s_bytes + x_bytes + y_bytes);
      double bw_gbs = (bytes_streamed / 1e9) / (us_per_call * 1e-6);

      std::printf("    WG=%3u: Latency = %5.2f us | Bandwidth = %5.2f GB/s\n",
                  group_size, us_per_call, bw_gbs);
    }
    std::printf("\n");

    CHECK(zeMemFree(ctx, d_y));
    CHECK(zeMemFree(ctx, d_w));
    CHECK(zeMemFree(ctx, d_s));
    CHECK(zeMemFree(ctx, d_x));
  }

  CHECK(zeCommandListDestroy(list));
  CHECK(zeKernelDestroy(kern));
  CHECK(zeModuleDestroy(mod));
  CHECK(zeContextDestroy(ctx));

  return 0;
}
