// Unit: int4_gemv_m2 (B=2 verify) vs 2x int4_gemv_m1 (decode) bitwise parity.
// Deterministic kernels must agree EXACTLY on identical inputs; any mismatch
// proves FP-order divergence (FMA contraction/scheduling) between the paths.
#include <level_zero/ze_api.h>

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

static std::vector<uint8_t> read_file(const char *p) {
  std::ifstream f(p, std::ios::binary);
  f.seekg(0, std::ios::end);
  size_t n = (size_t)f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> b(n);
  f.read((char *)b.data(), n);
  return b;
}

static uint16_t f32_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1]
                                    : "tools/kernels_258v/all_kernels.spv";
  const int M = 512, K = 2048, GS = 128;
  const int NG = K / GS;
  const int SEEDS = 5;

  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t nDrv = 0;
  CHECK(zeDriverGet(&nDrv, nullptr));
  std::vector<ze_driver_handle_t> drvs(nDrv);
  CHECK(zeDriverGet(&nDrv, drvs.data()));
  ze_driver_handle_t drv = drvs[0];
  uint32_t nDev = 0;
  CHECK(zeDeviceGet(drv, &nDev, nullptr));
  std::vector<ze_device_handle_t> devs(nDev);
  CHECK(zeDeviceGet(drv, &nDev, devs.data()));
  ze_device_handle_t dev = devs[0];
  ze_context_desc_t ctxd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t ctx = nullptr;
  CHECK(zeContextCreate(drv, &ctxd, &ctx));
  std::vector<uint8_t> spv = read_file(spv_path);
  ze_module_desc_t md = {ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
                         ZE_MODULE_FORMAT_IL_SPIRV, spv.size(), spv.data(),
                         nullptr, nullptr};
  ze_module_handle_t mod = nullptr;
  CHECK(zeModuleCreate(ctx, dev, &md, &mod, nullptr));
  ze_kernel_handle_t k_m1 = nullptr, k_m2 = nullptr;
  {
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           "int4_gemv_m1"};
    CHECK(zeKernelCreate(mod, &kd, &k_m1));
  }
  {
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           "int4_gemv_m2"};
    CHECK(zeKernelCreate(mod, &kd, &k_m2));
  }

  ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                nullptr, 0, 0, 0,
                                ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue = nullptr;
  CHECK(zeCommandQueueCreate(ctx, dev, &qd, &queue));
  ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                               nullptr, 0, 0};
  ze_command_list_handle_t list = nullptr;
  CHECK(zeCommandListCreate(ctx, dev, &ld, &list));
  ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                   nullptr, 0, 0};

  float *d_x = nullptr;    // [2, K]
  float *d_y1 = nullptr;   // [M] m1 token0
  float *d_y2 = nullptr;   // [M] m1 token1
  float *d_ym = nullptr;   // [2, M] m2
  uint8_t *d_w = nullptr;  // [M, K/2]
  uint16_t *d_s = nullptr; // [M, NG]
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)2 * K * 4, 64, dev, (void **)&d_x));
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)M * 4, 64, dev, (void **)&d_y1));
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)M * 4, 64, dev, (void **)&d_y2));
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)2 * M * 4, 64, dev,
                         (void **)&d_ym));
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)M * (K / 2), 64, dev,
                         (void **)&d_w));
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)M * NG * 2, 64, dev,
                         (void **)&d_s));
  ze_command_list_handle_t copy = nullptr;
  CHECK(zeCommandListCreate(ctx, dev, &ld, &copy));

  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_int_distribution<int> nib(-8, 7);
  std::uniform_real_distribution<float> sc(-2.0f, 2.0f);
  std::vector<uint8_t> hw((size_t)M * (K / 2));
  std::vector<uint16_t> hs((size_t)M * NG);
  for (int m = 0; m < M; ++m)
    for (int g = 0; g < NG; ++g) {
      hs[(size_t)m * NG + g] = f32_to_bf16(sc(rng));
      for (int i = 0; i < GS / 2; ++i) {
        int lo = nib(rng) & 0xF, hi = nib(rng) & 0xF;
        hw[((size_t)m * (K / 2)) + g * (GS / 2) + i] =
            (uint8_t)(lo | (hi << 4));
      }
    }

  int total_mismatch = 0;
  double worst = 0.0;
  for (int s = 0; s < SEEDS; ++s) {
    std::vector<float> hx((size_t)2 * K);
    for (size_t i = 0; i < hx.size(); ++i) {
      hx[i] = nd(rng);
      if (i % 511 == 0) hx[i] *= 40.0f; // activation outliers
    }
    CHECK(zeCommandListAppendMemoryCopy(copy, d_x, hx.data(),
                                        hx.size() * 4, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(copy, d_w, hw.data(), hw.size(),
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(copy, d_s, hs.data(), hs.size() * 2,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListClose(copy));
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &copy, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    CHECK(zeCommandListReset(copy));

    // m1 token0: y = W x[0]
    int m1m = M, m1k = K;
    float *x0 = d_x, *x1 = d_x + K;
    CHECK(zeKernelSetArgumentValue(k_m1, 0, sizeof(void *), &d_y1));
    uint8_t *wp = d_w;
    uint16_t *sp = d_s;
    CHECK(zeKernelSetArgumentValue(k_m1, 1, sizeof(void *), &wp));
    CHECK(zeKernelSetArgumentValue(k_m1, 2, sizeof(void *), &sp));
    CHECK(zeKernelSetArgumentValue(k_m1, 3, sizeof(void *), &x0));
    CHECK(zeKernelSetArgumentValue(k_m1, 4, sizeof(int), &m1m));
    CHECK(zeKernelSetArgumentValue(k_m1, 5, sizeof(int), &m1k));
    CHECK(zeKernelSetGroupSize(k_m1, 256, 1, 1));
    ze_group_count_t g1{(uint32_t)((M + 255) / 256), 1, 1};
    // m1 token1 + m2 use same list; set args per launch below via close/exec per step
    CHECK(zeCommandListAppendLaunchKernel(list, k_m1, &g1, nullptr, 0,
                                          nullptr));
    CHECK(zeCommandListClose(list));
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    CHECK(zeCommandListReset(list));

    CHECK(zeKernelSetArgumentValue(k_m1, 0, sizeof(void *), &d_y2));
    CHECK(zeKernelSetArgumentValue(k_m1, 3, sizeof(void *), &x1));
    CHECK(zeCommandListAppendLaunchKernel(list, k_m1, &g1, nullptr, 0,
                                          nullptr));
    CHECK(zeCommandListClose(list));
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    CHECK(zeCommandListReset(list));

    CHECK(zeKernelSetArgumentValue(k_m2, 0, sizeof(void *), &d_ym));
    CHECK(zeKernelSetArgumentValue(k_m2, 1, sizeof(void *), &wp));
    CHECK(zeKernelSetArgumentValue(k_m2, 2, sizeof(void *), &sp));
    CHECK(zeKernelSetArgumentValue(k_m2, 3, sizeof(void *), &d_x));
    CHECK(zeKernelSetArgumentValue(k_m2, 4, sizeof(int), &m1m));
    CHECK(zeKernelSetArgumentValue(k_m2, 5, sizeof(int), &m1k));
    CHECK(zeKernelSetGroupSize(k_m2, 256, 1, 1));
    CHECK(zeCommandListAppendLaunchKernel(list, k_m2, &g1, nullptr, 0,
                                          nullptr));
    CHECK(zeCommandListClose(list));
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    CHECK(zeCommandListReset(list));

    std::vector<float> hy1(M), hy2(M), hym((size_t)2 * M);
    ze_command_list_handle_t back = nullptr;
    CHECK(zeCommandListCreate(ctx, dev, &ld, &back));
    CHECK(zeCommandListAppendMemoryCopy(back, hy1.data(), d_y1, (size_t)M * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(back, hy2.data(), d_y2, (size_t)M * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(back, hym.data(), d_ym,
                                        (size_t)2 * M * 4, nullptr, 0,
                                        nullptr));
    CHECK(zeCommandListClose(back));
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &back, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    zeCommandListDestroy(back);

    int mm = 0;
    double w = 0.0;
    for (int m = 0; m < M; ++m) {
      uint32_t a, b;
      std::memcpy(&a, &hy1[m], 4);
      std::memcpy(&b, &hym[m], 4);
      if (a != b) {
        ++mm;
        w = std::max(w, (double)std::fabs(hy1[m] - hym[m]));
      }
      std::memcpy(&a, &hy2[m], 4);
      std::memcpy(&b, &hym[M + m], 4);
      if (a != b) {
        ++mm;
        w = std::max(w, (double)std::fabs(hy2[m] - hym[M + m]));
      }
    }
    std::printf("seed %d: mismatched outputs %d/%d, worst abs diff %.3e\n",
                s, mm, 2 * M, w);
    total_mismatch += mm;
    worst = std::max(worst, w);
  }
  std::printf(total_mismatch == 0 ? "GEMV-M1-M2-BITEXACT\n"
                                  : "GEMV-M1-M2-DIVERGE worst=%.3e\n",
              worst);
  return total_mismatch == 0 ? 0 : 2;
}
