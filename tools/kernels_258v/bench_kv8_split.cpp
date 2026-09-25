// Unit: KV8 split-T decode attention vs legacy kv8_attn_ctrl.
// Functional parity (ulp-level) + rough timing per (T, S).
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

#define CK(expr)                                                               \
  do {                                                                         \
    ze_result_t _r = (expr);                                                   \
    if (_r != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "L0 error %d at %s:%d (%s)\n", (int)_r, __FILE__,   \
                   __LINE__, #expr);                                          \
      ok = 0;                                                                  \
    }                                                                          \
  } while (0)

static int ok = 1;
static std::vector<uint8_t> read_file(const char *p) {
  std::ifstream f(p, std::ios::binary);
  f.seekg(0, std::ios::end);
  size_t n = (size_t)f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> b(n);
  f.read((char *)b.data(), n);
  return b;
}

int main(int argc, char **argv) {
  const char *spv_path =
      (argc > 1) ? argv[1] : "tools/kernels_258v/all_kernels.spv.kv8";
  const int HD = 256, NQ = 16, NKV = 2, MAXC = 8192, SMAX = 8;

  CK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t nDrv = 0;
  CK(zeDriverGet(&nDrv, nullptr));
  std::vector<ze_driver_handle_t> drvs(nDrv);
  CK(zeDriverGet(&nDrv, drvs.data()));
  uint32_t nDev = 0;
  CK(zeDeviceGet(drvs[0], &nDev, nullptr));
  std::vector<ze_device_handle_t> devs(nDev);
  CK(zeDeviceGet(drvs[0], &nDev, devs.data()));
  ze_device_handle_t dev = devs[0];
  ze_context_desc_t ctxd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t ctx = nullptr;
  CK(zeContextCreate(drvs[0], &ctxd, &ctx));
  std::vector<uint8_t> spv = read_file(spv_path);
  ze_module_desc_t md = {ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
                         ZE_MODULE_FORMAT_IL_SPIRV, spv.size(), spv.data(),
                         nullptr, nullptr};
  ze_module_handle_t mod = nullptr;
  CK(zeModuleCreate(ctx, dev, &md, &mod, nullptr));
  ze_kernel_handle_t k_leg = nullptr, k_sp = nullptr, k_cb = nullptr;
  {
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           "kv8_attn_ctrl"};
    CK(zeKernelCreate(mod, &kd, &k_leg));
  }
  {
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           "kv8_attn_decode_split"};
    CK(zeKernelCreate(mod, &kd, &k_sp));
  }
  {
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           "kv8_attn_combine"};
    CK(zeKernelCreate(mod, &kd, &k_cb));
  }
  if (!ok) return 1;

  ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                nullptr, 0, 0, 0,
                                ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue = nullptr;
  CK(zeCommandQueueCreate(ctx, dev, &qd, &queue));
  ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                               nullptr, 0, 0};
  ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                   nullptr, 0, 0};
  float *d_q = nullptr, *d_gate = nullptr, *d_o0 = nullptr, *d_o1 = nullptr,
        *d_part = nullptr, *d_ks = nullptr, *d_vs = nullptr;
  int8_t *d_kc = nullptr, *d_vc = nullptr;
  int32_t *d_ctrl = nullptr;
  CK(zeMemAllocDevice(ctx, &dd, (size_t)NQ * HD * 4, 64, dev, (void **)&d_q));
  CK(zeMemAllocDevice(ctx, &dd, (size_t)NQ * HD * 4, 64, dev,
                      (void **)&d_gate));
  CK(zeMemAllocDevice(ctx, &dd, (size_t)NQ * HD * 4, 64, dev, (void **)&d_o0));
  CK(zeMemAllocDevice(ctx, &dd, (size_t)NQ * HD * 4, 64, dev, (void **)&d_o1));
  CK(zeMemAllocDevice(ctx, &dd, (size_t)NQ * SMAX * 258 * 4, 64, dev,
                      (void **)&d_part));
  CK(zeMemAllocDevice(ctx, &dd, (size_t)NKV * MAXC * HD, 64, dev,
                      (void **)&d_kc));
  CK(zeMemAllocDevice(ctx, &dd, (size_t)NKV * MAXC * HD, 64, dev,
                      (void **)&d_vc));
  CK(zeMemAllocDevice(ctx, &dd, (size_t)MAXC * NKV * 4, 64, dev,
                      (void **)&d_ks));
  CK(zeMemAllocDevice(ctx, &dd, (size_t)MAXC * NKV * 4, 64, dev,
                      (void **)&d_vs));
  ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                 nullptr, 0};
  CK(zeMemAllocShared(ctx, &dd, &hd, 64, 64, dev, (void **)&d_ctrl));
  if (!ok) return 1;

  std::mt19937 rng(999);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_int_distribution<int> qi(-100, 100);
  std::uniform_real_distribution<float> sc(0.001f, 0.05f);
  std::vector<float> hq((size_t)NQ * HD), hg((size_t)NQ * HD);
  std::vector<int8_t> hkc((size_t)NKV * MAXC * HD),
      hvc((size_t)NKV * MAXC * HD);
  std::vector<float> hks((size_t)MAXC * NKV), hvs((size_t)MAXC * NKV);
  for (size_t i = 0; i < hq.size(); ++i) {
    hq[i] = nd(rng);
    hg[i] = nd(rng);
  }
  for (size_t i = 0; i < hkc.size(); ++i) {
    hkc[i] = (int8_t)qi(rng);
    hvc[i] = (int8_t)qi(rng);
  }
  for (size_t i = 0; i < hks.size(); ++i) {
    hks[i] = sc(rng);
    hvs[i] = sc(rng);
  }
  uint32_t maxc = MAXC;
  d_ctrl[0] = 0;

  ze_command_list_handle_t cp = nullptr;
  CK(zeCommandListCreate(ctx, dev, &ld, &cp));
  CK(zeCommandListAppendMemoryCopy(cp, d_q, hq.data(), hq.size() * 4, nullptr,
                                   0, nullptr));
  CK(zeCommandListAppendMemoryCopy(cp, d_gate, hg.data(), hg.size() * 4,
                                   nullptr, 0, nullptr));
  CK(zeCommandListAppendMemoryCopy(cp, d_kc, hkc.data(), hkc.size(), nullptr,
                                   0, nullptr));
  CK(zeCommandListAppendMemoryCopy(cp, d_vc, hvc.data(), hvc.size(), nullptr,
                                   0, nullptr));
  CK(zeCommandListAppendMemoryCopy(cp, d_ks, hks.data(), hks.size() * 4,
                                   nullptr, 0, nullptr));
  CK(zeCommandListAppendMemoryCopy(cp, d_vs, hvs.data(), hvs.size() * 4,
                                   nullptr, 0, nullptr));
  CK(zeCommandListClose(cp));
  CK(zeCommandQueueExecuteCommandLists(queue, 1, &cp, nullptr));
  CK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  const int TS[] = {512, 2048, 4096, 6720, 8192};
  const int SS[] = {2, 4, 8};
  std::printf("%-8s %-4s | %-14s %-14s | %-12s %-12s\n", "T", "S",
              "mismatch", "worst-abs", "legacy-us", "split-us");
  for (int T : TS) {
    d_ctrl[1] = T - 1;
    CK(zeKernelSetArgumentValue(k_leg, 0, sizeof(void *), &d_o0));
    CK(zeKernelSetArgumentValue(k_leg, 1, sizeof(void *), &d_q));
    CK(zeKernelSetArgumentValue(k_leg, 2, sizeof(void *), &d_gate));
    CK(zeKernelSetArgumentValue(k_leg, 3, sizeof(void *), &d_kc));
    CK(zeKernelSetArgumentValue(k_leg, 4, sizeof(void *), &d_vc));
    CK(zeKernelSetArgumentValue(k_leg, 5, sizeof(void *), &d_ks));
    CK(zeKernelSetArgumentValue(k_leg, 6, sizeof(void *), &d_vs));
    CK(zeKernelSetArgumentValue(k_leg, 7, sizeof(void *), &d_ctrl));
    CK(zeKernelSetArgumentValue(k_leg, 8, sizeof(uint32_t), &maxc));
    CK(zeKernelSetGroupSize(k_leg, 256, 1, 1));
    ze_group_count_t g0{(uint32_t)NQ, 1, 1};
    ze_command_list_handle_t li = nullptr;
    CK(zeCommandListCreate(ctx, dev, &ld, &li));
    CK(zeCommandListAppendLaunchKernel(li, k_leg, &g0, nullptr, 0, nullptr));
    CK(zeCommandListClose(li));
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 5; ++r)
      CK(zeCommandQueueExecuteCommandLists(queue, 1, &li, nullptr));
    CK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    auto t1 = std::chrono::steady_clock::now();
    double leg_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / 5;
    CK(zeCommandListReset(li));

    std::vector<float> ho0((size_t)NQ * HD);
    ze_command_list_handle_t bk = nullptr;
    CK(zeCommandListCreate(ctx, dev, &ld, &bk));
    CK(zeCommandListAppendMemoryCopy(bk, ho0.data(), d_o0, ho0.size() * 4,
                                     nullptr, 0, nullptr));
    CK(zeCommandListClose(bk));
    CK(zeCommandQueueExecuteCommandLists(queue, 1, &bk, nullptr));
    CK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    zeCommandListDestroy(bk);

    for (int S : SS) {
      CK(zeKernelSetArgumentValue(k_sp, 0, sizeof(void *), &d_part));
      CK(zeKernelSetArgumentValue(k_sp, 1, sizeof(void *), &d_q));
      CK(zeKernelSetArgumentValue(k_sp, 2, sizeof(void *), &d_kc));
      CK(zeKernelSetArgumentValue(k_sp, 3, sizeof(void *), &d_vc));
      CK(zeKernelSetArgumentValue(k_sp, 4, sizeof(void *), &d_ks));
      CK(zeKernelSetArgumentValue(k_sp, 5, sizeof(void *), &d_vs));
      CK(zeKernelSetArgumentValue(k_sp, 6, sizeof(void *), &d_ctrl));
      CK(zeKernelSetArgumentValue(k_sp, 7, sizeof(uint32_t), &maxc));
      CK(zeKernelSetArgumentValue(k_sp, 8, sizeof(int), &S));
      CK(zeKernelSetGroupSize(k_sp, 256, 1, 1));
      ze_group_count_t gs{(uint32_t)(NQ * S), 1, 1};
      CK(zeKernelSetArgumentValue(k_cb, 0, sizeof(void *), &d_o1));
      CK(zeKernelSetArgumentValue(k_cb, 1, sizeof(void *), &d_gate));
      CK(zeKernelSetArgumentValue(k_cb, 2, sizeof(void *), &d_part));
      CK(zeKernelSetArgumentValue(k_cb, 3, sizeof(int), &S));
      CK(zeKernelSetGroupSize(k_cb, 256, 1, 1));
      ze_group_count_t gc{(uint32_t)NQ, 1, 1};
      CK(zeCommandListCreate(ctx, dev, &ld, &li));
      CK(zeCommandListAppendLaunchKernel(li, k_sp, &gs, nullptr, 0,
                                         nullptr));
      CK(zeCommandListAppendBarrier(li, nullptr, 0, nullptr));
      CK(zeCommandListAppendLaunchKernel(li, k_cb, &gc, nullptr, 0,
                                         nullptr));
      CK(zeCommandListClose(li));
      auto s0 = std::chrono::steady_clock::now();
      for (int r = 0; r < 5; ++r)
        CK(zeCommandQueueExecuteCommandLists(queue, 1, &li, nullptr));
      CK(zeCommandQueueSynchronize(queue, UINT64_MAX));
      auto s1 = std::chrono::steady_clock::now();
      double sp_us =
          std::chrono::duration<double, std::micro>(s1 - s0).count() / 5;
      CK(zeCommandListReset(li));

      std::vector<float> ho1((size_t)NQ * HD);
      CK(zeCommandListCreate(ctx, dev, &ld, &bk));
      CK(zeCommandListAppendMemoryCopy(bk, ho1.data(), d_o1, ho1.size() * 4,
                                       nullptr, 0, nullptr));
      CK(zeCommandListClose(bk));
      CK(zeCommandQueueExecuteCommandLists(queue, 1, &bk, nullptr));
      CK(zeCommandQueueSynchronize(queue, UINT64_MAX));
      zeCommandListDestroy(bk);

      size_t mm = 0;
      double w = 0;
      for (size_t i = 0; i < ho0.size(); ++i) {
        uint32_t a, b;
        std::memcpy(&a, &ho0[i], 4);
        std::memcpy(&b, &ho1[i], 4);
        if (a != b) {
          ++mm;
          w = std::max(w, (double)std::fabs(ho0[i] - ho1[i]));
        }
      }
      std::printf("%-8d %-4d | %-14zu %-14.3e | %-12.1f %-12.1f\n", T, S,
                  mm, w, leg_us, sp_us);
    }
  }
  return 0;
}
