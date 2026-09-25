// Unit: bitwise parity across attention decode variants + LM-head variants.
// Any nonzero bit mismatch between kernels that must agree per-token proves
// the FP-order divergence behind long-context verify-vs-greedy DIFF.
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

static int fails = 0;
static void cmp_bits(const char *tag, const std::vector<float> &a,
                     const std::vector<float> &b) {
  size_t n = a.size(), mm = 0;
  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    uint32_t x, y;
    std::memcpy(&x, &a[i], 4);
    std::memcpy(&y, &b[i], 4);
    if (x != y) {
      ++mm;
      worst = std::max(worst, (double)std::fabs(a[i] - b[i]));
    }
  }
  std::printf("%-28s: mismatch %zu/%zu worst=%.3e %s\n", tag, mm, n, worst,
              mm == 0 ? "BITEXACT" : "DIVERGE");
  if (mm) ++fails;
}

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1]
                                    : "tools/kernels_258v/all_kernels.spv";
  const int HD = 256, NQ = 16, NKV = 2, T = 2048, MAXC = 2048;

  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t nDrv = 0;
  CHECK(zeDriverGet(&nDrv, nullptr));
  std::vector<ze_driver_handle_t> drvs(nDrv);
  CHECK(zeDriverGet(&nDrv, drvs.data()));
  uint32_t nDev = 0;
  CHECK(zeDeviceGet(drvs[0], &nDev, nullptr));
  std::vector<ze_device_handle_t> devs(nDev);
  CHECK(zeDeviceGet(drvs[0], &nDev, devs.data()));
  ze_device_handle_t dev = devs[0];
  ze_context_desc_t ctxd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t ctx = nullptr;
  CHECK(zeContextCreate(drvs[0], &ctxd, &ctx));
  std::vector<uint8_t> spv = read_file(spv_path);
  ze_module_desc_t md = {ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
                         ZE_MODULE_FORMAT_IL_SPIRV, spv.size(), spv.data(),
                         nullptr, nullptr};
  ze_module_handle_t mod = nullptr;
  CHECK(zeModuleCreate(ctx, dev, &md, &mod, nullptr));
  ze_kernel_handle_t k_dec = nullptr, k_v2 = nullptr, k_fl = nullptr;
  {
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           "gqa_attn_decode_ctrl"};
    CHECK(zeKernelCreate(mod, &kd, &k_dec));
  }
  {
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           "gqa_attn_prefill_batch_v2"};
    CHECK(zeKernelCreate(mod, &kd, &k_v2));
  }
  {
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           "flash_attn_prefill_b8_t16"};
    CHECK(zeKernelCreate(mod, &kd, &k_fl));
  }

  ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                nullptr, 0, 0, 0,
                                ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue = nullptr;
  CHECK(zeCommandQueueCreate(ctx, dev, &qd, &queue));
  ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                               nullptr, 0, 0};
  ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                   nullptr, 0, 0};
  float *d_q = nullptr, *d_gate = nullptr, *d_o1 = nullptr, *d_o2 = nullptr,
        *d_o3 = nullptr;
  uint16_t *d_kc = nullptr, *d_vc = nullptr;
  int32_t *d_ctrl = nullptr;
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)NQ * HD * 4, 64, dev,
                         (void **)&d_q));
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)NQ * HD * 4, 64, dev,
                         (void **)&d_gate));
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)NQ * HD * 4, 64, dev,
                         (void **)&d_o1));
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)NQ * HD * 4, 64, dev,
                         (void **)&d_o2));
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)NQ * HD * 4, 64, dev,
                         (void **)&d_o3));
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)NKV * MAXC * HD * 2, 64, dev,
                         (void **)&d_kc));
  CHECK(zeMemAllocDevice(ctx, &dd, (size_t)NKV * MAXC * HD * 2, 64, dev,
                         (void **)&d_vc));
  ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                 nullptr, 0};
  CHECK(zeMemAllocShared(ctx, &dd, &hd, 64, 64, dev, (void **)&d_ctrl));

  // Random Q/Gate/KV with a repetitive pattern (mimics the 6.7K prompt that
  // produces near-tie attention mass) + outliers.
  std::mt19937 rng(777);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> hq((size_t)NQ * HD), hg((size_t)NQ * HD);
  std::vector<uint16_t> hkc((size_t)NKV * MAXC * HD),
      hvc((size_t)NKV * MAXC * HD);
  for (size_t i = 0; i < hq.size(); ++i) {
    hq[i] = nd(rng);
    hg[i] = nd(rng);
  }
  for (int t = 0; t < MAXC; ++t)
    for (int h = 0; h < NKV; ++h)
      for (int d = 0; d < HD; ++d) {
        float kv = nd(rng) + ((t % 16 == 0) ? 3.0f : 0.0f);
        hkc[((size_t)h * MAXC + t) * HD + d] = f32_to_bf16(kv);
        hvc[((size_t)h * MAXC + t) * HD + d] = f32_to_bf16(nd(rng));
      }
  d_ctrl[0] = 0;
  d_ctrl[1] = T - 1; // decode position: attend 0..T-1
  uint32_t maxc = MAXC;
  int B = 1;

  ze_command_list_handle_t cp = nullptr, li = nullptr;
  CHECK(zeCommandListCreate(ctx, dev, &ld, &cp));
  CHECK(zeCommandListAppendMemoryCopy(cp, d_q, hq.data(), hq.size() * 4,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(cp, d_gate, hg.data(), hg.size() * 4,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(cp, d_kc, hkc.data(), hkc.size() * 2,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(cp, d_vc, hvc.data(), hvc.size() * 2,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListClose(cp));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cp, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  int l0_ok = 1;
#define CK(expr)                                                               \
  do {                                                                         \
    ze_result_t _r = (expr);                                                   \
    if (_r != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "L0 error %d at %s:%d (%s)\n", (int)_r, __FILE__,   \
                   __LINE__, #expr);                                          \
      l0_ok = 0;                                                               \
    }                                                                          \
  } while (0)
  auto run_attn = [&](ze_kernel_handle_t k, float *d_o, bool is_flash) {
    CK(zeKernelSetArgumentValue(k, 0, sizeof(void *), &d_o));
    CK(zeKernelSetArgumentValue(k, 1, sizeof(void *), &d_q));
    CK(zeKernelSetArgumentValue(k, 2, sizeof(void *), &d_gate));
    CK(zeKernelSetArgumentValue(k, 3, sizeof(void *), &d_kc));
    CK(zeKernelSetArgumentValue(k, 4, sizeof(void *), &d_vc));
    CK(zeKernelSetArgumentValue(k, 5, sizeof(void *), &d_ctrl));
    CK(zeKernelSetArgumentValue(k, 6, sizeof(uint32_t), &maxc));
    if (k != k_dec) {
      CK(zeKernelSetArgumentValue(k, 7, sizeof(int), &B));
    }
    CK(zeKernelSetGroupSize(k, 256, 1, 1));
    ze_group_count_t gc;
    if (is_flash) {
      gc = {(uint32_t)NQ, 1, 1}; // B=1 -> 1 b-block
    } else if (k == k_dec) {
      gc = {(uint32_t)NQ, 1, 1};
    } else {
      gc = {(uint32_t)(B * NQ), 1, 1};
    }
    CK(zeCommandListCreate(ctx, dev, &ld, &li));
    CK(zeCommandListAppendLaunchKernel(li, k, &gc, nullptr, 0, nullptr));
    CK(zeCommandListClose(li));
    CK(zeCommandQueueExecuteCommandLists(queue, 1, &li, nullptr));
    CK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    CK(zeCommandListReset(li));
  };
  run_attn(k_dec, d_o1, false);
  run_attn(k_v2, d_o2, false);
  run_attn(k_fl, d_o3, true);
  if (!l0_ok) return 1;

  std::vector<float> ho1((size_t)NQ * HD), ho2((size_t)NQ * HD),
      ho3((size_t)NQ * HD);
  ze_command_list_handle_t bk = nullptr;
  CHECK(zeCommandListCreate(ctx, dev, &ld, &bk));
  CHECK(zeCommandListAppendMemoryCopy(bk, ho1.data(), d_o1, ho1.size() * 4,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(bk, ho2.data(), d_o2, ho2.size() * 4,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(bk, ho3.data(), d_o3, ho3.size() * 4,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListClose(bk));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &bk, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  cmp_bits("attn decode-ctrl vs prefill-v2", ho1, ho2);
  cmp_bits("attn decode-ctrl vs flash-b8t16", ho1, ho3);
  cmp_bits("attn prefill-v2 vs flash-b8t16", ho2, ho3);
  std::printf(fails == 0 ? "ATTN-ALL-BITEXACT\n" : "ATTN-DIVERGE\n");
  return fails == 0 ? 0 : 2;
}
