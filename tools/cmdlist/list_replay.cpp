// T5.3 layer-class prototype: record ONCE a closed regular L0 list of three
// heterogeneous decode-exact kernels — RMSNormW(1+w) -> SiluMul -> ResAdd
// (MLP-tail subgraph shape; GEMVs stay SYCL until the raw-L0 GEMV port) —
// with fixed argument addresses, then replay 50x over varying inputs AND a
// varying control-block position (explicit 16 B update, T5.4 policy). Checks:
// bitwise replay determinism (same input twice -> identical bits) and host-ref
// accuracy (tolerance: norm's rsqrt differs in the last ulps across ISAs).
// Usage: list_replay <norm.spv> <silu.spv> <res.spv> [report.json]
// (one SPIR-V module per kernel entry; see extract_spv.py)
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <string>
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

struct DecodeControl {
  int token_id, position, active_length, selected_token;
};

static double now_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
static double med(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: list_replay <norm.spv> <silu.spv> <res.spv> "
                 "[report.json]\n");
    return 2;
  }
  const int H = 5120, I = 17408, GS = 256, IT = 50;
  auto load_spv = [&](const char *path) {
    FILE *sf = std::fopen(path, "rb");
    if (!sf) {
      std::fprintf(stderr, "no spv: %s\n", path);
      std::exit(2);
    }
    std::fseek(sf, 0, SEEK_END);
    size_t n = std::ftell(sf);
    std::fseek(sf, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (std::fread(v.data(), 1, n, sf) != n)
      std::exit(2);
    std::fclose(sf);
    return v;
  };
  std::vector<uint8_t> spvN = load_spv(argv[1]);
  std::vector<uint8_t> spvS = load_spv(argv[2]);
  std::vector<uint8_t> spvR = load_spv(argv[3]);

  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t nDrv = 0;
  CHECK(zeDriverGet(&nDrv, nullptr));
  std::vector<ze_driver_handle_t> drvs(nDrv);
  CHECK(zeDriverGet(&nDrv, drvs.data()));
  ze_device_handle_t dev = nullptr;
  ze_driver_handle_t drv = nullptr;
  for (auto d : drvs) {
    uint32_t nv = 0;
    if (zeDeviceGet(d, &nv, nullptr) != ZE_RESULT_SUCCESS)
      continue;
    std::vector<ze_device_handle_t> vs(nv);
    zeDeviceGet(d, &nv, vs.data());
    for (auto v : vs) {
      ze_device_properties_t pr = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      if (zeDeviceGetProperties(v, &pr) == ZE_RESULT_SUCCESS &&
          pr.vendorId == 0x8086 && pr.deviceId == 0xe211) {
        dev = v;
        drv = d;
      }
    }
  }
  if (!dev)
    return 1;
  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(drv, &cdesc, &ctx));
  ze_device_mem_alloc_desc_t mdesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  void *dX = nullptr, *dW = nullptr, *dY = nullptr, *dR = nullptr;
  void *dG = nullptr, *dU = nullptr, *dZ = nullptr, *dCtrl = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)H * 4, 4096, dev, &dX));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)H * 4, 4096, dev, &dW));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)H * 4, 4096, dev, &dY));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)H * 4, 4096, dev, &dR));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)I * 4, 4096, dev, &dG));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)I * 4, 4096, dev, &dU));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)I * 4, 4096, dev, &dZ));
  CHECK(zeMemAllocDevice(ctx, &mdesc, sizeof(DecodeControl), 4096, dev,
                         &dCtrl));

  auto mkmod = [&](const std::vector<uint8_t> &spv) -> ze_module_handle_t {
    ze_module_handle_t mod = nullptr;
    ze_module_desc_t mddesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                               nullptr,
                               ZE_MODULE_FORMAT_IL_SPIRV,
                               spv.size(),
                               spv.data(),
                               nullptr,
                               nullptr};
    if (zeModuleCreate(ctx, dev, &mddesc, &mod, nullptr) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "L0 module create failed\n");
      std::exit(1);
    }
    return mod;
  };
  ze_module_handle_t modN = mkmod(spvN), modS = mkmod(spvS),
                     modR = mkmod(spvR);
  ze_kernel_handle_t kNorm = nullptr, kSilu = nullptr, kRes = nullptr;
  ze_kernel_desc_t kd1 = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                          "_ZTS8RMSNormW"};
  ze_kernel_desc_t kd2 = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                          "_ZTS7SiluMul"};
  ze_kernel_desc_t kd3 = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                          "_ZTS6ResAdd"};
  if (zeKernelCreate(modN, &kd1, &kNorm) != ZE_RESULT_SUCCESS ||
      zeKernelCreate(modS, &kd2, &kSilu) != ZE_RESULT_SUCCESS ||
      zeKernelCreate(modR, &kd3, &kRes) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "layer-tail entry point not found\n");
    return 1;
  }
  CHECK(zeKernelSetGroupSize(kNorm, 256, 1, 1));
  CHECK(zeKernelSetGroupSize(kSilu, GS, 1, 1));
  CHECK(zeKernelSetGroupSize(kRes, GS, 1, 1));
  int nH = H, nI = I;
  CHECK(zeKernelSetArgumentValue(kNorm, 0, sizeof(void *), &dY));
  CHECK(zeKernelSetArgumentValue(kNorm, 1, sizeof(void *), &dX));
  CHECK(zeKernelSetArgumentValue(kNorm, 2, sizeof(void *), &dW));
  CHECK(zeKernelSetArgumentValue(kNorm, 3, sizeof(int), &nH));
  CHECK(zeKernelSetArgumentValue(kNorm, 4, (size_t)256 * 8, nullptr));
  CHECK(zeKernelSetArgumentValue(kSilu, 0, sizeof(void *), &dG));
  CHECK(zeKernelSetArgumentValue(kSilu, 1, sizeof(void *), &dU));
  CHECK(zeKernelSetArgumentValue(kSilu, 2, sizeof(void *), &dZ));
  // Residual consumes the NORMED vector (fixed address dY): the recorded list
  // is a live chain norm -> silu-mul -> residual, no dead kernels.
  CHECK(zeKernelSetArgumentValue(kRes, 0, sizeof(void *), &dR));
  CHECK(zeKernelSetArgumentValue(kRes, 1, sizeof(void *), &dY));
  CHECK(zeKernelSetArgumentValue(kRes, 2, sizeof(void *), &dZ));
  CHECK(zeKernelSetArgumentValue(kRes, 3, sizeof(void *), &dCtrl));
  CHECK(zeKernelSetArgumentValue(kRes, 4, sizeof(int), &nH));
  (void)nI;

  ze_command_queue_handle_t qq = nullptr;
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandQueueCreate(ctx, dev, &qdesc, &qq));
  ze_command_list_handle_t reg = nullptr;
  ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                                  nullptr, 0, 0};
  CHECK(zeCommandListCreate(ctx, dev, &ldesc, &reg));
  ze_group_count_t g1 = {1, 1, 1};
  ze_group_count_t gS = {(uint32_t)(I / GS), 1, 1};
  ze_group_count_t gR = {(uint32_t)(H / GS), 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(reg, kNorm, &g1, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(reg, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendLaunchKernel(reg, kSilu, &gS, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(reg, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendLaunchKernel(reg, kRes, &gR, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(reg, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(reg));
  ze_fence_handle_t fence = nullptr;
  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK(zeFenceCreate(qq, &fdesc, &fence));

  ze_command_list_handle_t up = nullptr;
  ze_command_queue_desc_t idesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &idesc, &up));

  std::vector<float> hX(H), hW(H), hY(H), hG(I), hU(I), hR(H), hR2(H);
  uint64_t s = 0x71A17;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  };
  for (auto &v : hW)
    v = rnd() * 0.1f;
  CHECK(zeCommandListAppendMemoryCopy(up, dW, hW.data(), (size_t)H * 4,
                                      nullptr, 0, nullptr));

  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0, worstAbs = 0;
  for (int it = 0; it < IT; ++it) {
    for (auto &v : hX)
      v = rnd() * 2.0f;
    for (auto &v : hG)
      v = rnd() * 2.0f;
    for (auto &v : hU)
      v = rnd() * 2.0f;
    int pos = (it * 53 + 7) % 128;
    DecodeControl c{5000 + it, pos, pos + 1, -1};
    // host reference (double accumulator, mirrors decode's rmsnorm)
    double ss = 0;
    for (int j = 0; j < H; ++j)
      ss += (double)hX[j] * hX[j];
    float inv = 1.0f / std::sqrt((float)(ss / H) + 1e-6f);
    for (int j = 0; j < H; ++j)
      hY[j] = hX[j] * inv * (1.0f + hW[j]);
    CHECK(zeCommandListAppendMemoryCopy(up, dX, hX.data(), (size_t)H * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dG, hG.data(), (size_t)I * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dU, hU.data(), (size_t)I * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                        nullptr));
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    tRep.push_back(now_ns() - t0);
    CHECK(zeCommandListAppendMemoryCopy(up, hR.data(), dR, (size_t)H * 4,
                                        nullptr, 0, nullptr));
    // bitwise determinism: replay same state again, expect identical bits
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    CHECK(zeCommandListAppendMemoryCopy(up, hR2.data(), dR, (size_t)H * 4,
                                        nullptr, 0, nullptr));
    for (int j = 0; j < H; ++j) {
      uint32_t a, b;
      __builtin_memcpy(&a, &hR[j], 4);
      __builtin_memcpy(&b, &hR2[j], 4);
      if (a != b) {
        ok = false;
        std::fprintf(stderr, "nondeterministic replay it %d j %d\n", it, j);
        break;
      }
      float g = hG[j % H], u = hU[j % H];
      float z = (g / (1.0f + expf(-g))) * u;
      float want = hY[j] + z + (float)pos;
      // Host ref mirrors the device chain (norm -> silu -> res). Drift
      // sources: norm rsqrt last-ulp across ISAs, single-precision exp —
      // a few ulps on O(1) components. Near-cancellation outputs (want~0)
      // make pure relative error meaningless, so the criterion is mixed:
      // rel<=1e-5 OR abs<=2e-6. Bitwise determinism (above) remains the
      // strict recording property.
      double abs = std::fabs((double)hR[j] - (double)want);
      double denom = std::fabs(want) > 1e-6 ? std::fabs(want) : 1e-6;
      double rel = abs / denom;
      if (rel > worstRel)
        worstRel = rel;
      if (abs > worstAbs)
        worstAbs = abs;
      if (rel > 1e-5 && abs > 2e-6) {
        ok = false;
        std::fprintf(stderr, "mismatch it %d j %d: got %g want %g\n", it, j,
                     hR[j], want);
        break;
      }
    }
    if (!ok)
      break;
  }
  double mR = med(tRep);
  std::printf("list-replay med %.2f us/iter worst-rel %.2e worst-abs %.2e %s\n",
              mR / 1e3, worstRel, worstAbs, ok ? "LIST-OK" : "MISMATCH");
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernels\":[\"_ZTS8RMSNormW\","
                "\"_ZTS7SiluMul\",\"_ZTS6ResAdd\"],\"H\":%d,\"I\":%d,"
                "\"iters\":%d,\"replay_us\":%.2f,\"worst_rel\":%.2e,"
                "\"worst_abs\":%.2e,\"bitwise_deterministic\":%s,"
                "\"list_ok\":%s}",
                H, I, IT, mR / 1e3, worstRel, worstAbs,
                ok ? "true" : "false", ok ? "true" : "false");
  if (argc > 4) {
    FILE *o = std::fopen(argv[4], "w");
    if (o) {
      std::fprintf(o, "%s\n", js);
      std::fclose(o);
    }
  } else {
    std::printf("%s\n", js);
  }
  return ok ? 0 : 1;
}
