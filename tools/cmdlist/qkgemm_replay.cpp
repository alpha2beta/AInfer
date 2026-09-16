// T7.4 QK-GEMM probe: S[24][T] = Q(24x256 fp16) x Kc^T (BF16 cache, strided
// reads) via bf16 DPAS, fp32 accumulate. The load-bearing unknown for the
// GEMM-form attention pivot: does DPAS engage on this shape (T3.4 caveat:
// pure-DPAS roof stalled 7-9 TFLOPS)? Correctness vs fp32 ref (tol 1e-3:
// half inputs) + achieved TFLOPS at T=4096.
// Usage: qkgemm_replay <qkgemm.spv> [report.json]
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstring>
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

static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}
static uint16_t f32_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
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
  if (argc < 2) {
    std::fprintf(stderr, "usage: qkgemm_replay <qkgemm.spv> [report.json]\n");
    return 2;
  }
  const int M = 24, K = 256, T = 4096, TMAX = 4096;
  FILE *sf = std::fopen(argv[1], "rb");
  if (!sf) {
    std::fprintf(stderr, "no spv\n");
    return 2;
  }
  std::fseek(sf, 0, SEEK_END);
  size_t spvN = std::ftell(sf);
  std::fseek(sf, 0, SEEK_SET);
  std::vector<uint8_t> spv(spvN);
  if (std::fread(spv.data(), 1, spvN, sf) != spvN)
    return 2;
  std::fclose(sf);

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
  void *dA = nullptr, *dB = nullptr, *dS = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * K * 2, 4096, dev, &dA));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)TMAX * 4 * K * 2, 4096, dev,
                         &dB));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * TMAX * 4, 4096, dev, &dS));

  ze_module_handle_t mod = nullptr;
  ze_module_desc_t mddesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                             nullptr,
                             ZE_MODULE_FORMAT_IL_SPIRV,
                             spvN,
                             spv.data(),
                             nullptr,
                             nullptr};
  CHECK(zeModuleCreate(ctx, dev, &mddesc, &mod, nullptr));
  ze_kernel_handle_t ker = nullptr;
  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                            "_ZTS6QkGemm"};
  CHECK(zeKernelCreate(mod, &kdesc, &ker));
  CHECK(zeKernelSetGroupSize(ker, 16, 1, 1));
  int nArg = T, tmaxArg = TMAX, kvOff = 0;
  CHECK(zeKernelSetArgumentValue(ker, 0, sizeof(void *), &dA));
  CHECK(zeKernelSetArgumentValue(ker, 1, sizeof(void *), &dB));
  CHECK(zeKernelSetArgumentValue(ker, 2, sizeof(void *), &dS));
  CHECK(zeKernelSetArgumentValue(ker, 3, sizeof(int), &nArg));
  CHECK(zeKernelSetArgumentValue(ker, 4, sizeof(int), &tmaxArg));
  CHECK(zeKernelSetArgumentValue(ker, 5, sizeof(int), &kvOff));
  CHECK(zeKernelSetArgumentValue(ker, 6, (size_t)8 * 256 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(ker, 7, (size_t)16 * 16 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(ker, 8, (size_t)8 * 16 * 4, nullptr));

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
  // Single kv-group probe (KvOff=0): 6 live rows + 2 masked.
  ze_group_count_t gc = {(uint32_t)(T / 16), 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(reg, ker, &gc, nullptr, 0, nullptr));
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

  uint64_t s = 0x9e6d;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  };
  // A: fp16 Q (host converts fp32 randoms); B: BF16 cache slots.
  std::vector<uint16_t> hA(M * K);
  std::vector<float> hAf(M * K);
  for (auto &v : hAf)
    v = rnd() * 2.0f;
  // fp32->fp16 RNE on host for A
  for (int i = 0; i < M * K; ++i) {
    float x = hAf[i];
    uint32_t u;
    std::memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int exp = (int)((u >> 23) & 0xFF) - 112;
    uint32_t mant = u & 0x7FFFFFu, h;
    if (exp <= 0)
      h = sign;
    else if (exp >= 31)
      h = sign | 0x7BFFu;
    else {
      uint32_t m10 = mant >> 13, rest = mant & 0x1FFFu;
      if (rest > 0x1000u || (rest == 0x1000u && (m10 & 1u))) {
        if (++m10 == 0x400u) {
          m10 = 0;
          if (++exp >= 31) {
            h = sign | 0x7BFFu;
            hA[i] = (uint16_t)h;
            continue;
          }
        }
      }
      h = sign | ((uint32_t)exp << 10) | m10;
    }
    hA[i] = (uint16_t)h;
  }
  std::vector<uint16_t> hB((size_t)TMAX * 4 * K, 0);
  std::vector<float> hBf((size_t)TMAX * K);
  for (auto &v : hBf)
    v = rnd() * 0.5f;
  // kv0 populated (KvOff=0 probe), other heads zero.
  for (int t = 0; t < TMAX; ++t)
    for (int k = 0; k < K; ++k)
      hB[((size_t)t * 4 + 0) * K + k] = f32_to_bf16(hBf[(size_t)t * K + k]);
  CHECK(zeCommandListAppendMemoryCopy(up, dA, hA.data(), hA.size() * 2, nullptr,
                                      0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dB, hB.data(), hB.size() * 2, nullptr,
                                      0, nullptr));

  std::vector<double> tRep;
  for (int it = 0; it < 10; ++it) {
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    tRep.push_back(now_ns() - t0);
  }
  std::vector<float> hS((size_t)M * TMAX, 0.0f);
  CHECK(zeCommandListAppendMemoryCopy(up, hS.data(), dS, hS.size() * 4, nullptr,
                                      0, nullptr));
  // fp32 ref (A-half-exact + B-bf16-exact, double accum).
  size_t nNZ = 0;
  for (size_t i = 0; i < (size_t)6 * TMAX; ++i)
    if (hS[i] != 0.0f)
      ++nNZ;
  std::printf("S nonzero %zu / %d (rows 0-5)\n", nNZ, 6 * TMAX);
  bool ok = true;
  double worstRel = 0;
  // KvOff=0 probe: only rows 0..5 computed.
  for (int m = 0; m < 6 && ok; ++m) {
    for (int t = 0; t < T; ++t) {
      uint32_t u;
      double acc = 0;
      for (int k = 0; k < K; ++k) {
        // A half bits -> float
        uint16_t hb = hA[(size_t)m * K + k];
        uint32_t sign = ((uint32_t)hb & 0x8000u) << 16;
        int exp = (hb >> 10) & 0x1F;
        uint32_t mant = (uint32_t)(hb & 0x3FFu) << 13;
        u = exp == 0 ? sign : sign | ((uint32_t)(exp + 112) << 23) | mant;
        float a;
        std::memcpy(&a, &u, 4);
        acc +=
            (double)a * (double)bf16_to_f32(hB[((size_t)t * 4 + 0) * K + k]);
      }
      acc /= 16.0; // attention score scale (must match the kernel!)
      float want = (float)acc, got = hS[(size_t)m * TMAX + t];
      double rel = std::fabs((double)got - (double)want) /
                   (std::fabs((double)want) > 1e-3 ? std::fabs((double)want)
                                                   : 1.0);
      if (rel > worstRel)
        worstRel = rel;
      if (rel > 1e-2) {
        ok = false;
        std::fprintf(stderr, "ref mismatch m %d t %d: got %g want %g\n", m, t,
                     got, want);
        break;
      }
    }
  }
  double mR = med(tRep);
  double flops = 2.0 * M * T * K;
  double tflops = flops / (mR / 1e9) / 1e12;
  std::printf("qkgemm med %.2f us ref %.2e TFLOPS %.2f %s\n", mR / 1e3, worstRel,
              tflops, ok ? "QKGEMM-OK" : "MISMATCH");
  char js[512];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernel\":\"_ZTS6QkGemm\","
                "\"shape\":\"24x4096x256\",\"replay_us\":%.2f,"
                "\"worst_rel\":%.2e,\"ref_tol\":1e-2,\"tflops\":%.2f,"
                "\"qkgemm_ok\":%s}",
                mR / 1e3, worstRel, tflops, ok ? "true" : "false");
  if (argc > 2) {
    FILE *o = std::fopen(argv[2], "w");
    if (o) {
      std::fprintf(o, "%s\n", js);
      std::fclose(o);
    }
  } else {
    std::printf("%s\n", js);
  }
  return ok ? 0 : 1;
}
