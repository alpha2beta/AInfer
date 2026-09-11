// T7.4 chunked-prefill risk retirement: record ONCE a closed regular L0 list
// launching the tiled INT4 DPAS GEMM (ChunkGemm, MT4 design: 1 SG16 per 8x16
// tile, M tiled x4) at chunk shape 256x5120x17408 (gate projection), replay
// 10x over varying fp16 activation chunks (weights/scales fixed). Checks:
// tight-tolerance match vs double-precision host ref with depacked weights +
// bitwise replay determinism. L0 launch: (M/32)*(TN/16) groups x 16 WIs,
// reqd SG16, 3 SLM args by size.
// Usage: chunkgemm_replay <chunkgemm.spv> [report.json]
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

static double now_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
static double med(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static uint16_t f32_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}
static uint16_t f32_to_f16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  uint32_t sign = (u >> 16) & 0x8000u;
  int exp = (int)((u >> 23) & 0xFF) - 112;
  uint32_t mant = u & 0x7FFFFFu;
  if (exp <= 0)
    return (uint16_t)sign;
  if (exp >= 31)
    return (uint16_t)(sign | 0x7BFFu);
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}
static float f16_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  int exp = ((h >> 10) & 0x1F);
  uint32_t mant = (uint32_t)(h & 0x3FFu) << 13;
  uint32_t u = exp == 0 ? sign : sign | ((uint32_t)(exp + 112) << 23) | mant;
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: chunkgemm_replay <chunkgemm.spv> [report]\n");
    return 2;
  }
  const int M = 256, K = 5120, N = 17408, GG = K / 128, IT = 10;
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
  void *dA = nullptr, *dP = nullptr, *dS = nullptr, *dC = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * K * 2, 4096, dev, &dA));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)N * K / 2, 4096, dev, &dP));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)N * GG * 2, 4096, dev, &dS));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * N * 4, 4096, dev, &dC));

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
                            "_ZTS9ChunkGemm"};
  if (zeKernelCreate(mod, &kdesc, &ker) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "kernel _ZTS9ChunkGemm not found\n");
    return 1;
  }
  CHECK(zeKernelSetGroupSize(ker, 16, 1, 1));
  int mA = M, kA = K, nA = N;
  CHECK(zeKernelSetArgumentValue(ker, 0, sizeof(void *), &dA));
  CHECK(zeKernelSetArgumentValue(ker, 1, sizeof(void *), &dP));
  CHECK(zeKernelSetArgumentValue(ker, 2, sizeof(void *), &dS));
  CHECK(zeKernelSetArgumentValue(ker, 3, sizeof(void *), &dC));
  CHECK(zeKernelSetArgumentValue(ker, 4, sizeof(int), &mA));
  CHECK(zeKernelSetArgumentValue(ker, 5, sizeof(int), &kA));
  CHECK(zeKernelSetArgumentValue(ker, 6, sizeof(int), &nA));
  CHECK(zeKernelSetArgumentValue(ker, 7, (size_t)512 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(ker, 8, (size_t)256 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(ker, 9, (size_t)512 * 4, nullptr));

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
  ze_group_count_t gc = {(uint32_t)((M / 32) * (N / 16)), 1, 1};
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

  // Synthetic weights, layout-0 INT4 + BF16 scales (gate shape).
  uint64_t s = 0xC14C4C;
  std::vector<uint8_t> packed((size_t)N * K / 2, 0);
  std::vector<uint16_t> scales((size_t)N * GG);
  std::vector<std::vector<float>> wfull(N, std::vector<float>(K));
  s = 0x9E3779B9;
  for (int m = 0; m < N; ++m)
    for (int g = 0; g < GG; ++g) {
      float amax = 0;
      for (int j = 0; j < 128; ++j) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        float w =
            (float)((int)((s >> 33) & 0xFFFF) - 32768) * (0.02f / 32768.0f);
        wfull[m][g * 128 + j] = w;
        amax = std::max(amax, std::fabs(w));
      }
      float sc = amax == 0 ? 1.0f : amax / 7.0f;
      scales[(size_t)m * GG + g] = f32_to_bf16(sc);
      for (int j = 0; j < 128; ++j) {
        int q = (int)lrintf(wfull[m][g * 128 + j] / sc);
        q = q < -8 ? -8 : (q > 7 ? 7 : q);
        size_t idx = (size_t)m * K + g * 128 + j;
        if (j & 1)
          packed[idx / 2] |= (uint8_t)((q & 0xF) << 4);
        else
          packed[idx / 2] = (uint8_t)(q & 0xF);
      }
    }
  CHECK(zeCommandListAppendMemoryCopy(up, dP, packed.data(), packed.size(),
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dS, scales.data(),
                                      scales.size() * 2, nullptr, 0, nullptr));

  std::vector<uint16_t> hA(M * K);
  std::vector<float> hC(M * N), hC2(M * N), hRef(M * N);
  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0;
  for (int it = 0; it < IT; ++it) {
    for (size_t i = 0; i < hA.size(); ++i) {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      float v =
          (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
      hA[i] = f32_to_f16(v);
    }
    // Double-precision host ref with depacked weights.
    for (int m = 0; m < M; ++m)
      for (int nn = 0; nn < N; ++nn) {
        double acc = 0;
        for (int g = 0; g < GG; ++g) {
          float sc = bf16_to_f32(scales[(size_t)nn * GG + g]);
          for (int j = 0; j < 128; ++j) {
            size_t idx = (size_t)nn * K + g * 128 + j;
            uint8_t b = packed[idx / 2];
            int nib = (idx & 1) ? (b >> 4) : (b & 0xF);
            if (nib >= 8)
              nib -= 16;
            acc += (double)f16_to_f32(hA[(size_t)m * K + g * 128 + j]) *
                   (double)((float)nib * sc);
          }
        }
        hRef[(size_t)m * N + nn] = (float)acc;
      }
    CHECK(zeCommandListAppendMemoryCopy(up, dA, hA.data(), hA.size() * 2,
                                        nullptr, 0, nullptr));
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    tRep.push_back(now_ns() - t0);
    CHECK(zeCommandListAppendMemoryCopy(up, hC.data(), dC, hC.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    CHECK(zeCommandListAppendMemoryCopy(up, hC2.data(), dC, hC2.size() * 4,
                                        nullptr, 0, nullptr));
    double refmax = 0;
    for (float v : hRef)
      refmax = std::max(refmax, (double)std::fabs(v));
    for (size_t i = 0; i < hRef.size(); ++i) {
      double rel = std::fabs((double)hC[i] - (double)hRef[i]) / refmax;
      if (rel > worstRel)
        worstRel = rel;
      if (rel > 1e-3) {
        ok = false;
        std::fprintf(stderr, "ref mismatch it %d i %zu: got %g want %g\n", it,
                     i, hC[i], hRef[i]);
        break;
      }
      uint32_t a, b;
      __builtin_memcpy(&a, &hC[i], 4);
      __builtin_memcpy(&b, &hC2[i], 4);
      if (a != b) {
        ok = false;
        std::fprintf(stderr, "nondeterministic replay it %d i %zu\n", it, i);
        break;
      }
    }
    if (!ok)
      break;
  }
  double mR = med(tRep);
  double tflops =
      (2.0 * M * N * K / (mR / 1e9)) / 1e12;
  std::printf("chunkgemm med %.2f us/iter worst-rel %.2e %.2f TFLOPS %s\n",
              mR / 1e3, worstRel, tflops, ok ? "CHUNK-OK" : "MISMATCH");
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernel\":\"_ZTS9ChunkGemm\","
                "\"shape\":[%d,%d,%d],\"iters\":%d,\"replay_us\":%.2f,"
                "\"tflops\":%.2f,\"worst_rel\":%.2e,\"ref_tol\":1e-3,"
                "\"bitwise_deterministic\":%s,\"chunk_ok\":%s}",
                M, K, N, IT, mR / 1e3, tflops, worstRel,
                ok ? "true" : "false", ok ? "true" : "false");
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
