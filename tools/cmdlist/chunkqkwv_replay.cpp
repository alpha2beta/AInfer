// T7.4 chunked prefill: GEMM-form chunk attention probe (ChunkQkGemm with
// causal mask -> ChunkSoftmaxRow -> CvtF32F16 -> ChunkWvGemm) as staged
// recorded lists, vs full host ref (fp16-Q x qb-cache, causal, /16).
// M=32 chunk rows, P=64 prefix, W=96 width. One geometry (baked grid).
// Usage: chunkqkwv_replay <chunkqkgemm.spv> <chunksoftmaxrow.spv> <cvtf32f16.spv> <chunkwvgemm.spv> [report.json]
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

struct DecodeControl {
  int token_id, position, active_length, selected_token;
};

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
static inline float qb(float v) { return bf16_to_f32(f32_to_bf16(v)); }
static uint16_t f32_to_f16(float x) {
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
          return (uint16_t)h;
        }
      }
    }
    h = sign | ((uint32_t)exp << 10) | m10;
  }
  return (uint16_t)h;
}
static float f16_to_f32(uint16_t hb) {
  uint32_t sign = ((uint32_t)hb & 0x8000u) << 16;
  int exp = (hb >> 10) & 0x1F;
  uint32_t mant = (uint32_t)(hb & 0x3FFu) << 13;
  uint32_t u = exp == 0 ? sign : sign | ((uint32_t)(exp + 112) << 23) | mant;
  float a;
  std::memcpy(&a, &u, 4);
  return a;
}
static double now_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: chunkqkwv_replay <cqk> <csm> <cvt> <cwv> "
                         "[report.json]\n");
    return 2;
  }
  const int NH = 24, NKV = 4, D = 256, M = 32, P = 64, W = P + M;
  const int ROWS = M * NH;
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
  std::vector<uint8_t> spvQK = load_spv(argv[1]), spvSM = load_spv(argv[2]),
                       spvCV = load_spv(argv[3]), spvWV = load_spv(argv[4]);

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
  void *dQ = nullptr, *dKc = nullptr, *dVc = nullptr, *dS = nullptr,
       *dW = nullptr, *dWh = nullptr, *dO = nullptr, *dCtrl = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)ROWS * D * 2, 4096, dev, &dQ));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)W * NKV * D * 2, 4096, dev,
                         &dKc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)W * NKV * D * 2, 4096, dev,
                         &dVc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)ROWS * W * 4, 4096, dev, &dS));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)ROWS * W * 4, 4096, dev, &dW));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)ROWS * W * 2, 4096, dev, &dWh));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)ROWS * D * 4, 4096, dev, &dO));
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
  auto mkker = [&](ze_module_handle_t mod, const char *entry,
                   int gx, int gy, int gz) -> ze_kernel_handle_t {
    ze_kernel_handle_t k = nullptr;
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, entry};
    if (zeKernelCreate(mod, &kd, &k) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "entry %s not found\n", entry);
      std::exit(1);
    }
    if (zeKernelSetGroupSize(k, gx, gy, gz) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "setgroupsize failed\n");
      std::exit(1);
    }
    return k;
  };
  ze_module_handle_t mQK = mkmod(spvQK), mSM = mkmod(spvSM),
                     mCV = mkmod(spvCV), mWV = mkmod(spvWV);
  ze_kernel_handle_t kQK = mkker(mQK, "_ZTS11ChunkQkGemm", 16, 1, 1);
  ze_kernel_handle_t kSM = mkker(mSM, "_ZTS15ChunkSoftmaxRow", 1, 1, 1);
  ze_kernel_handle_t kCV = mkker(mCV, "_ZTS9CvtF32F16", 1, 1, 1);
  ze_kernel_handle_t kWV = mkker(mWV, "_ZTS11ChunkWvGemm", 16, 1, 1);
  int pArg = P, mArg = M, wArg = W, smaxArg = W, tmaxArg = W, rowsArg = ROWS;
  CHECK(zeKernelSetArgumentValue(kQK, 0, sizeof(void *), &dQ));
  CHECK(zeKernelSetArgumentValue(kQK, 1, sizeof(void *), &dKc));
  CHECK(zeKernelSetArgumentValue(kQK, 2, sizeof(void *), &dS));
  CHECK(zeKernelSetArgumentValue(kQK, 3, sizeof(int), &pArg));
  CHECK(zeKernelSetArgumentValue(kQK, 4, sizeof(int), &mArg));
  CHECK(zeKernelSetArgumentValue(kQK, 5, sizeof(int), &wArg));
  CHECK(zeKernelSetArgumentValue(kQK, 6, sizeof(int), &smaxArg));
  CHECK(zeKernelSetArgumentValue(kQK, 7, (size_t)8 * 256 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(kQK, 8, (size_t)16 * 16 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(kQK, 9, (size_t)8 * 16 * 4, nullptr));
  CHECK(zeKernelSetArgumentValue(kSM, 0, sizeof(void *), &dW));
  CHECK(zeKernelSetArgumentValue(kSM, 1, sizeof(void *), &dS));
  CHECK(zeKernelSetArgumentValue(kSM, 2, sizeof(void *), &dCtrl));
  CHECK(zeKernelSetArgumentValue(kSM, 3, sizeof(int), &tmaxArg));
  CHECK(zeKernelSetArgumentValue(kSM, 4, sizeof(int), &rowsArg));
  CHECK(zeKernelSetArgumentValue(kCV, 0, sizeof(void *), &dWh));
  CHECK(zeKernelSetArgumentValue(kCV, 1, sizeof(void *), &dW));
  CHECK(zeKernelSetArgumentValue(kWV, 0, sizeof(void *), &dWh));
  CHECK(zeKernelSetArgumentValue(kWV, 1, sizeof(void *), &dVc));
  CHECK(zeKernelSetArgumentValue(kWV, 2, sizeof(void *), &dO));
  CHECK(zeKernelSetArgumentValue(kWV, 3, sizeof(int), &wArg));
  CHECK(zeKernelSetArgumentValue(kWV, 4, sizeof(int), &wArg));
  CHECK(zeKernelSetArgumentValue(kWV, 5, (size_t)8 * 16 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(kWV, 6, (size_t)16 * 16 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(kWV, 7, (size_t)8 * 16 * 4, nullptr));

  ze_command_queue_handle_t qq = nullptr;
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandQueueCreate(ctx, dev, &qdesc, &qq));
  ze_fence_handle_t fence = nullptr;
  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK(zeFenceCreate(qq, &fdesc, &fence));
  auto mkl = [&]() -> ze_command_list_handle_t {
    ze_command_list_handle_t l = nullptr;
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr,
                                 0, 0};
    if (zeCommandListCreate(ctx, dev, &ld, &l) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "mklist failed\n");
      std::exit(1);
    }
    return l;
  };
  auto app = [&](ze_command_list_handle_t l, ze_kernel_handle_t k,
                 uint32_t n) {
    ze_group_count_t gc = {n, 1, 1};
    CHECK(zeCommandListAppendLaunchKernel(l, k, &gc, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(l, nullptr, 0, nullptr));
  };
  int nBc = (W + 15) / 16;
  ze_command_list_handle_t lQK = mkl();
  app(lQK, kQK, (uint32_t)(M * 4 * nBc));
  CHECK(zeCommandListClose(lQK));
  ze_command_list_handle_t lSM = mkl();
  app(lSM, kSM, (uint32_t)ROWS);
  CHECK(zeCommandListClose(lSM));
  ze_command_list_handle_t lCV = mkl();
  app(lCV, kCV, (uint32_t)(ROWS * W));
  CHECK(zeCommandListClose(lCV));
  ze_command_list_handle_t lWV = mkl();
  app(lWV, kWV, (uint32_t)(M * 64));
  CHECK(zeCommandListClose(lWV));

  ze_command_list_handle_t up = nullptr;
  ze_command_queue_desc_t idesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &idesc, &up));

  uint64_t s = 0xc911c;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  };
  std::vector<float> hQ(ROWS * D), hKc(W * NKV * D), hVc(W * NKV * D);
  for (auto &v : hQ)
    v = rnd() * 2.0f;
  for (auto &v : hKc)
    v = qb(rnd() * 0.5f);
  for (auto &v : hVc)
    v = qb(rnd() * 0.5f);
  std::vector<uint16_t> hQh(hQ.size()), hKcB(hKc.size()), hVcB(hVc.size());
  for (size_t i = 0; i < hQ.size(); ++i)
    hQh[i] = f32_to_f16(hQ[i]);
  for (size_t i = 0; i < hKc.size(); ++i)
    hKcB[i] = f32_to_bf16(hKc[i]);
  for (size_t i = 0; i < hVc.size(); ++i)
    hVcB[i] = f32_to_bf16(hVc[i]);
  CHECK(zeCommandListAppendMemoryCopy(up, dQ, hQh.data(), hQh.size() * 2,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dKc, hKcB.data(), hKcB.size() * 2,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dVc, hVcB.data(), hVcB.size() * 2,
                                      nullptr, 0, nullptr));
  DecodeControl c{9100, W - 1, W, -1};
  CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                      nullptr));

  // Host ref: fp16-Q x qb-cache, causal (c <= P+m else -FLT_MAX), /16,
  // full-width softmax, weighted V.
  const float NEG = -3.402823466e+38f;
  std::vector<float> wts(W), hRef(ROWS * D);
  for (int m = 0; m < M; ++m) {
    for (int hh = 0; hh < NH; ++hh) {
      int kv = hh / 6, row = m * NH + hh;
      float mx = -1e30f;
      for (int t = 0; t < W; ++t) {
        float sc;
        if (t <= P + m) {
          sc = 0;
          for (int d = 0; d < D; ++d)
            sc += f16_to_f32(hQh[(size_t)row * D + d]) *
                  hKc[((size_t)t * NKV + kv) * D + d];
          sc /= 16.0f;
        } else {
          sc = NEG;
        }
        wts[t] = sc;
        mx = sc > mx ? sc : mx;
      }
      float se = 0;
      for (int t = 0; t < W; ++t) {
        float w = expf(wts[t] - mx);
        wts[t] = w;
        se += w;
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0;
        for (int t = 0; t < W; ++t)
          acc += wts[t] / se * hVc[((size_t)t * NKV + kv) * D + d];
        hRef[(size_t)row * D + d] = acc;
      }
    }
  }

  auto exec = [&](ze_command_list_handle_t l) -> double {
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &l, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    return now_ns() - t0;
  };
  double tQK = 0, tSM = 0, tCV = 0, tWV = 0;
  for (int r = 0; r < 3; ++r) {
    tQK += exec(lQK);
    tSM += exec(lSM);
    tCV += exec(lCV);
    tWV += exec(lWV);
  }
  tQK /= 3e3;
  tSM /= 3e3;
  tCV /= 3e3;
  tWV /= 3e3;
  std::vector<float> hO(hRef.size()), hS(ROWS * W);
  CHECK(zeCommandListAppendMemoryCopy(up, hO.data(), dO, hO.size() * 4,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, hS.data(), dS, hS.size() * 4,
                                      nullptr, 0, nullptr));
  bool ok = true;
  // Causal-mask direct check: row m=0, col W-1 (masked) must be -FLT_MAX.
  if (hS[(size_t)0 * W + (W - 1)] != NEG) {
    std::fprintf(stderr, "mask leak: S[0][W-1] = %g, want -FLT_MAX\n",
                 hS[(size_t)0 * W + (W - 1)]);
    ok = false;
  }
  // Unmasked score sanity: row M-1 sees all cols; spot-check finite.
  if (!(hS[(size_t)(ROWS - 1) * W + (W - 1)] > NEG / 2)) {
    std::fprintf(stderr, "last-row score not finite: %g\n",
                 hS[(size_t)(ROWS - 1) * W + (W - 1)]);
    ok = false;
  }
  double worstRel = 0, refmax = 0;
  for (float v : hRef)
    refmax = std::max(refmax, (double)std::fabs(v));
  for (size_t j = 0; j < hRef.size(); ++j) {
    double rel = std::fabs((double)hO[j] - (double)hRef[j]) /
                 (refmax > 0 ? refmax : 1);
    if (rel > worstRel)
      worstRel = rel;
    if (rel > 1e-3) {
      ok = false;
      std::fprintf(stderr, "ref mismatch j %zu: got %g want %g\n", j, hO[j],
                   hRef[j]);
      break;
    }
  }
  std::printf("chunkqkwv M=%d P=%d W=%d qk %.1f sm %.1f cv %.1f wv %.1f us "
              "total %.1f worst-rel %.2e %s\n",
              M, P, W, tQK, tSM, tCV, tWV, tQK + tSM + tCV + tWV, worstRel,
              ok ? "CHUNKQKWV-OK" : "MISMATCH");
  char js[512];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"chunkqkwv\":\"qk-gemm-causal+softmax+"
                "cvt+wv-gemm\",\"M\":%d,\"P\":%d,\"W\":%d,"
                "\"us\":{\"qk\":%.1f,\"sm\":%.1f,\"cv\":%.1f,\"wv\":%.1f},"
                "\"worst_rel\":%.2e,\"ref_tol\":1e-3,\"chunkqkwv_ok\":%s}",
                M, P, W, tQK, tSM, tCV, tWV, worstRel,
                ok ? "true" : "false");
  if (argc > 5) {
    FILE *o = std::fopen(argv[5], "w");
    if (o) {
      std::fprintf(o, "%s\n", js);
      std::fclose(o);
    }
  } else {
    std::printf("%s\n", js);
  }
  return ok ? 0 : 1;
}
