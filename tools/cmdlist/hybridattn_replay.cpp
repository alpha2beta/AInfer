// T7.4 hybrid attention probe: QK-GEMM (DPAS) -> row-softmax -> fp16 convert
// -> WV-GEMM (DPAS) as staged recorded lists, vs the SAME full host ref as
// attnfar (algorithm is an implementation detail; gate applied host-side on
// both sides so the device probe covers the core). Per-T lists (1024/2048/
// 4096) since GEMM grid geometry bakes N/K. Staged timing isolates each step.
// Usage: hybridattn_replay <qkgemm.spv> <softmaxrow.spv> <cvtf32f16.spv> <wvgemm.spv> [report.json]
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
  if (argc < 6) {
    std::fprintf(stderr, "usage: hybridattn_replay <qk> <sm> <cvt> <wv> "
                         "<resaddf> [report.json]\n");
    return 2;
  }
  const int NH = 24, NKV = 4, D = 256, TMAX = 4096;
  // LOOPCOND env: replicate loop conditions (T=1, N=16 baked width,
  // K magnitudes x20, V x89, zero prefix) to chase the L3 divergence.
  const bool loopcond = std::getenv("HYB_LOOPCOND") != nullptr;
  int TS[] = {16, 64, 1024};
  float kMag = 0.5f, vMag = 0.5f;
  if (loopcond) {
    TS[0] = 1;
    kMag = 20.0f;
    vMag = 89.0f;
  }
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
                       spvCV = load_spv(argv[3]), spvWV = load_spv(argv[4]),
                       spvRES = load_spv(argv[5]);

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
  void *dQh = nullptr, *dKc = nullptr, *dVc = nullptr, *dS = nullptr,
       *dW = nullptr, *dWh = nullptr, *dAtt = nullptr, *dCtrl = nullptr,
       *dWVO = nullptr, *dWTmp = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 2, 4096, dev, &dQh));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NKV * TMAX * D * 2, 4096, dev,
                         &dKc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NKV * TMAX * D * 2, 4096, dev,
                         &dVc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * TMAX * 4, 4096, dev, &dS));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * TMAX * 4, 4096, dev, &dW));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * TMAX * 2, 4096, dev, &dWh));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dAtt));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)4 * NH * D * 4, 4096, dev,
                         &dWVO));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dWTmp));
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
                     mCV = mkmod(spvCV), mWV = mkmod(spvWV),
                     mRES = mkmod(spvRES);
  ze_kernel_handle_t kRES = mkker(mRES, "_ZTS7ResAddF", 1, 1, 1);
  ze_kernel_handle_t kQK = mkker(mQK, "_ZTS6QkGemm", 16, 1, 1);
  ze_kernel_handle_t kSM = mkker(mSM, "_ZTS10SoftmaxRow", 1, 1, 1);
  ze_kernel_handle_t kCV = mkker(mCV, "_ZTS9CvtF32F16", 1, 1, 1);
  ze_kernel_handle_t kWV = mkker(mWV, "_ZTS6WvGemm", 16, 1, 1);
  int tmaxArg = TMAX;
  // QK args (N varies per list; set below per T).
  CHECK(zeKernelSetArgumentValue(kQK, 0, sizeof(void *), &dQh));
  CHECK(zeKernelSetArgumentValue(kQK, 1, sizeof(void *), &dKc));
  CHECK(zeKernelSetArgumentValue(kQK, 2, sizeof(void *), &dS));
  CHECK(zeKernelSetArgumentValue(kQK, 4, sizeof(int), &tmaxArg));
  CHECK(zeKernelSetArgumentValue(kQK, 6, (size_t)8 * 256 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(kQK, 7, (size_t)16 * 16 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(kQK, 8, (size_t)8 * 16 * 4, nullptr));
  CHECK(zeKernelSetArgumentValue(kSM, 0, sizeof(void *), &dW));
  CHECK(zeKernelSetArgumentValue(kSM, 1, sizeof(void *), &dS));
  CHECK(zeKernelSetArgumentValue(kSM, 2, sizeof(void *), &dCtrl));
  CHECK(zeKernelSetArgumentValue(kSM, 3, sizeof(int), &tmaxArg));
  CHECK(zeKernelSetArgumentValue(kCV, 0, sizeof(void *), &dWh));
  CHECK(zeKernelSetArgumentValue(kCV, 1, sizeof(void *), &dW));
  CHECK(zeKernelSetArgumentValue(kWV, 0, sizeof(void *), &dWh));
  CHECK(zeKernelSetArgumentValue(kWV, 1, sizeof(void *), &dVc));
  CHECK(zeKernelSetArgumentValue(kWV, 4, sizeof(int), &tmaxArg));
  CHECK(zeKernelSetArgumentValue(kWV, 7, (size_t)8 * 16 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(kWV, 8, (size_t)16 * 16 * 2, nullptr));
  CHECK(zeKernelSetArgumentValue(kWV, 9, (size_t)8 * 16 * 4, nullptr));

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
  // Per-T lists (grid geometry bakes N/K). GEMMs run 4 kv-group appends
  // per list (args bake per append); each covers Q rows kv*6..+6. WV splits
  // K into 4 chunks (KChunk) for occupancy; partials combine via ResAddF.
  // Softmax is Ctrl-driven: ONE list serves all T.
  ze_command_list_handle_t lQK[3], lSM1, lWV[3][4], lCB[3];
  auto resapp = [&](ze_command_list_handle_t l, void *Y, void *A, void *B) {
    CHECK(zeKernelSetArgumentValue(kRES, 0, sizeof(void *), &Y));
    CHECK(zeKernelSetArgumentValue(kRES, 1, sizeof(void *), &A));
    CHECK(zeKernelSetArgumentValue(kRES, 2, sizeof(void *), &B));
    app(l, kRES, (uint32_t)(NH * D));
  };
  for (int i = 0; i < 3; ++i) {
    int T = TS[i];
    lQK[i] = mkl();
    CHECK(zeKernelSetArgumentValue(kQK, 3, sizeof(int), &T));
    for (int kv = 0; kv < 4; ++kv) {
      CHECK(zeKernelSetArgumentValue(kQK, 5, sizeof(int), &kv));
      app(lQK[i], kQK, (uint32_t)(T / 16));
    }
    CHECK(zeCommandListClose(lQK[i]));
    for (int cc = 0; cc < 4; ++cc) {
      lWV[i][cc] = mkl();
      CHECK(zeKernelSetArgumentValue(kWV, 3, sizeof(int), &T));
      CHECK(zeKernelSetArgumentValue(kWV, 6, sizeof(int), &cc));
      for (int kv = 0; kv < 4; ++kv) {
        void *po = (char *)dWVO + (size_t)cc * NH * D * 4;
        CHECK(zeKernelSetArgumentValue(kWV, 2, sizeof(void *), &po));
        CHECK(zeKernelSetArgumentValue(kWV, 5, sizeof(int), &kv));
        app(lWV[i][cc], kWV, (uint32_t)16);
      }
      CHECK(zeCommandListClose(lWV[i][cc]));
    }
    // Combine: tmp = P0+P1; tmp += P2; dAtt = tmp+P3 (full 6144 spans).
    lCB[i] = mkl();
    void *p0 = dWVO, *p1 = (char *)dWVO + (size_t)NH * D * 4,
         *p2 = (char *)dWVO + (size_t)2 * NH * D * 4,
         *p3 = (char *)dWVO + (size_t)3 * NH * D * 4;
    resapp(lCB[i], dWTmp, p0, p1);
    resapp(lCB[i], dWTmp, dWTmp, p2);
    resapp(lCB[i], dAtt, dWTmp, p3);
    CHECK(zeCommandListClose(lCB[i]));
  }
  ze_command_list_handle_t lCV = mkl();
  app(lCV, kCV, (uint32_t)(NH * TMAX));
  CHECK(zeCommandListClose(lCV));
  // Softmax is Ctrl-driven: one list for all T (dCtrl updated per replay).
  lSM1 = mkl();
  app(lSM1, kSM, (uint32_t)NH);
  CHECK(zeCommandListClose(lSM1));

  ze_command_list_handle_t up = nullptr;
  ze_command_queue_desc_t idesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &idesc, &up));

  uint64_t s = 0x71b1d;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  };
  std::vector<float> hQ(NH * D), hG(NH * D), hAtt(NH * D), hRef(NH * D),
      hKc(NKV * TMAX * D), hVc(NKV * TMAX * D);
  for (auto &v : hKc)
    v = qb(rnd() * 0.5f);
  for (auto &v : hVc)
    v = qb(rnd() * 0.5f);
  std::vector<uint16_t> hKcB(hKc.size()), hVcB(hVc.size());
  for (size_t i = 0; i < hKc.size(); ++i)
    hKcB[i] = f32_to_bf16(hKc[i]);
  for (size_t i = 0; i < hVc.size(); ++i)
    hVcB[i] = f32_to_bf16(hVc[i]);
  CHECK(zeCommandListAppendMemoryCopy(up, dKc, hKcB.data(), hKcB.size() * 2,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dVc, hVcB.data(), hVcB.size() * 2,
                                      nullptr, 0, nullptr));

  auto exec = [&](ze_command_list_handle_t l) -> double {
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &l, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    return now_ns() - t0;
  };
  bool ok = true;
  double worstRel = 0, us4096 = 0, qk4096 = 0, sm4096 = 0, cv4096 = 0,
         wv4096 = 0;
  std::vector<float> wts(TMAX), hQh(NH * D);
  for (int ii = 0; ii < 3; ++ii) {
    int T = TS[ii];
    for (auto &v : hQ)
      v = rnd() * 2.0f;
    for (auto &v : hG)
      v = rnd();
    for (size_t i = 0; i < hQ.size(); ++i)
      hQh[i] = f32_to_f16(hQ[i]);
    DecodeControl c{9000 + T, T - 1, T, -1};
    // Host ref: fp16-Q x qb-cache, full softmax, weighted V (no gate here;
    // gate applied to NEITHER side — core probe).
    for (int hh = 0; hh < NH; ++hh) {
      int kv = hh / 6;
      float mx = -1e30f;
      for (int t = 0; t < T; ++t) {
        float sc = 0;
        for (int d = 0; d < D; ++d) {
          uint16_t hb = hQh[(size_t)hh * D + d];
          uint32_t sign = ((uint32_t)hb & 0x8000u) << 16;
          int exp = (hb >> 10) & 0x1F;
          uint32_t mant = (uint32_t)(hb & 0x3FFu) << 13;
          uint32_t u = exp == 0 ? sign : sign | ((uint32_t)(exp + 112) << 23) | mant;
          float a;
          std::memcpy(&a, &u, 4);
          sc += a * hKc[((size_t)t * NKV + kv) * D + d];
        }
        sc /= 16.0f;
        wts[t] = sc;
        mx = sc > mx ? sc : mx;
      }
      float se = 0;
      for (int t = 0; t < T; ++t) {
        float w = expf(wts[t] - mx);
        wts[t] = w;
        se += w;
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0;
        for (int t = 0; t < T; ++t)
          acc += wts[t] / se * hVc[((size_t)t * NKV + kv) * D + d];
        hRef[(size_t)hh * D + d] = acc;
      }
    }
    std::vector<uint16_t> hQh16(hQ.size());
    for (size_t i = 0; i < hQ.size(); ++i)
      hQh16[i] = f32_to_f16(hQ[i]);
    CHECK(zeCommandListAppendMemoryCopy(up, dQh, hQh16.data(), hQh16.size() * 2,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                        nullptr));
    double tQK = exec(lQK[ii]) / 1e3, tSM = exec(lSM1) / 1e3,
           tCV = exec(lCV) / 1e3;
    double tWVs = 0;
    for (int cc = 0; cc < 4; ++cc)
      tWVs += exec(lWV[ii][cc]);
    double tWV = tWVs / 1e3, tCB = exec(lCB[ii]) / 1e3;
    if (T == 4096) {
      qk4096 = tQK;
      sm4096 = tSM;
      cv4096 = tCV;
      wv4096 = tWV + tCB;
      us4096 = tQK + tSM + tCV + tWV + tCB;
    }
    CHECK(zeCommandListAppendMemoryCopy(up, hAtt.data(), dAtt, hAtt.size() * 4,
                                        nullptr, 0, nullptr));
    double refmax = 0;
    for (float v : hRef)
      refmax = std::max(refmax, (double)std::fabs(v));
    for (size_t j = 0; j < hRef.size(); ++j) {
      double rel = std::fabs((double)hAtt[j] - (double)hRef[j]) /
                   (refmax > 0 ? refmax : 1);
      if (rel > worstRel)
        worstRel = rel;
      if (rel > 1e-3) {
        ok = false;
        std::fprintf(stderr, "ref mismatch T=%d j %zu: got %g want %g\n", T, j,
                     hAtt[j], hRef[j]);
        break;
      }
    }
    if (!ok)
      break;
    std::printf("hybrid T=%d qk %.1f sm %.1f cv %.1f wv %.1f cb %.1f us "
                "total %.1f\n",
                T, tQK, tSM, tCV, tWV, tCB, tQK + tSM + tCV + tWV + tCB);
  }
  std::printf("hybrid worst-rel %.2e %s\n", worstRel,
              ok ? "HYBRID-OK" : "MISMATCH");
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"hybrid\":\"qk-gemm+softmax+cvt+wv-gemm\","
                "\"t_points\":[1024,2048,4096],\"us_at_4096\":%.1f,"
                "\"stages_4096\":{\"qk\":%.1f,\"sm\":%.1f,\"cv\":%.1f,"
                "\"wv\":%.1f},\"worst_rel\":%.2e,\"ref_tol\":1e-3,"
                "\"hybrid_ok\":%s}",
                us4096, qk4096, sm4096, cv4096, wv4096, worstRel,
                ok ? "true" : "false");
  if (argc > 6) {
    FILE *o = std::fopen(argv[6], "w");
    if (o) {
      std::fprintf(o, "%s\n", js);
      std::fclose(o);
    }
  } else {
    std::printf("%s\n", js);
  }
  return ok ? 0 : 1;
}
