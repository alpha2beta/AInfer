// T7.4 chunked prefill: FULL linear-layer chunk orchestration. One recorded
// list composes chunk GEMMs (qkv/z/b/a/out/gate/up/down at M=32) with chunk
// SSM (persistent conv history + recurrent state) and per-row stateless
// stages (norms, split, l2, beta, gating, silu, residuals), replayed as a
// 2-chunk sequence with conv/SSM state carrying A->B. Synthetic layout-0 INT4
// weights at real shapes; fp16 GEMM inputs via in-list converts (layer stream
// stays fp32). Checks: chunk outputs vs order-mirrored host float ref (tol
// 1e-3: fp16 path, no INT8 flips here) + cross-chunk continuity (B-after-A
// differs from B-alone) + full reset-rerun bitwise identical.
// Usage: chunklayer_replay <12 spv modules...> [report.json]
//   order: norm chunkgemm ssmconv(chunkssmconv) ssmrecur(chunkssmrecur) silu
//          splitrepeat l2normqk betag rmsinv normgated resaddf cvtf32f16
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
// Host fp16 convert with RNE (matches sycl::half conversion for verification;
// residual risk covered by the 1e-3 layer tolerance).
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
  uint32_t m10 = mant >> 13, rest = mant & 0x1FFFu;
  if (rest > 0x1000u || (rest == 0x1000u && (m10 & 1u))) {
    if (++m10 == 0x400u) {
      m10 = 0;
      if (++exp >= 31)
        return (uint16_t)(sign | 0x7BFFu);
    }
  }
  return (uint16_t)(sign | ((uint32_t)exp << 10) | m10);
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

static uint64_t prng = 0xC4C4C4;
static float frnd(float s) {
  prng = prng * 6364136223846793005ull + 1442695040888963407ull;
  return (float)((int)((prng >> 33) & 0xFFFF) - 32768) * (s / 32768.0f);
}

struct Mat {
  const char *nm;
  int M, K, GG;
  void *P = nullptr, *S = nullptr;
  std::vector<uint8_t> packed;
  std::vector<uint16_t> scales;
};

int main(int argc, char **argv) {
  if (argc < 13) {
    std::fprintf(stderr, "usage: chunklayer_replay <12 spv> [report]\n");
    return 2;
  }
  const int H = 5120, C = 10240, V6 = 6144, I = 17408, NH = 48, D = 128;
  const int M = 32;
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
  const char *entries[] = {
      "_ZTS8RMSNormW", "_ZTS9ChunkGemm",  "_ZTS12ChunkSsmConv",
      "_ZTS13ChunkSsmRecur", "_ZTS7SiluMul",  "_ZTS11SplitRepeat",
      "_ZTS8L2NormQK",   "_ZTS5BetaG",    "_ZTS6RmsInv",  "_ZTS9NormGated",
      "_ZTS7ResAddF",  "_ZTS9CvtF32F16"};

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
  auto alloc = [&](size_t n) -> void * {
    void *p = nullptr;
    if (zeMemAllocDevice(ctx, &mdesc, n, 4096, dev, &p) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "alloc %zu failed\n", n);
      std::exit(1);
    }
    return p;
  };
  const size_t MH = (size_t)M * H, MC = (size_t)M * C, MV = (size_t)M * V6,
               MI = (size_t)M * I, MH48 = (size_t)M * NH;
  void *dX = alloc(MH * 4), *dH = alloc(MH * 4), *dTmp = alloc(MH * 4),
       *dMix = alloc(MH * 4), *dHh = alloc(MH * 2);
  void *dQKV = alloc(MC * 4), *dMxC = alloc(MC * 4), *dCS = alloc(C * 3 * 4),
       *dCW = alloc((size_t)C * 4 * 4);
  // Dedicated recur-output scratch (M*6144): aliasing recur Out onto the
  // C-stride conv buffer corrupts rows m >= 1 (stride 6144 vs 10240 trap).
  void *dMxR = alloc(MV * 4);
  void *dZ = alloc(MV * 4), *dQ48 = alloc(MV * 4), *dK48 = alloc(MV * 4),
       *dV48 = alloc(MV * 4), *dAtt = alloc(MV * 4), *dAtth = alloc(MV * 2);
  void *dS = alloc((size_t)NH * D * D * 4);
  void *dB = alloc(MH48 * 4), *dA = alloc(MH48 * 4), *dBt = alloc(MH48 * 4),
       *dG48 = alloc(MH48 * 4);
  void *dG17 = alloc(MI * 4), *dU17 = alloc(MI * 4), *dG17h = alloc(MI * 2);
  void *dInN = alloc(H * 4), *dPostN = alloc(H * 4), *dNG = alloc(128 * 4),
       *dAL = alloc(NH * 4), *dDT = alloc(NH * 4);

  ze_kernel_handle_t kh[12] = {nullptr};
  std::vector<std::vector<uint8_t>> spvs;
  for (int i = 0; i < 12; ++i)
    spvs.push_back(load_spv(argv[1 + i]));
  enum K {
    NORM,
    GEMM,
    CONV,
    RECUR,
    SILU,
    SPLIT,
    L2,
    BETA,
    RMSI,
    GATE,
    RES,
    CVT
  };
  for (int i = 0; i < 12; ++i) {
    ze_module_handle_t mod = nullptr;
    ze_module_desc_t mddesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                               nullptr,
                               ZE_MODULE_FORMAT_IL_SPIRV,
                               spvs[i].size(),
                               spvs[i].data(),
                               nullptr,
                               nullptr};
    CHECK(zeModuleCreate(ctx, dev, &mddesc, &mod, nullptr));
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           entries[i]};
    if (zeKernelCreate(mod, &kd, &kh[i]) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "entry %s not found\n", entries[i]);
      return 1;
    }
    CHECK(zeKernelSetGroupSize(kh[i], 1, 1, 1));
  }
  // ChunkGemm needs groups of 16 (one SG16 subgroup per group; the entry
  // has no reqd_sub_group_size spelling on functor operator()).
  CHECK(zeKernelSetGroupSize(kh[GEMM], 16, 1, 1));
  // RMSNormW is the parallel 1-group-x-256-WI SLM tree (T6.1); group size 1
  // leaves 255 SLM lanes unwritten (garbage inv) — the failure this fixes.
  CHECK(zeKernelSetGroupSize(kh[NORM], 256, 1, 1));

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
  ze_command_list_handle_t up = nullptr;
  ze_command_queue_desc_t idesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &idesc, &up));
  auto setarg = [&](ze_kernel_handle_t k, uint32_t idx, size_t sz,
                    const void *p) {
    if (zeKernelSetArgumentValue(k, idx, sz, p) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "setArg %u failed\n", idx);
      std::exit(1);
    }
  };
  auto launch = [&](ze_kernel_handle_t k, uint32_t count) {
    ze_group_count_t gc = {count, 1, 1};
    ze_result_t r1 =
        zeCommandListAppendLaunchKernel(reg, k, &gc, nullptr, 0, nullptr);
    ze_result_t r2 = zeCommandListAppendBarrier(reg, nullptr, 0, nullptr);
    if (r1 != ZE_RESULT_SUCCESS || r2 != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "append failed\n");
      std::exit(1);
    }
  };
  // Per-row single-row kernels over the chunk (recorded; production would
  // batch these — noted, not blocking the orchestration proof).
  auto normYa = [&](void *Y, void *X, void *W) {
    int nn = H;
    for (int m = 0; m < M; ++m) {
      void *yy = (char *)Y + (size_t)m * H * 4;
      void *xx = (char *)X + (size_t)m * H * 4;
      setarg(kh[NORM], 0, sizeof(void *), &yy);
      setarg(kh[NORM], 1, sizeof(void *), &xx);
      setarg(kh[NORM], 2, sizeof(void *), &W);
      setarg(kh[NORM], 3, sizeof(int), &nn);
      setarg(kh[NORM], 4, (size_t)256 * 8, nullptr);
      launch(kh[NORM], 1);
    }
  };
  auto cvtYa = [&](void *Oh, void *X, int nn) {
    setarg(kh[CVT], 0, sizeof(void *), &Oh);
    setarg(kh[CVT], 1, sizeof(void *), &X);
    launch(kh[CVT], nn);
  };
  auto cgemm = [&](Mat &mt, void *Ah, void *Y) {
    int kk = mt.K, nn = mt.M, mm = M;
    setarg(kh[GEMM], 0, sizeof(void *), &Ah);
    setarg(kh[GEMM], 1, sizeof(void *), &mt.P);
    setarg(kh[GEMM], 2, sizeof(void *), &mt.S);
    setarg(kh[GEMM], 3, sizeof(void *), &Y);
    setarg(kh[GEMM], 4, sizeof(int), &mm);
    setarg(kh[GEMM], 5, sizeof(int), &kk);
    setarg(kh[GEMM], 6, sizeof(int), &nn);
    setarg(kh[GEMM], 7, (size_t)512 * 2, nullptr);
    setarg(kh[GEMM], 8, (size_t)256 * 2, nullptr);
    setarg(kh[GEMM], 9, (size_t)512 * 4, nullptr);
    launch(kh[GEMM], (mm / 32) * (nn / 16));
  };
  auto resYa = [&](void *Y, void *A, void *B) {
    for (int m = 0; m < M; ++m) {
      void *yy = (char *)Y + (size_t)m * H * 4;
      void *aa = (char *)A + (size_t)m * H * 4;
      void *bb = (char *)B + (size_t)m * H * 4;
      setarg(kh[RES], 0, sizeof(void *), &yy);
      setarg(kh[RES], 1, sizeof(void *), &aa);
      setarg(kh[RES], 2, sizeof(void *), &bb);
      launch(kh[RES], H);
    }
  };

  Mat mats[] = {{"qkv", C, H, 40},  {"z", V6, H, 40},   {"b", NH, H, 40},
                {"a", NH, H, 40},   {"out", H, V6, 48}, {"gate", I, H, 40},
                {"up", I, H, 40},   {"down", H, I, 136}};
  for (auto &mt : mats) {
    mt.packed.assign((size_t)mt.M * mt.K / 2, 0);
    mt.scales.assign((size_t)mt.M * mt.GG, 0);
    for (int m = 0; m < mt.M; ++m)
      for (int g = 0; g < mt.GG; ++g) {
        float amax = 0, w[128];
        for (int j = 0; j < 128; ++j) {
          w[j] = frnd(0.02f);
          amax = std::max(amax, std::fabs(w[j]));
        }
        float sc = amax == 0 ? 1.0f : amax / 7.0f;
        mt.scales[(size_t)m * mt.GG + g] = f32_to_bf16(sc);
        for (int j = 0; j < 128; ++j) {
          int q = (int)lrintf(w[j] / sc);
          q = q < -8 ? -8 : (q > 7 ? 7 : q);
          size_t idx = (size_t)m * mt.K + g * 128 + j;
          if (j & 1)
            mt.packed[idx / 2] |= (uint8_t)((q & 0xF) << 4);
          else
            mt.packed[idx / 2] = (uint8_t)(q & 0xF);
        }
      }
    mt.P = alloc(mt.packed.size());
    mt.S = alloc(mt.scales.size() * 2);
    CHECK(zeCommandListAppendMemoryCopy(up, mt.P, mt.packed.data(),
                                        mt.packed.size(), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, mt.S, mt.scales.data(),
                                        mt.scales.size() * 2, nullptr, 0,
                                        nullptr));
  }
  std::vector<float> hInN(H), hPostN(H), hNG(128), hAL(NH), hDT(NH),
      hCW(C * 4);
  for (auto &v : hInN)
    v = frnd(0.1f);
  for (auto &v : hPostN)
    v = frnd(0.1f);
  for (auto &v : hNG)
    v = frnd(0.1f);
  for (auto &v : hAL)
    v = frnd(0.05f);
  for (auto &v : hDT)
    v = frnd(0.05f);
  for (auto &v : hCW)
    v = frnd(0.5f);
  CHECK(zeCommandListAppendMemoryCopy(up, dInN, hInN.data(), H * 4, nullptr, 0,
                                      nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dPostN, hPostN.data(), H * 4, nullptr,
                                      0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dNG, hNG.data(), 128 * 4, nullptr, 0,
                                      nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dAL, hAL.data(), NH * 4, nullptr, 0,
                                      nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dDT, hDT.data(), NH * 4, nullptr, 0,
                                      nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dCW, hCW.data(), hCW.size() * 4,
                                      nullptr, 0, nullptr));

  // ---- record one full chunked linear layer ----
  int mmA = M;
  normYa(dH, dX, dInN);
  cvtYa(dHh, dH, M * H);
  cgemm(mats[0], dHh, dQKV);
  cgemm(mats[1], dHh, dZ);
  cgemm(mats[2], dHh, dB);
  cgemm(mats[3], dHh, dA);
  setarg(kh[CONV], 0, sizeof(void *), &dMxC);
  setarg(kh[CONV], 1, sizeof(void *), &dQKV);
  setarg(kh[CONV], 2, sizeof(void *), &dCS);
  setarg(kh[CONV], 3, sizeof(void *), &dCW);
  setarg(kh[CONV], 4, sizeof(int), &C);
  setarg(kh[CONV], 5, sizeof(int), &mmA);
  launch(kh[CONV], C);
  for (int m = 0; m < M; ++m) { // split + l2 + beta per row (recorded loop)
    void *mx = (char *)dMxC + (size_t)m * C * 4;
    void *q4 = (char *)dQ48 + (size_t)m * V6 * 4;
    void *k4 = (char *)dK48 + (size_t)m * V6 * 4;
    void *v4 = (char *)dV48 + (size_t)m * V6 * 4;
    setarg(kh[SPLIT], 0, sizeof(void *), &mx);
    setarg(kh[SPLIT], 1, sizeof(void *), &q4);
    setarg(kh[SPLIT], 2, sizeof(void *), &k4);
    setarg(kh[SPLIT], 3, sizeof(void *), &v4);
    launch(kh[SPLIT], V6);
    setarg(kh[L2], 0, sizeof(void *), &q4);
    setarg(kh[L2], 1, sizeof(void *), &k4);
    launch(kh[L2], 96);
    void *b1 = (char *)dB + (size_t)m * NH * 4;
    void *a1 = (char *)dA + (size_t)m * NH * 4;
    void *bt1 = (char *)dBt + (size_t)m * NH * 4;
    void *g1 = (char *)dG48 + (size_t)m * NH * 4;
    setarg(kh[BETA], 0, sizeof(void *), &bt1);
    setarg(kh[BETA], 1, sizeof(void *), &g1);
    setarg(kh[BETA], 2, sizeof(void *), &b1);
    setarg(kh[BETA], 3, sizeof(void *), &a1);
    setarg(kh[BETA], 4, sizeof(void *), &dAL);
    setarg(kh[BETA], 5, sizeof(void *), &dDT);
    launch(kh[BETA], NH);
  }
  setarg(kh[RECUR], 0, sizeof(void *), &dMxR);
  setarg(kh[RECUR], 1, sizeof(void *), &dQ48);
  setarg(kh[RECUR], 2, sizeof(void *), &dK48);
  setarg(kh[RECUR], 3, sizeof(void *), &dV48);
  setarg(kh[RECUR], 4, sizeof(void *), &dS);
  setarg(kh[RECUR], 5, sizeof(void *), &dBt);
  setarg(kh[RECUR], 6, sizeof(void *), &dG48);
  setarg(kh[RECUR], 7, sizeof(int), &mmA);
  launch(kh[RECUR], NH);
  for (int m = 0; m < M; ++m) { // rms + gate per row
    void *mx = (char *)dMxR + (size_t)m * V6 * 4;
    void *bt1 = (char *)dBt + (size_t)m * NH * 4;
    void *at = (char *)dAtt + (size_t)m * V6 * 4;
    void *zz = (char *)dZ + (size_t)m * V6 * 4;
    setarg(kh[RMSI], 0, sizeof(void *), &bt1);
    setarg(kh[RMSI], 1, sizeof(void *), &mx);
    launch(kh[RMSI], NH);
    setarg(kh[GATE], 0, sizeof(void *), &at);
    setarg(kh[GATE], 1, sizeof(void *), &mx);
    setarg(kh[GATE], 2, sizeof(void *), &zz);
    setarg(kh[GATE], 3, sizeof(void *), &dNG);
    setarg(kh[GATE], 4, sizeof(void *), &bt1);
    launch(kh[GATE], V6);
  }
  cvtYa(dAtth, dAtt, M * V6);
  cgemm(mats[4], dAtth, dMix);
  resYa(dTmp, dX, dMix);
  normYa(dH, dTmp, dPostN);
  cvtYa(dHh, dH, M * H);
  cgemm(mats[5], dHh, dG17);
  cgemm(mats[6], dHh, dU17);
  setarg(kh[SILU], 0, sizeof(void *), &dG17);
  setarg(kh[SILU], 1, sizeof(void *), &dU17);
  setarg(kh[SILU], 2, sizeof(void *), &dG17);
  launch(kh[SILU], MI);
  cvtYa(dG17h, dG17, MI);
  cgemm(mats[7], dG17h, dMix);
  resYa(dX, dTmp, dMix);
  ze_fence_handle_t fence = nullptr;
  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK(zeFenceCreate(qq, &fdesc, &fence));
  CHECK(zeCommandListClose(reg));

  // ---- host reference mirrors device order (float; cvt via RNE f16) ----
  std::vector<float> hX(MH), hH(MH), hTmp(MH), hMix(MH), hQKV(MC), hMxC(MC),
      hMxR(MV),
      hCS(C * 3, 0.0f), hZ(MV), hQ48(MV), hK48(MV), hV48(MV), hAtt(MV),
      hS(NH * D * D, 0.0f), hB(MH48), hA(MH48), hBt(MH48), hG48(MH48),
      hG17(MI), hU17(MI);
  std::vector<float> hXa0(MH), hXb0(MH), hRef(MH);
  for (auto &v : hXa0)
    v = frnd(1.0f);
  for (auto &v : hXb0)
    v = frnd(2.0f);
  auto hnorm = [&](float *Y, float *X, float *W) {
    double ss = 0;
    for (int j = 0; j < H; ++j)
      ss += (double)X[j] * X[j];
    float inv = 1.0f / std::sqrt((float)(ss / H) + 1e-6f);
    for (int j = 0; j < H; ++j)
      Y[j] = X[j] * inv * (1.0f + W[j]);
  };
  auto hgemm = [&](Mat &mt, uint16_t *Ah, float *Y) {
    int MR = mt.M, K = mt.K, GG = mt.GG;
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < MR; ++n) {
        double acc = 0;
        for (int g = 0; g < GG; ++g) {
          float sc = bf16_to_f32(mt.scales[(size_t)n * GG + g]);
          for (int j = 0; j < 128; ++j) {
            size_t idx = (size_t)n * K + g * 128 + j;
            uint8_t b = mt.packed[idx / 2];
            int nib = (idx & 1) ? (b >> 4) : (b & 0xF);
            if (nib >= 8)
              nib -= 16;
            acc += (double)f16_to_f32(Ah[(size_t)m * K + g * 128 + j]) *
                   (double)((float)nib * sc);
          }
        }
        Y[(size_t)m * MR + n] = (float)acc;
      }
  };
  std::vector<uint16_t> hHh(MH), hAtth(MV), hG17h(MI);
  auto hcvt = [&](uint16_t *O, float *X, int nn) {
    for (int i = 0; i < nn; ++i)
      O[i] = f32_to_f16(X[i]);
  };
  auto layer_ref = [&]() {
    for (int m = 0; m < M; ++m) {
      hnorm(hH.data() + (size_t)m * H, hX.data() + (size_t)m * H, hInN.data());
      float *qh = hQ48.data() + (size_t)m * V6; // scratch reuse ok (ref-only)
      (void)qh;
    }
    hcvt(hHh.data(), hH.data(), MH);
    hgemm(mats[0], hHh.data(), hQKV.data());
    hgemm(mats[1], hHh.data(), hZ.data());
    hgemm(mats[2], hHh.data(), hB.data());
    hgemm(mats[3], hHh.data(), hA.data());
    for (int cc = 0; cc < C; ++cc)
      for (int m = 0; m < M; ++m) {
        float acc = hCS[cc * 3 + 0] * hCW[(size_t)cc * 4 + 0] +
                    hCS[cc * 3 + 1] * hCW[(size_t)cc * 4 + 1] +
                    hCS[cc * 3 + 2] * hCW[(size_t)cc * 4 + 2] +
                    hQKV[(size_t)m * C + cc] * hCW[(size_t)cc * 4 + 3];
        hMxC[(size_t)m * C + cc] = acc / (1.0f + expf(-acc));
        hCS[cc * 3 + 0] = hCS[cc * 3 + 1];
        hCS[cc * 3 + 1] = hCS[cc * 3 + 2];
        hCS[cc * 3 + 2] = hQKV[(size_t)m * C + cc];
      }
    for (int m = 0; m < M; ++m) {
      for (int i = 0; i < V6; ++i) {
        int hh = i / 128, d = i % 128, kh = hh / 3;
        hQ48[(size_t)m * V6 + i] = hMxC[(size_t)m * C + kh * 128 + d];
        hK48[(size_t)m * V6 + i] =
            hMxC[(size_t)m * C + 2048 + kh * 128 + d];
        hV48[(size_t)m * V6 + i] =
            hMxC[(size_t)m * C + 4096 + hh * 128 + d];
      }
      for (int i = 0; i < 96; ++i) {
        float *X = i < 48 ? hQ48.data() + (size_t)m * V6 + i * 128
                          : hK48.data() + (size_t)m * V6 + (i - 48) * 128;
        float ss = 0;
        for (int d = 0; d < 128; ++d)
          ss += X[d] * X[d];
        float inv = 1.0f / std::sqrt(ss + 1e-6f);
        for (int d = 0; d < 128; ++d)
          X[d] *= inv * (i < 48 ? 0.0883883476f : 1.0f);
      }
      for (int hh = 0; hh < NH; ++hh) {
        hBt[(size_t)m * NH + hh] =
            1.0f / (1.0f + expf(-hB[(size_t)m * NH + hh]));
        float sa = hA[(size_t)m * NH + hh] + hDT[hh];
        float soft = sa > 20 ? sa : std::log(1.0f + expf(sa));
        hG48[(size_t)m * NH + hh] = -expf(hAL[hh]) * soft;
      }
    }
    for (int hh = 0; hh < NH; ++hh) {
      float *S0 = hS.data() + (size_t)hh * D * D;
      for (int m = 0; m < M; ++m) {
        const float *qh = hQ48.data() + ((size_t)m * NH + hh) * D;
        const float *kh = hK48.data() + ((size_t)m * NH + hh) * D;
        const float *vh = hV48.data() + ((size_t)m * NH + hh) * D;
        float gt = expf(hG48[(size_t)m * NH + hh]);
        float bt = hBt[(size_t)m * NH + hh];
        for (int i = 0; i < D * D; ++i)
          S0[i] *= gt;
        float kv[128];
        for (int v = 0; v < D; ++v) {
          float a = 0;
          for (int k = 0; k < D; ++k)
            a += S0[k * D + v] * kh[k];
          kv[v] = a;
        }
        for (int v = 0; v < D; ++v)
          kv[v] = (vh[v] - kv[v]) * bt;
        for (int k = 0; k < D; ++k)
          for (int v = 0; v < D; ++v)
            S0[k * D + v] += kh[k] * kv[v];
        float *oh = hMxR.data() + ((size_t)m * NH + hh) * D;
        for (int v = 0; v < D; ++v) {
          float a = 0;
          for (int k = 0; k < D; ++k)
            a += S0[k * D + v] * qh[k];
          oh[v] = a;
        }
      }
    }
    for (int m = 0; m < M; ++m) {
      for (int hh = 0; hh < NH; ++hh) {
        float ss = 0;
        for (int d = 0; d < 128; ++d) {
          float v = hMxR[((size_t)m * NH + hh) * 128 + d];
          ss += v * v;
        }
        hBt[(size_t)m * NH + hh] = 1.0f / std::sqrt(ss / 128 + 1e-6f);
      }
      for (int i = 0; i < V6; ++i) {
        int hh = i / 128, d = i % 128;
        float zv = hZ[(size_t)m * V6 + i];
        hAtt[(size_t)m * V6 + i] =
            hNG[d] * hMxR[((size_t)m * NH + hh) * 128 + d] *
            hBt[(size_t)m * NH + hh] * (zv / (1.0f + expf(-zv)));
      }
    }
    hcvt(hAtth.data(), hAtt.data(), MV);
    hgemm(mats[4], hAtth.data(), hMix.data());
    for (int j = 0; j < MH; ++j)
      hTmp[j] = hX[j] + hMix[j];
    for (int m = 0; m < M; ++m)
      hnorm(hH.data() + (size_t)m * H, hTmp.data() + (size_t)m * H,
            hPostN.data());
    hcvt(hHh.data(), hH.data(), MH);
    hgemm(mats[5], hHh.data(), hG17.data());
    hgemm(mats[6], hHh.data(), hU17.data());
    for (int j = 0; j < MI; ++j) {
      float g = hG17[j];
      hG17[j] = (g / (1.0f + expf(-g))) * hU17[j];
    }
    hcvt(hG17h.data(), hG17.data(), MI);
    hgemm(mats[7], hG17h.data(), hMix.data());
    for (int j = 0; j < MH; ++j)
      hX[j] = hTmp[j] + hMix[j];
  };

  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0;
  std::vector<float> outA(MH), outB(MH);
  // Chunk inputs are INDEPENDENT per chunk (prefill-faithful); only conv/SSM
  // states carry across chunks. zero_states clears states; load_chunk swaps
  // the X stream without touching states.
  auto zero_states = [&]() {
    std::fill(hCS.begin(), hCS.end(), 0.0f);
    std::fill(hS.begin(), hS.end(), 0.0f);
    CHECK(zeCommandListAppendMemoryCopy(up, dCS, hCS.data(), hCS.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dS, hS.data(), hS.size() * 4,
                                        nullptr, 0, nullptr));
  };
  auto load_chunk = [&](const std::vector<float> &xin) {
    hX = xin;
    CHECK(zeCommandListAppendMemoryCopy(up, dX, hX.data(), hX.size() * 4,
                                        nullptr, 0, nullptr));
  };
  auto run_list = [&](std::vector<double> *timing) {
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    if (timing)
      timing->push_back(now_ns() - t0);
    CHECK(zeCommandListAppendMemoryCopy(up, hRef.data(), dX, MH * 4, nullptr,
                                        0, nullptr));
  };
  auto check_full = [&](const char *tag) {
    double refmax = 0;
    for (float v : hX)
      refmax = std::max(refmax, (double)std::fabs(v));
    for (int j = 0; j < MH; ++j) {
      double rel = std::fabs((double)hRef[j] - (double)hX[j]) / refmax;
      if (rel > worstRel)
        worstRel = rel;
      if (rel > 1e-3) {
        ok = false;
        std::fprintf(stderr, "ref mismatch %s j %d: got %g want %g\n", tag, j,
                     hRef[j], hX[j]);
        break;
      }
    }
  };
  auto check_bits = [&](const std::vector<float> &want, const char *tag) {
    for (int j = 0; j < MH; ++j) {
      uint32_t a, b;
      __builtin_memcpy(&a, &hRef[j], 4);
      __builtin_memcpy(&b, &want[j], 4);
      if (a != b) {
        ok = false;
        std::fprintf(stderr, "reset mismatch %s j %d\n", tag, j);
        break;
      }
    }
  };
  // Run 1: chunk A then B (conv/SSM states carry; X inputs independent).
  zero_states();
  load_chunk(hXa0);
  layer_ref();
  run_list(&tRep);
  if (std::getenv("AINFER_DBG")) {
    struct SB {
      void *d;
      std::vector<float> *h;
      const char *tag;
    } stages[] = {{dH, &hH, "normH"},     {dQKV, &hQKV, "qkv"},
                      {dMxC, &hMxC, "mxC"},   {dAtt, &hAtt, "att"},
                      {dMix, &hMix, "mix"},   {dTmp, &hTmp, "tmp"},
                      {dG17, &hG17, "g17"},   {dZ, &hZ, "zemb"},
                      {dBt, &hBt, "btinv"},   {dMxR, &hMxR, "mxR"}};
    for (auto &st : stages) {
      std::vector<float> got(st.h->size());
      CHECK(zeCommandListAppendMemoryCopy(up, got.data(), st.d,
                                          got.size() * 4, nullptr, 0,
                                          nullptr));
      double mx = 0, me = 0;
      int nbad = 0;
      for (size_t j = 0; j < got.size(); ++j) {
        mx = std::max(mx, (double)std::fabs((*st.h)[j]));
        double e = (double)std::fabs(got[j] - (*st.h)[j]);
        me = std::max(me, e);
        if (e / (mx > 0 ? mx : 1) > 1e-3)
          ++nbad;
      }
      std::fprintf(stderr, "DBG %-5s maxrel %.3e nbad=%d\n", st.tag,
                   me / (mx > 0 ? mx : 1), nbad);
    }
  }
  check_full("A");
  if (ok)
    outA = hRef;
  load_chunk(hXb0);
  layer_ref();
  run_list(&tRep);
  check_full("B-after-A");
  if (ok)
    outB = hRef;
  // Continuity: B-alone from zeroed states must DIFFER (state is live).
  bool differs = false;
  if (ok) {
    zero_states();
    load_chunk(hXb0);
    run_list(nullptr);
    for (int j = 0; j < MH; j += 97)
      if (hRef[j] != outB[j]) {
        differs = true;
        break;
      }
    if (!differs) {
      ok = false;
      std::fprintf(stderr, "continuity failure\n");
    } else {
      std::printf("continuity: B-after-A differs from B-alone (state live)\n");
    }
  }
  // Run 2: reset + identical A,B -> bitwise identical.
  if (ok) {
    zero_states();
    load_chunk(hXa0);
    layer_ref();
    run_list(nullptr);
    check_bits(outA, "A");
  }
  if (ok) {
    load_chunk(hXb0);
    layer_ref();
    run_list(nullptr);
    check_bits(outB, "B");
  }

  double mR = med(tRep);
  std::printf("chunklayer med %.2f us/chunk worst-rel %.2e %s\n", mR / 1e3,
              worstRel, ok ? "CHUNKLAYER-OK" : "MISMATCH");
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"layer\":\"linear-attn chunked "
                "(synthetic)\",\"chunk\":%d,\"chunks\":2,"
                "\"replay_us\":%.2f,\"worst_rel\":%.2e,\"ref_tol\":1e-3,"
                "\"cross_chunk_continuity\":%s,\"reset_deterministic\":%s,"
                "\"chunklayer_ok\":%s}",
                M, mR / 1e3, worstRel, differs ? "true" : "false",
                ok ? "true" : "false", ok ? "true" : "false");
  const char *rp = (argc > 13) ? argv[13] : nullptr;
  if (rp) {
    FILE *o = std::fopen(rp, "w");
    if (o) {
      std::fprintf(o, "%s\n", js);
      std::fclose(o);
    }
  } else {
    std::printf("%s\n", js);
  }
  return ok ? 0 : 1;
}
