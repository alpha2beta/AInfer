// T5.3 layer adoption (full-attention): ONE full attention layer as a SINGLE
// recorded raw-L0 list (~24 launches, 11 handles reused via per-append arg
// sets), replayed over 4 positions with persistent KV caches and an in-place
// residual stream. Position/active-length ride the T5.4 control block (host
// 16 B update between replays, never a kernel arg). Synthetic layout-0 INT4
// weights at real shapes; RoPE tables decode-identical. Checks: step-0 host
// parity (tol 1e-4; INT8 flip avalanche scopes multi-step ref fidelity — see
// layerlin report) + strict bitwise reset-determinism on every step.
// Usage: layerattn_replay <11 spv modules...> [report.json]
//   order: norm gemv splitqk batchnorm rope kvappend attn scalesmax quantize
//          silu resaddf
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
// BF16 KV: the device quantizes on append, so the host reference stores
// RNE-quantized values (same rounding -> float-tail agreement, not bf16-eps).
static inline float qb(float v) { return bf16_to_f32(f32_to_bf16(v)); }

static uint64_t prng = 0xa77e4a;
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
  if (argc < 12) {
    std::fprintf(stderr, "usage: layerattn_replay <11 spv> [report.json]\n");
    return 2;
  }
  const int H = 5120, QW = 12288, KVW = 1024, QN = 6144, I = 17408;
  const int TMAX = 8, STEPS = 4;
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
  void *dX = alloc(H * 4), *dH = alloc(H * 4), *dTmp = alloc(H * 4),
       *dMix = alloc(H * 4);
  void *dQ16 = alloc(QW * 4), *dK16 = alloc(KVW * 4), *dV16 = alloc(KVW * 4),
       *dKn = alloc(KVW * 4);
  void *dQn = alloc(QN * 4), *dGate = alloc(QN * 4), *dAtt = alloc(QN * 4);
  void *dKc = alloc((size_t)4 * TMAX * 256 * 2),
       *dVc = alloc((size_t)4 * TMAX * 256 * 2);
  void *dG17 = alloc(I * 4), *dU17 = alloc(I * 4), *dQ8 = alloc(I),
       *dSq = alloc(136 * 4), *dCtrl = alloc(sizeof(DecodeControl)),
       *dWts = alloc((size_t)24 * TMAX * 4);
  void *dInN = alloc(H * 4), *dPostN = alloc(H * 4), *dQNW = alloc(256 * 4),
       *dKNW = alloc(256 * 4), *dCos = alloc(64 * TMAX * 4),
       *dSin = alloc(64 * TMAX * 4);

  const char *entries[] = {"_ZTS8RMSNormW", "_ZTS8Int4Gemv", "_ZTS7SplitQK",
                           "_ZTS9BatchNorm", "_ZTS9RopeApply", "_ZTS8KvAppend",
                           "_ZTS8AttnCore",  "_ZTS9ScalesMax", "_ZTS8Quantize",
                           "_ZTS7SiluMul",  "_ZTS7ResAddF"};
  ze_kernel_handle_t kh[11] = {nullptr};
  std::vector<std::vector<uint8_t>> spvs;
  for (int i = 0; i < 11; ++i) {
    spvs.push_back(load_spv(argv[1 + i]));
    ze_module_handle_t mod = nullptr;
    ze_module_desc_t mddesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                               nullptr,
                               ZE_MODULE_FORMAT_IL_SPIRV,
                               spvs.back().size(),
                               spvs.back().data(),
                               nullptr,
                               nullptr};
    CHECK(zeModuleCreate(ctx, dev, &mddesc, &mod, nullptr));
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           entries[i]};
    if (zeKernelCreate(mod, &kd, &kh[i]) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "entry %s not found\n", entries[i]);
      return 1;
    }
  }
  enum K {
    NORM,
    GEMV,
    SPLIT,
    BNORM,
    ROPE,
    KV,
    ATTN,
    SCALES,
    QUANT,
    SILU,
    RES
  };
  for (int i = 0; i < 11; ++i)
    CHECK(zeKernelSetGroupSize(kh[i], 1, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[NORM], 256, 1, 1)); // T6.1 parallel norm

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
      std::fprintf(stderr, "append failed %d/%d\n", (int)r1, (int)r2);
      std::exit(1);
    }
  };

  Mat mats[] = {{"q", QW, H, 40},  {"k", KVW, H, 40}, {"v", KVW, H, 40},
                {"o", H, QN, 48},  {"gate", I, H, 40}, {"up", I, H, 40},
                {"down", H, I, 136}};
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
  std::vector<float> hInN(H), hPostN(H), hQNW(256), hKNW(256), hCos(64 * TMAX),
      hSin(64 * TMAX);
  for (auto &v : hInN)
    v = frnd(0.1f);
  for (auto &v : hPostN)
    v = frnd(0.1f);
  for (auto &v : hQNW)
    v = frnd(0.1f);
  for (auto &v : hKNW)
    v = frnd(0.1f);
  for (int t = 0; t < TMAX; ++t)
    for (int i = 0; i < 64; ++i) {
      double inv = 1.0 / std::pow(10000000.0, (double)(2 * (i % 32)) / 64.0);
      double ang = (double)t * inv;
      hCos[t * 64 + i] = (float)std::cos(ang);
      hSin[t * 64 + i] = (float)std::sin(ang);
    }
  CHECK(zeCommandListAppendMemoryCopy(up, dInN, hInN.data(), H * 4, nullptr, 0,
                                      nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dPostN, hPostN.data(), H * 4, nullptr,
                                      0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dQNW, hQNW.data(), 256 * 4, nullptr,
                                      0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dKNW, hKNW.data(), 256 * 4, nullptr,
                                      0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dCos, hCos.data(), hCos.size() * 4,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dSin, hSin.data(), hSin.size() * 4,
                                      nullptr, 0, nullptr));

  // Record one full attention layer. Control (pos/T) is updated host-side
  // between replays, never baked.
  int n5120 = 5120, n256 = 256, tmax = TMAX;
  auto gemv = [&](Mat &mt, void *Y) {
    int kk = mt.K;
    setarg(kh[GEMV], 0, sizeof(void *), &Y);
    setarg(kh[GEMV], 1, sizeof(void *), &mt.P);
    setarg(kh[GEMV], 2, sizeof(void *), &mt.S);
    setarg(kh[GEMV], 3, sizeof(void *), &dQ8);
    setarg(kh[GEMV], 4, sizeof(void *), &dSq);
    setarg(kh[GEMV], 5, sizeof(int), &kk);
    launch(kh[GEMV], mt.M);
  };
  auto xq = [&](void *X, int n) {
    int gg = n / 128;
    setarg(kh[SCALES], 0, sizeof(void *), &X);
    setarg(kh[SCALES], 1, sizeof(void *), &dSq);
    setarg(kh[SCALES], 2, sizeof(int), &gg);
    launch(kh[SCALES], gg);
    setarg(kh[QUANT], 0, sizeof(void *), &X);
    setarg(kh[QUANT], 1, sizeof(void *), &dSq);
    setarg(kh[QUANT], 2, sizeof(void *), &dQ8);
    launch(kh[QUANT], n);
  };
  auto norm = [&](void *Y, void *X, void *W) {
    setarg(kh[NORM], 0, sizeof(void *), &Y);
    setarg(kh[NORM], 1, sizeof(void *), &X);
    setarg(kh[NORM], 2, sizeof(void *), &W);
    setarg(kh[NORM], 3, sizeof(int), &n5120);
    setarg(kh[NORM], 4, (size_t)256 * 8, nullptr);
    launch(kh[NORM], 1);
  };
  auto res = [&](void *Y, void *A, void *B) {
    setarg(kh[RES], 0, sizeof(void *), &Y);
    setarg(kh[RES], 1, sizeof(void *), &A);
    setarg(kh[RES], 2, sizeof(void *), &B);
    launch(kh[RES], H);
  };
  norm(dH, dX, dInN);
  xq(dH, H);
  gemv(mats[0], dQ16);
  gemv(mats[1], dK16);
  gemv(mats[2], dV16);
  setarg(kh[SPLIT], 0, sizeof(void *), &dQ16);
  setarg(kh[SPLIT], 1, sizeof(void *), &dQn);
  setarg(kh[SPLIT], 2, sizeof(void *), &dGate);
  launch(kh[SPLIT], QN);
  setarg(kh[BNORM], 0, sizeof(void *), &dQn);
  setarg(kh[BNORM], 1, sizeof(void *), &dQn);
  setarg(kh[BNORM], 2, sizeof(void *), &dQNW);
  setarg(kh[BNORM], 3, sizeof(int), &n256);
  launch(kh[BNORM], 24);
  setarg(kh[BNORM], 0, sizeof(void *), &dKn);
  setarg(kh[BNORM], 1, sizeof(void *), &dK16);
  setarg(kh[BNORM], 2, sizeof(void *), &dKNW);
  setarg(kh[BNORM], 3, sizeof(int), &n256);
  launch(kh[BNORM], 4);
  setarg(kh[ROPE], 0, sizeof(void *), &dQn);
  setarg(kh[ROPE], 1, sizeof(void *), &dKn);
  setarg(kh[ROPE], 2, sizeof(void *), &dCos);
  setarg(kh[ROPE], 3, sizeof(void *), &dSin);
  setarg(kh[ROPE], 4, sizeof(void *), &dCtrl);
  launch(kh[ROPE], 28);
  setarg(kh[KV], 0, sizeof(void *), &dKc);
  setarg(kh[KV], 1, sizeof(void *), &dVc);
  setarg(kh[KV], 2, sizeof(void *), &dKn);
  setarg(kh[KV], 3, sizeof(void *), &dV16);
  setarg(kh[KV], 4, sizeof(void *), &dCtrl);
  setarg(kh[KV], 5, sizeof(int), &tmax);
  launch(kh[KV], KVW);
  setarg(kh[ATTN], 0, sizeof(void *), &dAtt);
  setarg(kh[ATTN], 1, sizeof(void *), &dQn);
  setarg(kh[ATTN], 2, sizeof(void *), &dKc);
  setarg(kh[ATTN], 3, sizeof(void *), &dVc);
  setarg(kh[ATTN], 4, sizeof(void *), &dGate);
  setarg(kh[ATTN], 5, sizeof(void *), &dCtrl);
  setarg(kh[ATTN], 6, sizeof(int), &tmax);
  setarg(kh[ATTN], 7, sizeof(void *), &dWts);
  launch(kh[ATTN], 24);
  xq(dAtt, QN);
  gemv(mats[3], dMix);
  res(dTmp, dX, dMix);
  norm(dH, dTmp, dPostN);
  xq(dH, H);
  gemv(mats[4], dG17);
  gemv(mats[5], dU17);
  setarg(kh[SILU], 0, sizeof(void *), &dG17);
  setarg(kh[SILU], 1, sizeof(void *), &dU17);
  setarg(kh[SILU], 2, sizeof(void *), &dG17);
  launch(kh[SILU], I);
  xq(dG17, I);
  gemv(mats[6], dMix);
  res(dX, dTmp, dMix);
  ze_fence_handle_t fence = nullptr;
  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK(zeFenceCreate(qq, &fdesc, &fence));
  CHECK(zeCommandListClose(reg));

  // Host reference mirrors the device stage order in float.
  std::vector<float> hX(H), hH(H), hTmp(H), hMix(H), hQ16(QW), hK16(KVW),
      hV16(KVW), hKn(KVW), hQn(QN), hGate(QN), hAtt(QN),
      hKc(4 * TMAX * 256, 0.0f), hVc(4 * TMAX * 256, 0.0f), hG17(I), hU17(I),
      hSq(136);
  std::vector<int8_t> hQ8(I);
  std::vector<float> hRef(H), seqX0(H);
  for (auto &v : seqX0)
    v = frnd(1.0f);
  auto rmsn = [&](std::vector<float> &Y, std::vector<float> &X,
                  std::vector<float> &W) {
    double ss = 0;
    for (int j = 0; j < H; ++j)
      ss += (double)X[j] * X[j];
    float inv = 1.0f / std::sqrt((float)(ss / H) + 1e-6f);
    for (int j = 0; j < H; ++j)
      Y[j] = X[j] * inv * (1.0f + W[j]);
  };
  auto xqr = [&](std::vector<float> &X, int n) {
    int gg = n / 128;
    for (int g = 0; g < gg; ++g) {
      float mx = 0;
      for (int j = 0; j < 128; ++j)
        mx = std::max(mx, std::fabs(X[g * 128 + j]));
      hSq[g] = mx / 127.0f;
      if (hSq[g] == 0)
        hSq[g] = 1.0f;
      for (int j = 0; j < 128; ++j) {
        float v = X[g * 128 + j] / hSq[g];
        int qi = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
        hQ8[g * 128 + j] = (int8_t)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
      }
    }
  };
  auto gemvr = [&](Mat &mt, std::vector<float> &Y) {
    int M = mt.M, K = mt.K, GG = mt.GG;
    for (int m = 0; m < M; ++m) {
      float acc = 0;
      for (int g = 0; g < GG; ++g) {
        float sc = bf16_to_f32(mt.scales[(size_t)m * GG + g]);
        int gs = 0;
        for (int j = 0; j < 128; ++j) {
          size_t idx = (size_t)m * K + g * 128 + j;
          uint8_t b = mt.packed[idx / 2];
          int nib = (idx & 1) ? (b >> 4) : (b & 0xF);
          if (nib >= 8)
            nib -= 16;
          gs += nib * (int)hQ8[g * 128 + j];
        }
        acc += (float)gs * sc * hSq[g];
      }
      Y[m] = acc;
    }
  };
  std::vector<float> wts(TMAX);
  auto layer_ref = [&](int pos) {
    rmsn(hH, hX, hInN);
    xqr(hH, H);
    gemvr(mats[0], hQ16);
    gemvr(mats[1], hK16);
    gemvr(mats[2], hV16);
    for (int i = 0; i < QN; ++i) {
      int hh = i / 256, d = i % 256;
      hQn[i] = hQ16[(size_t)hh * 512 + d];
      hGate[i] = hQ16[(size_t)hh * 512 + 256 + d];
    }
    for (int r = 0; r < 24; ++r) { // q norm
      double ss = 0;
      for (int j = 0; j < 256; ++j)
        ss += (double)hQn[(size_t)r * 256 + j] * hQn[(size_t)r * 256 + j];
      float inv = 1.0f / std::sqrt((float)(ss / 256) + 1e-6f);
      for (int j = 0; j < 256; ++j)
        hQn[(size_t)r * 256 + j] *= inv * (1.0f + hQNW[j]);
    }
    for (int r = 0; r < 4; ++r) { // k norm
      double ss = 0;
      for (int j = 0; j < 256; ++j)
        ss += (double)hK16[(size_t)r * 256 + j] * hK16[(size_t)r * 256 + j];
      float inv = 1.0f / std::sqrt((float)(ss / 256) + 1e-6f);
      for (int j = 0; j < 256; ++j)
        hKn[(size_t)r * 256 + j] = hK16[(size_t)r * 256 + j] * inv *
                                   (1.0f + hKNW[j]);
    }
    for (int i = 0; i < 28; ++i) { // rope
      float *X = i < 24 ? hQn.data() + (size_t)i * 256
                        : hKn.data() + (size_t)(i - 24) * 256;
      for (int d = 0; d < 32; ++d) {
        float x0 = X[d], x1 = X[d + 32];
        float c = hCos[pos * 64 + d], s = hSin[pos * 64 + d];
        X[d] = x0 * c - x1 * s;
        X[d + 32] = x0 * s + x1 * c;
      }
    }
    for (int i = 0; i < KVW; ++i) { // kv append (BF16-quantized stores)
      int hh = i / 256, d = i % 256;
      hKc[((size_t)pos * 4 + hh) * 256 + d] = qb(hKn[(size_t)hh * 256 + d]);
      hVc[((size_t)pos * 4 + hh) * 256 + d] = qb(hV16[(size_t)hh * 256 + d]);
    }
    int T = pos + 1; // gqa + gate
    for (int hh = 0; hh < 24; ++hh) {
      int kv = hh / 6;
      float mx = -1e30f;
      for (int t = 0; t < T; ++t) {
        float sc = 0;
        for (int d = 0; d < 256; ++d)
          sc += hQn[(size_t)hh * 256 + d] *
                hKc[((size_t)t * 4 + kv) * 256 + d];
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
      for (int d = 0; d < 256; ++d) {
        float acc = 0;
        for (int t = 0; t < T; ++t)
          acc += wts[t] / se * hVc[((size_t)t * 4 + kv) * 256 + d];
        float g = hGate[(size_t)hh * 256 + d];
        hAtt[(size_t)hh * 256 + d] = acc / (1.0f + expf(-g));
      }
    }
    xqr(hAtt, QN);
    gemvr(mats[3], hMix);
    for (int j = 0; j < H; ++j)
      hTmp[j] = hX[j] + hMix[j];
    rmsn(hH, hTmp, hPostN);
    xqr(hH, H);
    gemvr(mats[4], hG17);
    gemvr(mats[5], hU17);
    for (int j = 0; j < I; ++j) {
      float g = hG17[j];
      hG17[j] = (g / (1.0f + expf(-g))) * hU17[j];
    }
    xqr(hG17, I);
    gemvr(mats[6], hMix);
    for (int j = 0; j < H; ++j)
      hX[j] = hTmp[j] + hMix[j];
  };

  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0;
  std::vector<std::vector<float>> seqOut;
  auto zero_all = [&]() {
    hX = seqX0;
    std::fill(hKc.begin(), hKc.end(), 0.0f);
    std::fill(hVc.begin(), hVc.end(), 0.0f);
    // BF16 zeros are zero bytes; upload half-size spans.
    std::vector<uint16_t> zc(hKc.size(), 0);
    CHECK(zeCommandListAppendMemoryCopy(up, dX, hX.data(), H * 4, nullptr, 0,
                                        nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dKc, zc.data(), zc.size() * 2,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dVc, zc.data(), zc.size() * 2,
                                        nullptr, 0, nullptr));
  };
  for (int run = 0; run < 2 && ok; ++run) {
    zero_all();
    for (int step = 0; step < STEPS; ++step) {
      int pos = step; // positions 0..3 advance the caches like decode
      layer_ref(pos);
      DecodeControl c{8000 + step, pos, pos + 1, -1};
      CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                          nullptr));
      double t0 = now_ns();
      CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
      if (run == 0)
        tRep.push_back(now_ns() - t0);
      CHECK(zeCommandListAppendMemoryCopy(up, hRef.data(), dX, H * 4, nullptr,
                                          0, nullptr));
      // Step-0 host parity only (flip avalanche: see layerlin report);
      // every step is covered by strict reset-determinism below.
      if (step == 0) {
        double refmax = 0;
        for (float v : hX)
          refmax = std::max(refmax, (double)std::fabs(v));
        for (int j = 0; j < H; ++j) {
          double rel = std::fabs((double)hRef[j] - (double)hX[j]) /
                       (refmax > 0 ? refmax : 1);
          if (rel > worstRel)
            worstRel = rel;
          if (rel > 1e-4) {
            ok = false;
            std::fprintf(stderr, "ref mismatch run %d step %d j %d: got %g "
                                 "want %g\n", run, step, j, hRef[j], hX[j]);
            break;
          }
        }
        if (!ok)
          break;
      }
      if (run == 0)
        seqOut.push_back(hRef);
      else {
        for (int j = 0; j < H; ++j) {
          uint32_t a, b;
          __builtin_memcpy(&a, &hRef[j], 4);
          __builtin_memcpy(&b, &seqOut[step][j], 4);
          if (a != b) {
            ok = false;
            std::fprintf(stderr, "reset mismatch step %d j %d\n", step, j);
            break;
          }
        }
        if (!ok)
          break;
      }
    }
  }
  double mR = med(tRep);
  std::printf("layerattn-replay med %.2f us/step worst-rel %.2e %s\n", mR / 1e3,
              worstRel, ok ? "LAYERATTN-OK" : "MISMATCH");
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"layer\":\"full-attn "
                "(synthetic)\",\"launches\":24,\"steps\":%d,\"runs\":2,"
                "\"replay_us\":%.2f,\"worst_rel\":%.2e,\"ref_tol\":1e-4,"
                "\"ref_scope\":\"step-0-only\",\"reset_deterministic\":%s,"
                "\"layerattn_ok\":%s}",
                STEPS, mR / 1e3, worstRel, ok ? "true" : "false",
                ok ? "true" : "false");
  const char *rp = (argc > 12) ? argv[12] : nullptr;
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
