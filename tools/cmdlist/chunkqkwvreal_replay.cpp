// T7.4 chunk production: GEMM-form full-attention chunk on REAL weights.
// Same composition as chunkattnreal (M=32 chunk at prefix P0=64, trunk layer
// 3 INT4 spans, per-row norms, ChunkRope + ChunkKvAppend, o_proj + MLP all
// in-list) with the scan ChunkAttn swapped for GEMM-form stages: Q fp16
// convert + ChunkQkGemm (causal) + ChunkSoftmaxRow + W fp16 convert +
// ChunkWvGemm + GateMul. Host ref is implementation-independent attention
// math, unchanged. Checks: final hidden vs host float ref (tol 1e-3: fp16
// chunk path) + prefix-guard integrity + strict bitwise reset-determinism.
// Usage: chunkqkwvreal_replay <model.binfer> <13 spv...> [report.json]
//   order: norm cvtf32f16 chunkgemm splitqk batchnorm chunkrope chunkkvappend
//          chunkqkgemm chunksoftmaxrow chunkwvgemm gatemul silumul resaddf
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <fstream>
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
static inline float qb(float v) { return bf16_to_f32(f32_to_bf16(v)); }
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

static uint64_t prng = 0x9e3779b9;
static float frnd(float s) {
  prng = prng * 6364136223846793005ull + 1442695040888963407ull;
  return (float)((int)((prng >> 33) & 0xFFFF) - 32768) * (s / 32768.0f);
}

struct Entry {
  char name[64];
  uint64_t d_off, d_bytes, sc_off, sc_bytes;
  uint32_t crc;
};
static uint32_t rd32(std::ifstream &f) {
  uint32_t v;
  f.read((char *)&v, 4);
  return v;
}
static uint64_t rd64(std::ifstream &f) {
  uint64_t v;
  f.read((char *)&v, 8);
  return v;
}

struct Mat {
  const char *nm;
  int M, K, GG;
  void *P = nullptr, *S = nullptr;
  std::vector<uint8_t> packed;
  std::vector<uint16_t> scales;
};

int main(int argc, char **argv) {
  if (argc < 15) {
    std::fprintf(stderr, "usage: chunkqkwvreal_replay <model> <13 spv> "
                         "[report]\n");
    return 2;
  }
  const char *model = argv[1];
  const int H = 5120, QW = 12288, KVW = 1024, QN = 6144, I = 17408;
  // Geometry override for scaling experiments (ctest contract stays M=32):
  // CHUNK_M / CHUNK_P env, defaults 32/64. Buffers + launches are runtime-
  // sized, so any M/P runs unmodified (host ref cost grows as M*W*D).
  int M = 32, P0 = 64;
  if (const char *e = std::getenv("CHUNK_M")) {
    int v = std::atoi(e);
    if (v >= 8 && v <= 512)
      M = v;
  }
  if (const char *e = std::getenv("CHUNK_P")) {
    int v = std::atoi(e);
    if (v >= 16 && v <= 65536)
      P0 = v;
  }
  const int TC = P0 + M;
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
  const size_t MH = (size_t)M * H, MV = (size_t)M * QN, MI = (size_t)M * I;
  void *dX = alloc(MH * 4), *dH = alloc(MH * 4), *dTmp = alloc(MH * 4),
       *dMix = alloc(MH * 4), *dHh = alloc(MH * 2);
  void *dQKV = alloc((size_t)M * QW * 4), *dK16 = alloc((size_t)M * KVW * 4),
       *dV16 = alloc((size_t)M * KVW * 4), *dKn = alloc((size_t)M * KVW * 4);
  void *dQn = alloc(MV * 4), *dGate = alloc(MV * 4), *dAtt = alloc(MV * 4),
       *dAtth = alloc(MV * 2);
  void *dQnh = alloc(MV * 2), *dWsm = alloc((size_t)M * 24 * TC * 4),
       *dWsmh = alloc((size_t)M * 24 * TC * 2), *dCore = alloc(MV * 4);
  void *dMKc = alloc((size_t)4 * TC * 256 * 2),
       *dMVc = alloc((size_t)4 * TC * 256 * 2);
  void *dWts = alloc((size_t)M * 24 * TC * 4);
  void *dG17 = alloc(MI * 4), *dU17 = alloc(MI * 4), *dG17h = alloc(MI * 2);
  void *dCtrl = alloc(sizeof(DecodeControl));
  void *dCtrlSm = alloc(sizeof(DecodeControl));
  void *dInN = alloc(H * 4), *dPostN = alloc(H * 4), *dQNW = alloc(256 * 4),
       *dKNW = alloc(256 * 4), *dCos = alloc(64 * TC * 4),
       *dSin = alloc(64 * TC * 4);

  const char *entries[] = {
      "_ZTS8RMSNormW", "_ZTS9CvtF32F16", "_ZTS9ChunkGemm", "_ZTS7SplitQK",
      "_ZTS9BatchNorm", "_ZTS9ChunkRope", "_ZTS13ChunkKvAppend",
      "_ZTS11ChunkQkGemm", "_ZTS15ChunkSoftmaxRow", "_ZTS11ChunkWvGemm",
      "_ZTS7GateMul", "_ZTS7SiluMul", "_ZTS7ResAddF"};
  ze_kernel_handle_t kh[13] = {nullptr};
  std::vector<std::vector<uint8_t>> spvs;
  for (int i = 0; i < 13; ++i) {
    spvs.push_back(load_spv(argv[2 + i]));
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
    NORM, CVT, GEMM, SPLIT, BNORM, CROPE, CKV, CQK, CSM, CWV, GMUL, SILU, RES
  };
  for (int i = 0; i < 13; ++i)
    CHECK(zeKernelSetGroupSize(kh[i], 1, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[NORM], 256, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[GEMM], 16, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[CQK], 16, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[CWV], 16, 1, 1));

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
    if (zeCommandListAppendLaunchKernel(reg, k, &gc, nullptr, 0, nullptr) !=
            ZE_RESULT_SUCCESS ||
        zeCommandListAppendBarrier(reg, nullptr, 0, nullptr) !=
            ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "append failed\n");
      std::exit(1);
    }
  };

  // ---- real weights: trunk layer-3 spans ----
  std::ifstream f(model, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "no model %s\n", model);
    return 2;
  }
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "BINFER\x00\x01", 8) != 0)
    return 2;
  rd32(f);
  rd32(f);
  uint64_t n;
  f.read((char *)&n, 8);
  uint64_t table_off = rd64(f);
  f.seekg((std::streamoff)table_off);
  uint64_t dir_off = 0;
  for (int i = 0; i < 5; ++i) {
    uint32_t sid = rd32(f);
    uint64_t off = rd64(f), nb = rd64(f);
    rd32(f);
    f.seekg(8, std::ios::cur);
    if (sid == 5)
      dir_off = off;
    (void)nb;
  }
  f.seekg((std::streamoff)dir_off);
  std::vector<Entry> ents(n);
  for (uint64_t i = 0; i < n; ++i) {
    Entry &e = ents[i];
    f.read(e.name, 64);
    e.name[63] = 0;
    f.seekg(1 + 7 + 64 + 1 + 1 + 2 + 4, std::ios::cur);
    e.sc_off = rd64(f);
    e.sc_bytes = rd64(f);
    e.d_off = rd64(f);
    e.d_bytes = rd64(f);
    e.crc = rd32(f);
    f.seekg(12, std::ios::cur);
  }
  auto find = [&](const std::string &nm) -> const Entry * {
    for (auto &e : ents)
      if (nm == e.name)
        return &e;
    return nullptr;
  };
  Mat mats[] = {
      {"model.language_model.layers.3.self_attn.q_proj.weight", QW, H, 40},
      {"model.language_model.layers.3.self_attn.k_proj.weight", KVW, H, 40},
      {"model.language_model.layers.3.self_attn.v_proj.weight", KVW, H, 40},
      {"model.language_model.layers.3.self_attn.o_proj.weight", H, QN, 48},
      {"model.language_model.layers.3.mlp.gate_proj.weight", I, H, 40},
      {"model.language_model.layers.3.mlp.up_proj.weight", I, H, 40},
      {"model.language_model.layers.3.mlp.down_proj.weight", H, I, 136}};
  for (auto &mt : mats) {
    const Entry *e = find(mt.nm);
    if (!e) {
      std::fprintf(stderr, "missing %s\n", mt.nm);
      return 2;
    }
    mt.packed.assign(e->d_bytes, 0);
    f.clear();
    f.seekg((std::streamoff)e->d_off);
    f.read((char *)mt.packed.data(), e->d_bytes);
    mt.scales.assign(e->sc_bytes / 2, 0);
    f.clear();
    f.seekg((std::streamoff)e->sc_off);
    f.read((char *)mt.scales.data(), e->sc_bytes);
    mt.P = alloc(mt.packed.size());
    mt.S = alloc(mt.scales.size() * 2);
    CHECK(zeCommandListAppendMemoryCopy(up, mt.P, mt.packed.data(),
                                        mt.packed.size(), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, mt.S, mt.scales.data(),
                                        mt.scales.size() * 2, nullptr, 0,
                                        nullptr));
  }
  auto load_norm = [&](const char *nm, void *dst, int cnt,
                       std::vector<float> &host) {
    const Entry *e = find(nm);
    if (!e) {
      std::fprintf(stderr, "missing %s\n", nm);
      std::exit(2);
    }
    std::vector<uint16_t> raw(cnt);
    f.clear();
    f.seekg((std::streamoff)e->d_off);
    f.read((char *)raw.data(), cnt * 2);
    host.assign(cnt, 0);
    for (int i = 0; i < cnt; ++i)
      host[i] = bf16_to_f32(raw[i]);
    CHECK(zeCommandListAppendMemoryCopy(up, dst, host.data(), cnt * 4, nullptr,
                                        0, nullptr));
  };
  std::vector<float> hInN, hPostN, hQNW, hKNW, hCos, hSin;
  load_norm("model.language_model.layers.3.input_layernorm.weight", dInN, H,
            hInN);
  load_norm("model.language_model.layers.3.post_attention_layernorm.weight",
            dPostN, H, hPostN);
  load_norm("model.language_model.layers.3.self_attn.q_norm.weight", dQNW, 256,
            hQNW);
  load_norm("model.language_model.layers.3.self_attn.k_norm.weight", dKNW, 256,
            hKNW);
  hCos.assign(64 * TC, 0);
  hSin.assign(64 * TC, 0);
  for (int t = 0; t < TC; ++t)
    for (int i = 0; i < 64; ++i) {
      double inv = 1.0 / std::pow(10000000.0, (double)(2 * (i % 32)) / 64.0);
      double ang = (double)t * inv;
      hCos[t * 64 + i] = (float)std::cos(ang);
      hSin[t * 64 + i] = (float)std::sin(ang);
    }
  CHECK(zeCommandListAppendMemoryCopy(up, dCos, hCos.data(), hCos.size() * 4,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dSin, hSin.data(), hSin.size() * 4,
                                      nullptr, 0, nullptr));

  // ---- record one full-attention chunk (base = P0) ----
  int n256 = 256, tmax = TC, mmA = M, base = P0;
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
    launch(kh[GEMM], ((mm + 31) / 32) * (nn / 16));
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
  normYa(dH, dX, dInN);
  cvtYa(dHh, dH, M * H);
  cgemm(mats[0], dHh, dQKV);
  cgemm(mats[1], dHh, dK16);
  cgemm(mats[2], dHh, dV16);
  for (int m = 0; m < M; ++m) { // split + Q/K norms per row
    void *q16 = (char *)dQKV + (size_t)m * QW * 4;
    void *k16 = (char *)dK16 + (size_t)m * KVW * 4;
    void *qn = (char *)dQn + (size_t)m * QN * 4;
    void *gt = (char *)dGate + (size_t)m * QN * 4;
    void *kn = (char *)dKn + (size_t)m * KVW * 4;
    setarg(kh[SPLIT], 0, sizeof(void *), &q16);
    setarg(kh[SPLIT], 1, sizeof(void *), &qn);
    setarg(kh[SPLIT], 2, sizeof(void *), &gt);
    launch(kh[SPLIT], QN);
    setarg(kh[BNORM], 0, sizeof(void *), &qn);
    setarg(kh[BNORM], 1, sizeof(void *), &qn);
    setarg(kh[BNORM], 2, sizeof(void *), &dQNW);
    setarg(kh[BNORM], 3, sizeof(int), &n256);
    launch(kh[BNORM], 24);
    setarg(kh[BNORM], 0, sizeof(void *), &kn);
    setarg(kh[BNORM], 1, sizeof(void *), &k16);
    setarg(kh[BNORM], 2, sizeof(void *), &dKNW);
    setarg(kh[BNORM], 3, sizeof(int), &n256);
    launch(kh[BNORM], 4);
  }
  // ChunkRope: M rows at global positions base+m (Ctrl[1] = base).
  setarg(kh[CROPE], 0, sizeof(void *), &dQn);
  setarg(kh[CROPE], 1, sizeof(void *), &dKn);
  setarg(kh[CROPE], 2, sizeof(void *), &dCos);
  setarg(kh[CROPE], 3, sizeof(void *), &dSin);
  setarg(kh[CROPE], 4, sizeof(void *), &dCtrl);
  setarg(kh[CROPE], 5, sizeof(int), &tmax);
  setarg(kh[CROPE], 6, sizeof(int), &mmA);
  launch(kh[CROPE], M * 28);
  // ChunkKvAppend: M rows into the shared cache at base+m.
  setarg(kh[CKV], 0, sizeof(void *), &dMKc);
  setarg(kh[CKV], 1, sizeof(void *), &dMVc);
  setarg(kh[CKV], 2, sizeof(void *), &dKn);
  setarg(kh[CKV], 3, sizeof(void *), &dV16);
  setarg(kh[CKV], 4, sizeof(void *), &dCtrl);
  setarg(kh[CKV], 5, sizeof(int), &tmax);
  setarg(kh[CKV], 6, sizeof(int), &mmA);
  launch(kh[CKV], M * 1024);
  // GEMM-form attention over prefix + freshly appended chunk rows:
  // Q fp16 convert + causal QK-GEMM + row softmax + W convert + WV-GEMM +
  // gate multiply. Softmax reads its width from a dedicated control block
  // (active_length = TC) so the rope/append control stays untouched.
  cvtYa(dQnh, dQn, M * QN);
  {
    int pp = P0, ww = TC, ss = TC;
    setarg(kh[CQK], 0, sizeof(void *), &dQnh);
    setarg(kh[CQK], 1, sizeof(void *), &dMKc);
    setarg(kh[CQK], 2, sizeof(void *), &dWts);
    setarg(kh[CQK], 3, sizeof(int), &pp);
    setarg(kh[CQK], 4, sizeof(int), &mmA);
    setarg(kh[CQK], 5, sizeof(int), &ww);
    setarg(kh[CQK], 6, sizeof(int), &ss);
    setarg(kh[CQK], 7, (size_t)8 * 256 * 2, nullptr);
    setarg(kh[CQK], 8, (size_t)16 * 16 * 2, nullptr);
    setarg(kh[CQK], 9, (size_t)8 * 16 * 4, nullptr);
    launch(kh[CQK], (uint32_t)(M * 4 * ((TC + 15) / 16)));
  }
  {
    int rows = M * 24;
    setarg(kh[CSM], 0, sizeof(void *), &dWsm);
    setarg(kh[CSM], 1, sizeof(void *), &dWts);
    setarg(kh[CSM], 2, sizeof(void *), &dCtrlSm);
    setarg(kh[CSM], 3, sizeof(int), &tmax);
    setarg(kh[CSM], 4, sizeof(int), &rows);
    launch(kh[CSM], (uint32_t)rows);
  }
  cvtYa(dWsmh, dWsm, M * 24 * TC);
  {
    setarg(kh[CWV], 0, sizeof(void *), &dWsmh);
    setarg(kh[CWV], 1, sizeof(void *), &dMVc);
    setarg(kh[CWV], 2, sizeof(void *), &dCore);
    setarg(kh[CWV], 3, sizeof(int), &tmax);
    setarg(kh[CWV], 4, sizeof(int), &tmax);
    setarg(kh[CWV], 5, (size_t)8 * 16 * 2, nullptr);
    setarg(kh[CWV], 6, (size_t)16 * 16 * 2, nullptr);
    setarg(kh[CWV], 7, (size_t)8 * 16 * 4, nullptr);
    launch(kh[CWV], (uint32_t)(M * 64));
  }
  setarg(kh[GMUL], 0, sizeof(void *), &dAtt);
  setarg(kh[GMUL], 1, sizeof(void *), &dCore);
  setarg(kh[GMUL], 2, sizeof(void *), &dGate);
  launch(kh[GMUL], (uint32_t)(M * QN));
  cvtYa(dAtth, dAtt, M * QN);
  cgemm(mats[3], dAtth, dMix);
  resYa(dTmp, dX, dMix);
  normYa(dH, dTmp, dPostN);
  cvtYa(dHh, dH, M * H);
  cgemm(mats[4], dHh, dG17);
  cgemm(mats[5], dHh, dU17);
  setarg(kh[SILU], 0, sizeof(void *), &dG17);
  setarg(kh[SILU], 1, sizeof(void *), &dU17);
  setarg(kh[SILU], 2, sizeof(void *), &dG17);
  launch(kh[SILU], MI);
  cvtYa(dG17h, dG17, MI);
  cgemm(mats[6], dG17h, dMix);
  resYa(dX, dTmp, dMix);
  ze_fence_handle_t fence = nullptr;
  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK(zeFenceCreate(qq, &fdesc, &fence));
  CHECK(zeCommandListClose(reg));
  (void)n256;
  (void)tmax;
  (void)mmA;
  (void)base;

  // ---- host reference (float, dequant on the fly, qb caches) ----
  std::vector<float> hX(MH), hH(MH), hTmp(MH), hMix(MH), hQKV((size_t)M * QW),
      hK16((size_t)M * KVW), hV16((size_t)M * KVW), hKn((size_t)M * KVW),
      hQn(MV), hGate(MV), hAtt(MV),
      hMKc((size_t)4 * TC * 256, 0.0f), hMVc((size_t)4 * TC * 256, 0.0f),
      hG17(MI), hU17(MI), hXa(MH), hRef(MH);
  // Fixed inputs reused by both runs (determinism needs identical inputs).
  for (auto &v : hXa)
    v = frnd(1.0f);
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
    for (int j = 0; j < nn; ++j)
      O[j] = f32_to_f16(X[j]);
  };
  std::vector<float> wts(TC);
  auto layer_ref = [&](int chunk_base) {
    for (int m = 0; m < M; ++m)
      hnorm(hH.data() + (size_t)m * H, hX.data() + (size_t)m * H,
            hInN.data());
    hcvt(hHh.data(), hH.data(), MH);
    hgemm(mats[0], hHh.data(), hQKV.data());
    hgemm(mats[1], hHh.data(), hK16.data());
    hgemm(mats[2], hHh.data(), hV16.data());
    for (int m = 0; m < M; ++m) {
      int pos = chunk_base + m;
      for (int i = 0; i < QN; ++i) {
        int hh = i / 256, d = i % 256;
        hQn[(size_t)m * QN + i] = hQKV[(size_t)m * QW + hh * 512 + d];
        hGate[(size_t)m * QN + i] = hQKV[(size_t)m * QW + hh * 512 + 256 + d];
      }
      for (int r = 0; r < 24; ++r) {
        double ss = 0;
        for (int j = 0; j < 256; ++j) {
          float v = hQn[(size_t)m * QN + r * 256 + j];
          ss += (double)v * v;
        }
        float inv = 1.0f / std::sqrt((float)(ss / 256) + 1e-6f);
        for (int j = 0; j < 256; ++j)
          hQn[(size_t)m * QN + r * 256 + j] *= inv * (1.0f + hQNW[j]);
      }
      for (int r = 0; r < 4; ++r) {
        double ss = 0;
        for (int j = 0; j < 256; ++j) {
          float v = hK16[(size_t)m * KVW + r * 256 + j];
          ss += (double)v * v;
        }
        float inv = 1.0f / std::sqrt((float)(ss / 256) + 1e-6f);
        for (int j = 0; j < 256; ++j)
          hKn[(size_t)m * KVW + r * 256 + j] =
              hK16[(size_t)m * KVW + r * 256 + j] * inv * (1.0f + hKNW[j]);
      }
      // rope at global pos
      for (int i = 0; i < 28; ++i) {
        float *X = i < 24 ? hQn.data() + (size_t)m * QN + (size_t)i * 256
                          : hKn.data() + (size_t)m * KVW + (size_t)(i - 24) * 256;
        for (int d = 0; d < 32; ++d) {
          float x0 = X[d], x1 = X[d + 32];
          float c = hCos[pos * 64 + d], s = hSin[pos * 64 + d];
          X[d] = x0 * c - x1 * s;
          X[d + 32] = x0 * s + x1 * c;
        }
      }
      // kv append at global pos (qb stores)
      for (int i = 0; i < KVW; ++i) {
        int hh = i / 256, d = i % 256;
        hMKc[((size_t)pos * 4 + hh) * 256 + d] =
            qb(hKn[(size_t)m * KVW + hh * 256 + d]);
        hMVc[((size_t)pos * 4 + hh) * 256 + d] =
            qb(hV16[(size_t)m * KVW + hh * 256 + d]);
      }
      // causal scan over 0..pos
      int T = pos + 1;
      for (int hh = 0; hh < 24; ++hh) {
        int kv = hh / 6;
        float mx = -1e30f;
        for (int t = 0; t < T; ++t) {
          float sc = 0;
          for (int d = 0; d < 256; ++d)
            sc += hQn[(size_t)m * QN + hh * 256 + d] *
                  hMKc[((size_t)t * 4 + kv) * 256 + d];
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
            acc += wts[t] / se * hMVc[((size_t)t * 4 + kv) * 256 + d];
          float g = hGate[(size_t)m * QN + hh * 256 + d];
          hAtt[(size_t)m * QN + hh * 256 + d] = acc / (1.0f + expf(-g));
        }
      }
    }
    hcvt(hAtth.data(), hAtt.data(), MV);
    hgemm(mats[3], hAtth.data(), hMix.data());
    for (int j = 0; j < (int)MH; ++j)
      hTmp[j] = hX[j] + hMix[j];
    for (int m = 0; m < M; ++m)
      hnorm(hH.data() + (size_t)m * H, hTmp.data() + (size_t)m * H,
            hPostN.data());
    hcvt(hHh.data(), hH.data(), MH);
    hgemm(mats[4], hHh.data(), hG17.data());
    hgemm(mats[5], hHh.data(), hU17.data());
    for (int j = 0; j < (int)MI; ++j) {
      float g = hG17[j];
      hG17[j] = (g / (1.0f + expf(-g))) * hU17[j];
    }
    hcvt(hG17h.data(), hG17.data(), MI);
    hgemm(mats[6], hG17h.data(), hMix.data());
    for (int j = 0; j < (int)MH; ++j)
      hX[j] = hTmp[j] + hMix[j];
  };

  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0;
  std::vector<float> seqOut;
  auto upload_inputs = [&](const std::vector<float> &xs) {
    hX = xs;
    CHECK(zeCommandListAppendMemoryCopy(up, dX, hX.data(), MH * 4, nullptr, 0,
                                        nullptr));
  };
  // Fixed prefix reused by both runs (same determinism rule as inputs).
  std::vector<float> seqPK((size_t)P0 * 4 * 256), seqPV((size_t)P0 * 4 * 256);
  for (size_t i = 0; i < seqPK.size(); ++i) {
    seqPK[i] = qb(frnd(0.5f));
    seqPV[i] = qb(frnd(0.5f));
  }
  auto fill_prefix = [&]() {
    // Prefix [0, P0): fixed qb values directly (fill path proven elsewhere).
    for (int t = 0; t < P0; ++t)
      for (int i = 0; i < 4 * 256; ++i) {
        hMKc[(size_t)t * 4 * 256 + i] = seqPK[(size_t)t * 4 * 256 + i];
        hMVc[(size_t)t * 4 * 256 + i] = seqPV[(size_t)t * 4 * 256 + i];
      }
    // Chunk slots zeroed (append writes them); guard slot TC-1 stays zero.
    for (int t = P0; t < TC; ++t)
      for (int i = 0; i < 4 * 256; ++i) {
        hMKc[(size_t)t * 4 * 256 + i] = 0.0f;
        hMVc[(size_t)t * 4 * 256 + i] = 0.0f;
      }
    std::vector<uint16_t> kc(hMKc.size()), vc(hMVc.size());
    for (size_t i = 0; i < kc.size(); ++i) {
      kc[i] = f32_to_bf16(hMKc[i]);
      vc[i] = f32_to_bf16(hMVc[i]);
    }
    CHECK(zeCommandListAppendMemoryCopy(up, dMKc, kc.data(), kc.size() * 2,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dMVc, vc.data(), vc.size() * 2,
                                        nullptr, 0, nullptr));
  };
  {
    DecodeControl csm{0, 0, TC, 0};
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrlSm, &csm, sizeof(csm),
                                        nullptr, 0, nullptr));
  }
  for (int run = 0; run < 2 && ok; ++run) {
    fill_prefix();
    upload_inputs(hXa);
    hX = hXa;
    DecodeControl c{9000 + run, P0, P0 + 1, -1};
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                        nullptr));
    layer_ref(P0);
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    if (run == 0)
      tRep.push_back(now_ns() - t0);
    CHECK(zeCommandListAppendMemoryCopy(up, hRef.data(), dX, MH * 4, nullptr, 0,
                                        nullptr));
    if (run == 0) {
      double refmax = 0;
      for (float v : hX)
        refmax = std::max(refmax, (double)std::fabs(v));
      for (size_t j = 0; j < hX.size(); ++j) {
        double rel = std::fabs((double)hRef[j] - (double)hX[j]) /
                     (refmax > 0 ? refmax : 1);
        if (rel > worstRel)
          worstRel = rel;
        if (rel > 1e-3) {
          ok = false;
          std::fprintf(stderr, "ref mismatch j %zu: got %g want %g\n", j,
                       hRef[j], hX[j]);
          break;
        }
      }
      if (!ok)
        break;
      seqOut = hRef;
      // Guard: prefix slot 3 must still hold its fill exactly (chunk appends
      // at 64..95 must not disturb prefix or stride into it).
      std::vector<uint16_t> guard(4 * 256);
      CHECK(zeCommandListAppendMemoryCopy(
          up, guard.data(), (char *)dMKc + (size_t)3 * 4 * 256 * 2,
          guard.size() * 2, nullptr, 0, nullptr));
      for (size_t j = 0; j < guard.size(); ++j) {
        uint16_t want = f32_to_bf16(seqPK[(size_t)3 * 4 * 256 + j]);
        if (guard[j] != want) {
          ok = false;
          std::fprintf(stderr, "guard mismatch j %zu: got %04x want %04x\n", j,
                       guard[j], want);
          break;
        }
      }
      if (!ok)
        break;
    } else {
      for (size_t j = 0; j < hX.size(); ++j) {
        uint32_t a, b;
        __builtin_memcpy(&a, &hRef[j], 4);
        __builtin_memcpy(&b, &seqOut[j], 4);
        if (a != b) {
          ok = false;
          std::fprintf(stderr, "reset mismatch j %zu\n", j);
          break;
        }
      }
      if (!ok)
        break;
    }
  }
  double mR = med(tRep);
  std::printf("chunkqkwvreal med %.2f us/chunk worst-rel %.2e %s\n", mR / 1e3,
              worstRel, ok ? "CHUNKQKWVREAL-OK" : "MISMATCH");
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"layer\":\"full-attn chunked GEMM-form "
                "(real weights, layer 3)\",\"chunk\":%d,\"prefix\":%d,"
                "\"runs\":2,\"replay_us\":%.2f,\"worst_rel\":%.2e,"
                "\"ref_tol\":1e-3,\"guards_zero\":%s,"
                "\"reset_deterministic\":%s,\"chunkqkwvreal_ok\":%s}",
                M, P0, mR / 1e3, worstRel, ok ? "true" : "false",
                ok ? "true" : "false", ok ? "true" : "false");
  const char *rp = (argc > 15) ? argv[15] : nullptr;
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
