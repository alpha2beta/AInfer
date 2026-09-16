// T7.4 chunk production: FULL 64-layer chunked prefill on REAL weights.
// One M=32 chunk at prefix P=0 (single-chunk proof; multi-chunk + handoff
// are the follow-ons): layers L%4==3 use the GEMM-form full block
// (chunkqkwvreal composition), all others the chunked linear block
// (chunklayerreal composition). Hidden state ping-pongs through shared
// scratch; per-layer state (SSM conv-history + recur state x48, KV caches
// x16) is dedicated per layer and zero-filled device-side. Weights stream
// per layer into max-sized buffers (31 MB/layer). Checks: 2-run bitwise
// reset-determinism + final-hidden dump for the python float ref
// (tools/forward/ref_chunk64.py vs fwd_cpu INT4-dequant chain).
// Usage: chunk64real_replay <model.binfer> <20 spv...> [report.json]
//   order: norm chunkgemm chunkssmconv chunkssmrecur silumul splitrepeat
//   l2normqk betag rmsinv normgated resaddf cvtf32f16 splitqk batchnorm
//   chunkrope chunkkvappend chunkqkgemm chunksoftmaxrow chunkwvgemm gatemul
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

static uint64_t prng = 0x64de17;
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

int main(int argc, char **argv) {
  if (argc < 22) {
    std::fprintf(stderr, "usage: chunk64real_replay <model> <20 spv> "
                         "[report]\n");
    return 2;
  }
  const char *model = argv[1];
  const int H = 5120, C = 10240, V6 = 6144, I = 17408, NH = 48, DL = 128;
  const int QW = 12288, KVW = 1024, QN = 6144, NQ = 24;
  const int M = 32, TC = M;
  const size_t MH = (size_t)M * H, MC = (size_t)M * C, MV = (size_t)M * V6,
               MI = (size_t)M * I, MH48 = (size_t)M * NH, MVQ = (size_t)M * QN;
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
      "_ZTS8RMSNormW", "_ZTS9ChunkGemm", "_ZTS12ChunkSsmConv",
      "_ZTS13ChunkSsmRecur", "_ZTS7SiluMul", "_ZTS11SplitRepeat",
      "_ZTS8L2NormQK", "_ZTS5BetaG", "_ZTS6RmsInv", "_ZTS9NormGated",
      "_ZTS7ResAddF", "_ZTS9CvtF32F16", "_ZTS7SplitQK", "_ZTS9BatchNorm",
      "_ZTS9ChunkRope", "_ZTS13ChunkKvAppend", "_ZTS11ChunkQkGemm",
      "_ZTS15ChunkSoftmaxRow", "_ZTS11ChunkWvGemm", "_ZTS7GateMul"};

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
  // Hidden ping-pong + shared scratch (max of both layer classes).
  void *dXa = alloc(MH * 4), *dXb = alloc(MH * 4);
  void *dH = alloc(MH * 4), *dTmp = alloc(MH * 4), *dMix = alloc(MH * 4),
       *dHh = alloc(MH * 2);
  void *dQKV = alloc((size_t)M * QW * 4); // max(C, QW) rows
  void *dMxC = alloc(MC * 4), *dMxR = alloc(MV * 4);
  void *dZ = alloc(MV * 4), *dQ48 = alloc(MV * 4), *dK48 = alloc(MV * 4),
       *dV48 = alloc(MV * 4);
  void *dAttL = alloc(MV * 4), *dAtthL = alloc(MV * 2);
  void *dB = alloc(MH48 * 4), *dA = alloc(MH48 * 4), *dBt = alloc(MH48 * 4),
       *dG48 = alloc(MH48 * 4);
  void *dK16 = alloc((size_t)M * KVW * 4), *dV16 = alloc((size_t)M * KVW * 4),
       *dKn = alloc((size_t)M * KVW * 4);
  void *dQn = alloc(MVQ * 4), *dGate = alloc(MVQ * 4),
       *dAttF = alloc(MVQ * 4), *dCore = alloc(MVQ * 4),
       *dQnh = alloc(MVQ * 2), *dAtthF = alloc(MVQ * 2);
  void *dWts = alloc((size_t)M * NQ * TC * 4),
       *dWsm = alloc((size_t)M * NQ * TC * 4),
       *dWsmh = alloc((size_t)M * NQ * TC * 2);
  void *dG17 = alloc(MI * 4), *dU17 = alloc(MI * 4), *dG17h = alloc(MI * 2);
  void *dCtrl = alloc(sizeof(DecodeControl)),
       *dCtrlSm = alloc(sizeof(DecodeControl));
  void *dInN = alloc(H * 4), *dPostN = alloc(H * 4), *dQNW = alloc(256 * 4),
       *dKNW = alloc(256 * 4), *dNG = alloc(128 * 4), *dAL = alloc(NH * 4),
       *dDT = alloc(NH * 4), *dCW = alloc((size_t)C * 4 * 4),
       *dCos = alloc(64 * TC * 4), *dSin = alloc(64 * TC * 4);
  // Per-layer state: 48x SSM (conv hist + recur), 16x KV caches.
  void *dCSL[64] = {nullptr}, *dSL[64] = {nullptr}, *dMKcL[64] = {nullptr},
       *dMVcL[64] = {nullptr};
  for (int L = 0; L < 64; ++L) {
    if (L % 4 == 3) {
      dMKcL[L] = alloc((size_t)4 * TC * 256 * 2);
      dMVcL[L] = alloc((size_t)4 * TC * 256 * 2);
    } else {
      dCSL[L] = alloc((size_t)C * 3 * 4);
      dSL[L] = alloc((size_t)NH * DL * DL * 4);
    }
  }
  // Weight buffers (INT4 packed + BF16 scales), one pair PER TENSOR KIND.
  // Two load-bearing rules: (1) recorded lists capture ADDRESSES, so kinds
  // must never share a pair within a layer (in_proj_a once lost to the MLP
  // gate this way); (2) uploads happen per layer BEFORE that layer executes
  // (upload_layer), because all 64 lists share these buffers — uploading all
  // layers up front would leave every list reading layer 63's weights.
  // in_proj_z has V6 rows (not KVW — the naming trap that cost a segfault).
  void *dWq = alloc((size_t)QW * H / 2), *dWqS = alloc((size_t)QW * 40 * 2);
  void *dWz = alloc((size_t)V6 * H / 2), *dWzS = alloc((size_t)V6 * 40 * 2);
  void *dWb = alloc((size_t)NH * H / 2), *dWbS = alloc((size_t)NH * 40 * 2);
  void *dWa = alloc((size_t)NH * H / 2), *dWaS = alloc((size_t)NH * 40 * 2);
  void *dWk = alloc((size_t)KVW * H / 2), *dWkS = alloc((size_t)KVW * 40 * 2);
  void *dWv = alloc((size_t)KVW * H / 2), *dWvS = alloc((size_t)KVW * 40 * 2);
  void *dWo = alloc((size_t)H * QN / 2), *dWoS = alloc((size_t)QN * 48 * 2);
  void *dWgu = alloc((size_t)I * H / 2), *dWguS = alloc((size_t)I * 40 * 2);
  void *dWgu2 = alloc((size_t)I * H / 2), *dWguS2 = alloc((size_t)I * 40 * 2);
  void *dWd = alloc((size_t)H * I / 2), *dWdS = alloc((size_t)H * 136 * 2);

  enum K {
    NORM, GEMM, CONV, RECUR, SILU, SPLIT, L2, BETA, RMSI, GATE, RES, CVT,
    FSPLIT, FBNORM, FCROPE, FCKV, FCQK, FCSM, FCWV, FGMUL
  };
  ze_kernel_handle_t kh[20] = {nullptr};
  std::vector<std::vector<uint8_t>> spvs;
  for (int i = 0; i < 20; ++i) {
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
    CHECK(zeKernelSetGroupSize(kh[i], 1, 1, 1));
  }
  CHECK(zeKernelSetGroupSize(kh[NORM], 256, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[GEMM], 16, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[FCQK], 16, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[FCWV], 16, 1, 1));


  ze_command_queue_handle_t qq = nullptr;
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandQueueCreate(ctx, dev, &qdesc, &qq));
  ze_fence_handle_t fence = nullptr;
  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK(zeFenceCreate(qq, &fdesc, &fence));
  ze_command_list_handle_t up = nullptr;
  ze_command_queue_desc_t idesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &idesc, &up));
  auto setarg = [&](ze_command_list_handle_t rg, ze_kernel_handle_t k,
                    uint32_t idx, size_t sz, const void *p) {
    if (zeKernelSetArgumentValue(k, idx, sz, p) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "setArg %u failed\n", idx);
      std::exit(1);
    }
    (void)rg;
  };
  auto launch = [&](ze_command_list_handle_t rg, ze_kernel_handle_t k,
                    uint32_t count) {
    ze_group_count_t gc = {count, 1, 1};
    if (zeCommandListAppendLaunchKernel(rg, k, &gc, nullptr, 0, nullptr) !=
            ZE_RESULT_SUCCESS ||
        zeCommandListAppendBarrier(rg, nullptr, 0, nullptr) !=
            ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "append failed\n");
      std::exit(1);
    }
  };

  // ---- .binfer directory ----
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
  auto upload_span = [&](const char *nm, void *dP, void *dS) {
    const Entry *e = find(nm);
    if (!e) {
      std::fprintf(stderr, "missing %s\n", nm);
      std::exit(2);
    }

    std::vector<uint8_t> pk(e->d_bytes);
    f.clear();
    f.seekg((std::streamoff)e->d_off);
    f.read((char *)pk.data(), e->d_bytes);
    std::vector<uint16_t> sc(e->sc_bytes / 2);
    f.clear();
    f.seekg((std::streamoff)e->sc_off);
    f.read((char *)sc.data(), e->sc_bytes);
    CHECK(zeCommandListAppendMemoryCopy(up, dP, pk.data(), pk.size(), nullptr,
                                        0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dS, sc.data(), sc.size() * 2,
                                        nullptr, 0, nullptr));
  };
  std::vector<float> hNorm(H);
  auto upload_norm = [&](const char *nm, void *dst, int cnt) {
    const Entry *e = find(nm);
    if (!e) {
      std::fprintf(stderr, "missing %s\n", nm);
      std::exit(2);
    }
    std::vector<uint16_t> raw(cnt);
    f.clear();
    f.seekg((std::streamoff)e->d_off);
    f.read((char *)raw.data(), cnt * 2);
    hNorm.assign(cnt, 0);
    for (int i = 0; i < cnt; ++i)
      hNorm[i] = bf16_to_f32(raw[i]);
    CHECK(zeCommandListAppendMemoryCopy(up, dst, hNorm.data(), cnt * 4,
                                        nullptr, 0, nullptr));
  };


  // Rope tables (global positions 0..TC) + controls, uploaded once.
  {
    std::vector<float> hCos(64 * TC), hSin(64 * TC);
    for (int t = 0; t < TC; ++t)
      for (int i = 0; i < 64; ++i) {
        double inv = 1.0 / std::pow(10000000.0, (double)(2 * (i % 32)) / 64.0);
        double ang = (double)t * inv;
        hCos[t * 64 + i] = (float)std::cos(ang);
        hSin[t * 64 + i] = (float)std::sin(ang);
      }
    CHECK(zeCommandListAppendMemoryCopy(up, dCos, hCos.data(),
                                        hCos.size() * 4, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dSin, hSin.data(),
                                        hSin.size() * 4, nullptr, 0, nullptr));
    DecodeControl c{9000, 0, 1, -1}, csm{0, 0, TC, 0};
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                        nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrlSm, &csm, sizeof(csm),
                                        nullptr, 0, nullptr));
  }
  // Fixed random input (also written to file for the python ref).
  std::vector<float> hXa(MH);
  for (auto &v : hXa)
    v = frnd(1.0f);
  CHECK(zeCommandListAppendMemoryCopy(up, dXa, hXa.data(), MH * 4, nullptr, 0,
                                      nullptr));
  {
    FILE *o = std::fopen("/tmp/chunk64_in.bin", "wb");
    if (o) {
      std::fwrite(hXa.data(), 4, MH, o);
      std::fclose(o);
    }
  }

  // Per-layer weight/norm upload (immediate, synchronous). Called before
  // that layer's list executes — never in bulk (shared-buffer rule above).
  auto upload_layer = [&](int L) {
    char nm[128];
    std::snprintf(nm, sizeof nm, "model.language_model.layers.%d.", L);
    std::string P(nm);
    bool full = (L % 4 == 3);
    if (!full) {
      upload_span((P + "linear_attn.in_proj_qkv.weight").c_str(), dWq, dWqS);
      upload_span((P + "linear_attn.in_proj_z.weight").c_str(), dWz, dWzS);
      upload_span((P + "linear_attn.in_proj_b.weight").c_str(), dWb, dWbS);
      upload_span((P + "linear_attn.in_proj_a.weight").c_str(), dWa, dWaS);
      upload_span((P + "linear_attn.out_proj.weight").c_str(), dWo, dWoS);
      upload_norm((P + "input_layernorm.weight").c_str(), dInN, H);
      upload_norm((P + "post_attention_layernorm.weight").c_str(), dPostN, H);
      upload_norm((P + "linear_attn.norm.weight").c_str(), dNG, 128);
      upload_norm((P + "linear_attn.A_log").c_str(), dAL, NH);
      upload_norm((P + "linear_attn.dt_bias").c_str(), dDT, NH);
      upload_norm((P + "linear_attn.conv1d.weight").c_str(), dCW, C * 4);
    } else {
      upload_span((P + "self_attn.q_proj.weight").c_str(), dWq, dWqS);
      upload_span((P + "self_attn.k_proj.weight").c_str(), dWk, dWkS);
      upload_span((P + "self_attn.v_proj.weight").c_str(), dWv, dWvS);
      upload_span((P + "self_attn.o_proj.weight").c_str(), dWo, dWoS);
      upload_norm((P + "input_layernorm.weight").c_str(), dInN, H);
      upload_norm((P + "post_attention_layernorm.weight").c_str(), dPostN, H);
      upload_norm((P + "self_attn.q_norm.weight").c_str(), dQNW, 256);
      upload_norm((P + "self_attn.k_norm.weight").c_str(), dKNW, 256);
    }
    upload_span((P + "mlp.gate_proj.weight").c_str(), dWgu, dWguS);
    upload_span((P + "mlp.up_proj.weight").c_str(), dWgu2, dWguS2);
    upload_span((P + "mlp.down_proj.weight").c_str(), dWd, dWdS);
  };


  // ---- record 64 per-layer lists (addresses only; weights per exec) ----
  std::vector<ze_command_list_handle_t> lists;
  int mmA = M, tmax = TC, n256 = 256;
  int nBcQK = (TC + 15) / 16, rowsSM = M * NQ;
  for (int L = 0; L < 64; ++L) {
    bool full = (L % 4 == 3);


    ze_command_list_handle_t rg = nullptr;
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr,
                                 0, 0};
    CHECK(zeCommandListCreate(ctx, dev, &ld, &rg));

    void *dXi = (L % 2 == 0) ? dXa : dXb;
    void *dXo = (L % 2 == 0) ? dXb : dXa;
    // Per-row norm helper (M appends).
    auto normYa = [&](void *Y, void *X, void *W) {
      int nn = H;
      for (int m = 0; m < M; ++m) {
        void *yy = (char *)Y + (size_t)m * H * 4;
        void *xx = (char *)X + (size_t)m * H * 4;
        setarg(rg, kh[NORM], 0, sizeof(void *), &yy);
        setarg(rg, kh[NORM], 1, sizeof(void *), &xx);
        setarg(rg, kh[NORM], 2, sizeof(void *), &W);
        setarg(rg, kh[NORM], 3, sizeof(int), &nn);
        setarg(rg, kh[NORM], 4, (size_t)256 * 8, nullptr);
        launch(rg, kh[NORM], 1);
      }
    };
    auto cvtYa = [&](void *Oh, void *X, int nn) {
      setarg(rg, kh[CVT], 0, sizeof(void *), &Oh);
      setarg(rg, kh[CVT], 1, sizeof(void *), &X);
      launch(rg, kh[CVT], nn);
    };
    // ChunkGemm with explicit weight buffers (shared dWq/dWkv/... pairs).
    auto cgemmW = [&](void *Wp, void *Ws, int nn, int kk, void *Ah, void *Y,
                      int gg) {
      int mm = M;
      setarg(rg, kh[GEMM], 0, sizeof(void *), &Ah);
      setarg(rg, kh[GEMM], 1, sizeof(void *), &Wp);
      setarg(rg, kh[GEMM], 2, sizeof(void *), &Ws);
      setarg(rg, kh[GEMM], 3, sizeof(void *), &Y);
      setarg(rg, kh[GEMM], 4, sizeof(int), &mm);
      setarg(rg, kh[GEMM], 5, sizeof(int), &kk);
      setarg(rg, kh[GEMM], 6, sizeof(int), &nn);
      setarg(rg, kh[GEMM], 7, (size_t)512 * 2, nullptr);
      setarg(rg, kh[GEMM], 8, (size_t)256 * 2, nullptr);
      setarg(rg, kh[GEMM], 9, (size_t)512 * 4, nullptr);
      launch(rg, kh[GEMM], ((mm + 31) / 32) * (nn / 16));
      (void)gg;
    };
    auto resYa = [&](void *Y, void *A, void *B) {
      for (int m = 0; m < M; ++m) {
        void *yy = (char *)Y + (size_t)m * H * 4;
        void *aa = (char *)A + (size_t)m * H * 4;
        void *bb = (char *)B + (size_t)m * H * 4;
        setarg(rg, kh[RES], 0, sizeof(void *), &yy);
        setarg(rg, kh[RES], 1, sizeof(void *), &aa);
        setarg(rg, kh[RES], 2, sizeof(void *), &bb);
        launch(rg, kh[RES], H);
      }
    };
    // MLP tail (shared by both layer classes; norms already uploaded).
    auto mlpTail = [&](void *dXo_, void *dTmp_) {
      normYa(dH, dTmp_, dPostN);
      cvtYa(dHh, dH, M * H);
      cgemmW(dWgu, dWguS, I, H, dHh, dG17, 40);
      cgemmW(dWgu2, dWguS2, I, H, dHh, dU17, 40);
      setarg(rg, kh[SILU], 0, sizeof(void *), &dG17);
      setarg(rg, kh[SILU], 1, sizeof(void *), &dU17);
      setarg(rg, kh[SILU], 2, sizeof(void *), &dG17);
      launch(rg, kh[SILU], MI);
      cvtYa(dG17h, dG17, MI);
      cgemmW(dWd, dWdS, H, I, dG17h, dMix, 136);
      resYa(dXo_, dTmp_, dMix);
    };
    if (!full) {
      normYa(dH, dXi, dInN);
      cvtYa(dHh, dH, M * H);
      cgemmW(dWq, dWqS, C, H, dHh, dQKV, 40);
      cgemmW(dWz, dWzS, V6, H, dHh, dZ, 40);
      cgemmW(dWb, dWbS, NH, H, dHh, dB, 40);
      cgemmW(dWa, dWaS, NH, H, dHh, dA, 40);
      setarg(rg, kh[CONV], 0, sizeof(void *), &dMxC);
      setarg(rg, kh[CONV], 1, sizeof(void *), &dQKV);
      setarg(rg, kh[CONV], 2, sizeof(void *), &dCSL[L]);
      setarg(rg, kh[CONV], 3, sizeof(void *), &dCW);
      setarg(rg, kh[CONV], 4, sizeof(int), &C);
      setarg(rg, kh[CONV], 5, sizeof(int), &mmA);
      launch(rg, kh[CONV], C);
      for (int m = 0; m < M; ++m) {
        void *mx = (char *)dMxC + (size_t)m * C * 4;
        void *q4 = (char *)dQ48 + (size_t)m * V6 * 4;
        void *k4 = (char *)dK48 + (size_t)m * V6 * 4;
        void *v4 = (char *)dV48 + (size_t)m * V6 * 4;
        setarg(rg, kh[SPLIT], 0, sizeof(void *), &mx);
        setarg(rg, kh[SPLIT], 1, sizeof(void *), &q4);
        setarg(rg, kh[SPLIT], 2, sizeof(void *), &k4);
        setarg(rg, kh[SPLIT], 3, sizeof(void *), &v4);
        launch(rg, kh[SPLIT], V6);
        setarg(rg, kh[L2], 0, sizeof(void *), &q4);
        setarg(rg, kh[L2], 1, sizeof(void *), &k4);
        launch(rg, kh[L2], 96);
        void *b1 = (char *)dB + (size_t)m * NH * 4;
        void *a1 = (char *)dA + (size_t)m * NH * 4;
        void *bt1 = (char *)dBt + (size_t)m * NH * 4;
        void *g1 = (char *)dG48 + (size_t)m * NH * 4;
        setarg(rg, kh[BETA], 0, sizeof(void *), &bt1);
        setarg(rg, kh[BETA], 1, sizeof(void *), &g1);
        setarg(rg, kh[BETA], 2, sizeof(void *), &b1);
        setarg(rg, kh[BETA], 3, sizeof(void *), &a1);
        setarg(rg, kh[BETA], 4, sizeof(void *), &dAL);
        setarg(rg, kh[BETA], 5, sizeof(void *), &dDT);
        launch(rg, kh[BETA], NH);
      }
      setarg(rg, kh[RECUR], 0, sizeof(void *), &dMxR);
      setarg(rg, kh[RECUR], 1, sizeof(void *), &dQ48);
      setarg(rg, kh[RECUR], 2, sizeof(void *), &dK48);
      setarg(rg, kh[RECUR], 3, sizeof(void *), &dV48);
      setarg(rg, kh[RECUR], 4, sizeof(void *), &dSL[L]);
      setarg(rg, kh[RECUR], 5, sizeof(void *), &dBt);
      setarg(rg, kh[RECUR], 6, sizeof(void *), &dG48);
      setarg(rg, kh[RECUR], 7, sizeof(int), &mmA);
      launch(rg, kh[RECUR], NH);
      for (int m = 0; m < M; ++m) {
        void *mx = (char *)dMxR + (size_t)m * V6 * 4;
        void *bt1 = (char *)dBt + (size_t)m * NH * 4;
        void *at = (char *)dAttL + (size_t)m * V6 * 4;
        void *zz = (char *)dZ + (size_t)m * V6 * 4;
        setarg(rg, kh[RMSI], 0, sizeof(void *), &bt1);
        setarg(rg, kh[RMSI], 1, sizeof(void *), &mx);
        launch(rg, kh[RMSI], NH);
        setarg(rg, kh[GATE], 0, sizeof(void *), &at);
        setarg(rg, kh[GATE], 1, sizeof(void *), &mx);
        setarg(rg, kh[GATE], 2, sizeof(void *), &zz);
        setarg(rg, kh[GATE], 3, sizeof(void *), &dNG);
        setarg(rg, kh[GATE], 4, sizeof(void *), &bt1);
        launch(rg, kh[GATE], V6);
      }
      cvtYa(dAtthL, dAttL, M * V6);
      cgemmW(dWo, dWoS, H, V6, dAtthL, dMix, 48);
      resYa(dTmp, dXi, dMix);
      mlpTail(dXo, dTmp);
    } else {
      normYa(dH, dXi, dInN);
      cvtYa(dHh, dH, M * H);
      cgemmW(dWq, dWqS, QW, H, dHh, dQKV, 40);
      cgemmW(dWk, dWkS, KVW, H, dHh, dK16, 40);
      cgemmW(dWv, dWvS, KVW, H, dHh, dV16, 40);
      for (int m = 0; m < M; ++m) {
        void *q16 = (char *)dQKV + (size_t)m * QW * 4;
        void *k16 = (char *)dK16 + (size_t)m * KVW * 4;
        void *qn = (char *)dQn + (size_t)m * QN * 4;
        void *gt = (char *)dGate + (size_t)m * QN * 4;
        void *kn = (char *)dKn + (size_t)m * KVW * 4;
        setarg(rg, kh[FSPLIT], 0, sizeof(void *), &q16);
        setarg(rg, kh[FSPLIT], 1, sizeof(void *), &qn);
        setarg(rg, kh[FSPLIT], 2, sizeof(void *), &gt);
        launch(rg, kh[FSPLIT], QN);
        setarg(rg, kh[FBNORM], 0, sizeof(void *), &qn);
        setarg(rg, kh[FBNORM], 1, sizeof(void *), &qn);
        setarg(rg, kh[FBNORM], 2, sizeof(void *), &dQNW);
        setarg(rg, kh[FBNORM], 3, sizeof(int), &n256);
        launch(rg, kh[FBNORM], 24);
        setarg(rg, kh[FBNORM], 0, sizeof(void *), &kn);
        setarg(rg, kh[FBNORM], 1, sizeof(void *), &k16);
        setarg(rg, kh[FBNORM], 2, sizeof(void *), &dKNW);
        setarg(rg, kh[FBNORM], 3, sizeof(int), &n256);
        launch(rg, kh[FBNORM], 4);
      }
      setarg(rg, kh[FCROPE], 0, sizeof(void *), &dQn);
      setarg(rg, kh[FCROPE], 1, sizeof(void *), &dKn);
      setarg(rg, kh[FCROPE], 2, sizeof(void *), &dCos);
      setarg(rg, kh[FCROPE], 3, sizeof(void *), &dSin);
      setarg(rg, kh[FCROPE], 4, sizeof(void *), &dCtrl);
      setarg(rg, kh[FCROPE], 5, sizeof(int), &tmax);
      setarg(rg, kh[FCROPE], 6, sizeof(int), &mmA);
      launch(rg, kh[FCROPE], M * 28);
      setarg(rg, kh[FCKV], 0, sizeof(void *), &dMKcL[L]);
      setarg(rg, kh[FCKV], 1, sizeof(void *), &dMVcL[L]);
      setarg(rg, kh[FCKV], 2, sizeof(void *), &dKn);
      setarg(rg, kh[FCKV], 3, sizeof(void *), &dV16);
      setarg(rg, kh[FCKV], 4, sizeof(void *), &dCtrl);
      setarg(rg, kh[FCKV], 5, sizeof(int), &tmax);
      setarg(rg, kh[FCKV], 6, sizeof(int), &mmA);
      launch(rg, kh[FCKV], M * 1024);
      cvtYa(dQnh, dQn, M * QN);
      {
        int pp = 0, ww = TC, ss = TC;
        setarg(rg, kh[FCQK], 0, sizeof(void *), &dQnh);
        setarg(rg, kh[FCQK], 1, sizeof(void *), &dMKcL[L]);
        setarg(rg, kh[FCQK], 2, sizeof(void *), &dWts);
        setarg(rg, kh[FCQK], 3, sizeof(int), &pp);
        setarg(rg, kh[FCQK], 4, sizeof(int), &mmA);
        setarg(rg, kh[FCQK], 5, sizeof(int), &ww);
        setarg(rg, kh[FCQK], 6, sizeof(int), &ss);
        setarg(rg, kh[FCQK], 7, (size_t)8 * 256 * 2, nullptr);
        setarg(rg, kh[FCQK], 8, (size_t)16 * 16 * 2, nullptr);
        setarg(rg, kh[FCQK], 9, (size_t)8 * 16 * 4, nullptr);
        launch(rg, kh[FCQK], (uint32_t)(M * 4 * nBcQK));
      }
      {
        int rows = M * NQ;
        setarg(rg, kh[FCSM], 0, sizeof(void *), &dWsm);
        setarg(rg, kh[FCSM], 1, sizeof(void *), &dWts);
        setarg(rg, kh[FCSM], 2, sizeof(void *), &dCtrlSm);
        setarg(rg, kh[FCSM], 3, sizeof(int), &tmax);
        setarg(rg, kh[FCSM], 4, sizeof(int), &rows);
        launch(rg, kh[FCSM], (uint32_t)rows);
      }
      cvtYa(dWsmh, dWsm, M * NQ * TC);
      {
        setarg(rg, kh[FCWV], 0, sizeof(void *), &dWsmh);
        setarg(rg, kh[FCWV], 1, sizeof(void *), &dMVcL[L]);
        setarg(rg, kh[FCWV], 2, sizeof(void *), &dCore);
        setarg(rg, kh[FCWV], 3, sizeof(int), &tmax);
        setarg(rg, kh[FCWV], 4, sizeof(int), &tmax);
        setarg(rg, kh[FCWV], 5, (size_t)8 * 16 * 2, nullptr);
        setarg(rg, kh[FCWV], 6, (size_t)16 * 16 * 2, nullptr);
        setarg(rg, kh[FCWV], 7, (size_t)8 * 16 * 4, nullptr);
        launch(rg, kh[FCWV], (uint32_t)(M * 64));
      }
      setarg(rg, kh[FGMUL], 0, sizeof(void *), &dAttF);
      setarg(rg, kh[FGMUL], 1, sizeof(void *), &dCore);
      setarg(rg, kh[FGMUL], 2, sizeof(void *), &dGate);
      launch(rg, kh[FGMUL], (uint32_t)(M * QN));
      cvtYa(dAtthF, dAttF, M * QN);
      cgemmW(dWo, dWoS, H, QN, dAtthF, dMix, 48);
      resYa(dTmp, dXi, dMix);
      mlpTail(dXo, dTmp);
    }
    CHECK(zeCommandListClose(rg));
    lists.push_back(rg);
    std::printf("layer %d recorded (%s)\n", L, full ? "full" : "linear");
  }


  // Zero all per-layer states device-side (caches + SSM), then execute.
  // The fill pattern is a POINTER (a literal 0 here once segfaulted the
  // driver with a NULL read — always pass a real pattern word).
  uint8_t zpat = 0;
  auto zero_states = [&]() {
    for (int L = 0; L < 64; ++L) {
      if (L % 4 == 3) {
        CHECK(zeCommandListAppendMemoryFill(up, dMKcL[L], &zpat, 1,
                                            (size_t)4 * TC * 256 * 2, nullptr,
                                            0, nullptr));
        CHECK(zeCommandListAppendMemoryFill(up, dMVcL[L], &zpat, 1,
                                            (size_t)4 * TC * 256 * 2, nullptr,
                                            0, nullptr));
      } else {
        CHECK(zeCommandListAppendMemoryFill(up, dCSL[L], &zpat, 1,
                                            (size_t)C * 3 * 4, nullptr, 0,
                                            nullptr));
        CHECK(zeCommandListAppendMemoryFill(up, dSL[L], &zpat, 1,
                                            (size_t)NH * DL * DL * 4, nullptr,
                                            0, nullptr));
      }
    }
  };
  zero_states();
  auto exec = [&](ze_command_list_handle_t l) -> double {
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &l, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    return now_ns() - t0;
  };
  double tAll = 0;
  for (int L = 0; L < 64; ++L) {
    upload_layer(L); // weights must precede THEIR list (shared buffers)
    tAll += exec(lists[L]);
  }
  std::vector<float> hOut(MH);
  CHECK(zeCommandListAppendMemoryCopy(up, hOut.data(), dXa, MH * 4, nullptr, 0,
                                      nullptr));
  {
    FILE *o = std::fopen("/tmp/chunk64_out.bin", "wb");
    if (o) {
      std::fwrite(hOut.data(), 4, MH, o);
      std::fclose(o);
    }
  }
  double sum = 0, mx = 0;
  for (float v : hOut) {
    sum += v;
    mx = std::max(mx, (double)std::fabs(v));
  }
  bool finite = std::isfinite(sum);
  std::printf("chunk64 64 layers %.1f ms sum %.4f maxabs %.4f %s\n", tAll / 1e6,
              sum, mx, finite ? "FINITE" : "NONFINITE");
  // Run 2: re-zero states, re-execute, bitwise determinism vs run 1.
  zero_states();
  CHECK(zeCommandListAppendMemoryCopy(up, dXa, hXa.data(), MH * 4, nullptr, 0,
                                      nullptr));
  for (int L = 0; L < 64; ++L) {
    upload_layer(L);
    exec(lists[L]);
  }
  std::vector<float> hOut2(MH);
  CHECK(zeCommandListAppendMemoryCopy(up, hOut2.data(), dXa, MH * 4, nullptr, 0,
                                      nullptr));
  bool det = true;
  for (size_t j = 0; j < hOut.size(); ++j) {
    uint32_t a, b;
    std::memcpy(&a, &hOut[j], 4);
    std::memcpy(&b, &hOut2[j], 4);
    if (a != b) {
      det = false;
      std::fprintf(stderr, "reset mismatch j %zu\n", j);
      break;
    }
  }
  std::printf("chunk64 determinism %s\n", det ? "BITWISE" : "MISMATCH");
  char js[512];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"chunk64\":\"64-layer chunked prefill "
                "(real weights, M=32, P=0)\",\"ms_all_layers\":%.1f,"
                "\"out_sum\":%.4f,\"out_maxabs\":%.4f,\"finite\":%s,"
                "\"reset_deterministic\":%s}",
                tAll / 1e6, sum, mx, finite ? "true" : "false",
                det ? "true" : "false");
  if (argc > 22) {
    FILE *o = std::fopen(argv[22], "w");
    if (o) {
      std::fprintf(o, "%s\n", js);
      std::fclose(o);
    }
  } else {
    std::printf("%s\n", js);
  }
  return (finite && det) ? 0 : 1;
}
