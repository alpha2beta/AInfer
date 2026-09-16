// T7.4 chunk production: N-chunk 64-layer prefill on REAL weights
// (default N=2, M=32: chunk A base 0 builds prefix state; chunk B base 32
// runs with carried SSM/KV state; N*64 recorded lists).
// Same compositions as chunk64real (linear + GEMM-form full blocks).
// Geometry override for production scaling (ctest contract stays M=32/N=2):
// CHUNK_M / CHUNK_N env. Buffers + launches are runtime-sized.
// Checks: last-after-prefix differs from last-alone (continuity) + full
// rerun bitwise (determinism) + last-chunk output dump for the python float
// ref + layer-3 Kn/cache snapshot dump for the decode-handoff slot check.
// Usage: chunk64mc_replay <model.binfer> <20 spv...> [report.json]
//   order: norm chunkgemm chunkssmconv chunkssmrecur silumul splitrepeat
//   l2normqk betag rmsinv normgated resaddf cvtf32f16 splitqk batchnorm
//   chunkrope chunkkvappend chunkqkgemm chunksoftmaxrow chunkwvgemm gatemul
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
      std::exit(1); /* NOT return: inside ->double/void lambdas return would  \
                       continue the run (v3 needle: 19h cascade+spin) */       \
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
    std::fprintf(stderr, "usage: chunk64mc_replay <model> <20 spv> "
                         "[report]\n");
    return 2;
  }
  const char *model = argv[1];
  const int H = 5120, C = 10240, V6 = 6144, I = 17408, NH = 48, DL = 128;
  const int QW = 12288, KVW = 1024, QN = 6144, NQ = 24;
  // Production geometry (defaults preserve the ctest contract M=32/N=2).
  int M = 32, NCH = 2;
  if (const char *e = std::getenv("CHUNK_M")) {
    int v = std::atoi(e);
    if (v >= 8 && v <= 512)
      M = v;
  }
  if (const char *e = std::getenv("CHUNK_N")) {
    int v = std::atoi(e);
    if (v >= 1 && v <= 256)
      NCH = v;
  }
  const int TC = NCH * M; // total positions across all chunks
  // Stream mode (CHUNK_STREAM=1): production single-pass — record+exec+
  // destroy per chunk, bounding live lists at 64 for unbounded NCH.
  // Validation mode (default) records all, then runs continuity/determinism.
  const bool stream = std::getenv("CHUNK_STREAM") != nullptr;
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
  // Per-layer state in decode_l0's arena layout (zero-copy handoff path):
  // KV: single dKc/dVc, full layer L -> slot L/4, base + slot*TC*4*256;
  // SSM: single dConv/dS, linear layer L -> sl = L-(L+1)/4
  // (same formulas as decode_l0.cpp:625,769,680-681,774,798).
  void *dKc = alloc((size_t)16 * TC * 4 * 256 * 2),
       *dVc = alloc((size_t)16 * TC * 4 * 256 * 2);
  void *dConv = alloc((size_t)48 * C * 3 * 4),
       *dSsm = alloc((size_t)48 * NH * DL * DL * 4);
  void *dCSL[64] = {nullptr}, *dSL[64] = {nullptr}, *dMKcL[64] = {nullptr},
       *dMVcL[64] = {nullptr};
  for (int L = 0; L < 64; ++L) {
    if (L % 4 == 3) {
      size_t slot = (size_t)(L / 4);
      dMKcL[L] = (char *)dKc + slot * TC * 4 * 256 * 2;
      dMVcL[L] = (char *)dVc + slot * TC * 4 * 256 * 2;
    } else {
      size_t sl = (size_t)(L - (L + 1) / 4);
      dCSL[L] = (char *)dConv + sl * C * 3 * 4;
      dSL[L] = (char *)dSsm + sl * NH * DL * DL * 4;
    }
  }
  // Resident weight arenas (INT4 packed + BF16 scales): one uniform-stride
  // arena PER TENSOR KIND across all 64 layers (~13 GB total). Recorded
  // lists capture ADDRESSES, so each (layer, kind) needs its own slot:
  // sharing one pair across layers forced per-layer streaming uploads
  // (~12.8 GB/chunk-pass over USB — prohibitive at production N).
  // Uploads happen ONCE (first use per layer); norms stay streamed (tiny).
  // in_proj_z has V6 rows (not KVW — the naming trap that cost a segfault).
  auto arena64 = [&](size_t perL) -> void * { return alloc(perL * 64); };
  void *dWqA = arena64((size_t)QW * H / 2),
       *dWqSA = arena64((size_t)QW * 40 * 2);
  void *dWzA = arena64((size_t)V6 * H / 2),
       *dWzSA = arena64((size_t)V6 * 40 * 2);
  void *dWbA = arena64((size_t)NH * H / 2),
       *dWbSA = arena64((size_t)NH * 40 * 2);
  void *dWaA = arena64((size_t)NH * H / 2),
       *dWaSA = arena64((size_t)NH * 40 * 2);
  void *dWkA = arena64((size_t)KVW * H / 2),
       *dWkSA = arena64((size_t)KVW * 40 * 2);
  void *dWvA = arena64((size_t)KVW * H / 2),
       *dWvSA = arena64((size_t)KVW * 40 * 2);
  void *dWoA = arena64((size_t)H * QN / 2),
       *dWoSA = arena64((size_t)QN * 48 * 2);
  void *dWguA = arena64((size_t)I * H / 2),
       *dWguSA = arena64((size_t)I * 40 * 2);
  void *dWgu2A = arena64((size_t)I * H / 2),
       *dWgu2SA = arena64((size_t)I * 40 * 2);
  void *dWdA = arena64((size_t)H * I / 2),
       *dWdSA = arena64((size_t)H * 136 * 2);
  void *dWqL[64], *dWqSL[64], *dWzL[64], *dWzSL[64], *dWbL[64], *dWbSL[64],
       *dWaL[64], *dWaSL[64], *dWkL[64], *dWkSL[64], *dWvL[64], *dWvSL[64],
       *dWoL[64], *dWoSL[64], *dWguL[64], *dWguSL[64], *dWgu2L[64],
       *dWgu2SL[64], *dWdL[64], *dWdSL[64];
  for (int L = 0; L < 64; ++L) {
    dWqL[L] = (char *)dWqA + (size_t)L * QW * H / 2;
    dWqSL[L] = (char *)dWqSA + (size_t)L * QW * 40 * 2;
    dWzL[L] = (char *)dWzA + (size_t)L * V6 * H / 2;
    dWzSL[L] = (char *)dWzSA + (size_t)L * V6 * 40 * 2;
    dWbL[L] = (char *)dWbA + (size_t)L * NH * H / 2;
    dWbSL[L] = (char *)dWbSA + (size_t)L * NH * 40 * 2;
    dWaL[L] = (char *)dWaA + (size_t)L * NH * H / 2;
    dWaSL[L] = (char *)dWaSA + (size_t)L * NH * 40 * 2;
    dWkL[L] = (char *)dWkA + (size_t)L * KVW * H / 2;
    dWkSL[L] = (char *)dWkSA + (size_t)L * KVW * 40 * 2;
    dWvL[L] = (char *)dWvA + (size_t)L * KVW * H / 2;
    dWvSL[L] = (char *)dWvSA + (size_t)L * KVW * 40 * 2;
    dWoL[L] = (char *)dWoA + (size_t)L * H * QN / 2;
    dWoSL[L] = (char *)dWoSA + (size_t)L * QN * 48 * 2;
    dWguL[L] = (char *)dWguA + (size_t)L * I * H / 2;
    dWguSL[L] = (char *)dWguSA + (size_t)L * I * 40 * 2;
    dWgu2L[L] = (char *)dWgu2A + (size_t)L * I * H / 2;
    dWgu2SL[L] = (char *)dWgu2SA + (size_t)L * I * 40 * 2;
    dWdL[L] = (char *)dWdA + (size_t)L * H * I / 2;
    dWdSL[L] = (char *)dWdSA + (size_t)L * H * 136 * 2;
  }

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


  // Rope tables (global positions 0..TC-1), uploaded once.
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
  }
  // Per-chunk control upload (control-only mutation at exec, T5.4 pattern):
  // rope/append read position=base, softmax reads active_length=W.
  auto upload_chunk_ctrl = [&](int ch) {
    int base = ch * M, W = (ch + 1) * M;
    DecodeControl c{9000 + ch, base, base + 1, -1}, csm{0, 0, W, 0};
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                        nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrlSm, &csm, sizeof(csm),
                                        nullptr, 0, nullptr));
  };
  // Chunk inputs: random by default; real hidden states when
  // CHUNK64MC_HOST_IN=<prefix> (reads prefix_<ch>.bin, with legacy
  // prefix_a/b.bin fallback for ch 0/1, e.g. host-embedded token ids
  // for quality arbitration).
  std::vector<std::vector<float>> hX(NCH, std::vector<float>(MH));
  if (const char *ip = std::getenv("CHUNK64MC_HOST_IN")) {
    for (int ch = 0; ch < NCH; ++ch) {
      char pp[256];
      std::snprintf(pp, sizeof pp, "%s_%d.bin", ip, ch);
      FILE *fi = std::fopen(pp, "rb");
      if (!fi && ch < 2) {
        std::snprintf(pp, sizeof pp, "%s_%c.bin", ip, ch == 0 ? 'a' : 'b');
        fi = std::fopen(pp, "rb");
      }
      if (!fi) {
        std::fprintf(stderr, "no host input %s\n", pp);
        return 2;
      }
      size_t nr = std::fread(hX[ch].data(), 4, MH, fi);
      std::fclose(fi);
      if (nr != (size_t)MH) {
        std::fprintf(stderr, "short host input %s\n", pp);
        return 2;
      }
    }
    std::fprintf(stderr, "[mc] host inputs from %s_<ch>.bin x%d\n", ip, NCH);
  } else {
    for (int ch = 0; ch < NCH; ++ch)
      for (auto &v : hX[ch])
        v = frnd(1.0f + 0.5f * ch);
  }
  // Dump prefix override (CHUNK_DUMP_PREFIX) so geometry-specific runs
  // never clobber each other's inputs (ctest M=32 vs scaling runs).
  const char *dpre = std::getenv("CHUNK_DUMP_PREFIX");
  if (!dpre)
    dpre = "/tmp/chunk64mc_in";
  for (int ch = 0; ch < NCH; ++ch) {
    char pp[256];
    std::snprintf(pp, sizeof pp, "%s_%d.bin", dpre, ch);
    FILE *o = std::fopen(pp, "wb");
    if (o) {
      std::fwrite(hX[ch].data(), 4, MH, o);
      std::fclose(o);
    }
  }
  // Legacy aliases (ref scripts + arbitration read _a/_b).
  if (NCH == 2) {
    std::FILE *s = std::fopen("/tmp/chunk64mc_in_a.bin", "wb");
    if (s) {
      std::fwrite(hX[0].data(), 4, MH, s);
      std::fclose(s);
    }
    s = std::fopen("/tmp/chunk64mc_in_b.bin", "wb");
    if (s) {
      std::fwrite(hX[1].data(), 4, MH, s);
      std::fclose(s);
    }
  }

  // Per-layer weight/norm upload (immediate, synchronous) into resident
  // per-layer slots. INT4 spans upload ONCE (done-guarded); norms re-upload
  // (tiny) to keep the pre-exec call pattern uniform across runs.
  bool wDone[64] = {false};
  auto upload_layer = [&](int L) {
    char nm[128];
    std::snprintf(nm, sizeof nm, "model.language_model.layers.%d.", L);
    std::string P(nm);
    bool full = (L % 4 == 3);
    if (!wDone[L]) {
      if (!full) {
        upload_span((P + "linear_attn.in_proj_qkv.weight").c_str(), dWqL[L],
                    dWqSL[L]);
        upload_span((P + "linear_attn.in_proj_z.weight").c_str(), dWzL[L],
                    dWzSL[L]);
        upload_span((P + "linear_attn.in_proj_b.weight").c_str(), dWbL[L],
                    dWbSL[L]);
        upload_span((P + "linear_attn.in_proj_a.weight").c_str(), dWaL[L],
                    dWaSL[L]);
        upload_span((P + "linear_attn.out_proj.weight").c_str(), dWoL[L],
                    dWoSL[L]);
      } else {
        upload_span((P + "self_attn.q_proj.weight").c_str(), dWqL[L],
                    dWqSL[L]);
        upload_span((P + "self_attn.k_proj.weight").c_str(), dWkL[L],
                    dWkSL[L]);
        upload_span((P + "self_attn.v_proj.weight").c_str(), dWvL[L],
                    dWvSL[L]);
        upload_span((P + "self_attn.o_proj.weight").c_str(), dWoL[L],
                    dWoSL[L]);
      }
      upload_span((P + "mlp.gate_proj.weight").c_str(), dWguL[L], dWguSL[L]);
      upload_span((P + "mlp.up_proj.weight").c_str(), dWgu2L[L], dWgu2SL[L]);
      upload_span((P + "mlp.down_proj.weight").c_str(), dWdL[L], dWdSL[L]);
      wDone[L] = true;
    }
    if (!full) {
      upload_norm((P + "input_layernorm.weight").c_str(), dInN, H);
      upload_norm((P + "post_attention_layernorm.weight").c_str(), dPostN, H);
      upload_norm((P + "linear_attn.norm.weight").c_str(), dNG, 128);
      upload_norm((P + "linear_attn.A_log").c_str(), dAL, NH);
      upload_norm((P + "linear_attn.dt_bias").c_str(), dDT, NH);
      upload_norm((P + "linear_attn.conv1d.weight").c_str(), dCW, C * 4);
    } else {
      upload_norm((P + "input_layernorm.weight").c_str(), dInN, H);
      upload_norm((P + "post_attention_layernorm.weight").c_str(), dPostN, H);
      upload_norm((P + "self_attn.q_norm.weight").c_str(), dQNW, 256);
      upload_norm((P + "self_attn.k_norm.weight").c_str(), dKNW, 256);
    }
  };


  // Exec helpers (defined before record so stream mode can exec inline).
  std::vector<std::vector<ze_command_list_handle_t>> lists(NCH);
  std::vector<double> tms(NCH, 0);
  // Zero all per-layer states device-side (caches + SSM), then execute.
  // The fill pattern is a POINTER (a literal 0 here once segfaulted the
  // driver with a NULL read — always pass a real pattern word).
  uint8_t zpat = 0;
  auto zero_states = [&]() {
    CHECK(zeCommandListAppendMemoryFill(up, dKc, &zpat, 1,
                                        (size_t)16 * TC * 4 * 256 * 2, nullptr,
                                        0, nullptr));
    CHECK(zeCommandListAppendMemoryFill(up, dVc, &zpat, 1,
                                        (size_t)16 * TC * 4 * 256 * 2, nullptr,
                                        0, nullptr));
    CHECK(zeCommandListAppendMemoryFill(up, dConv, &zpat, 1,
                                        (size_t)48 * C * 3 * 4, nullptr, 0,
                                        nullptr));
    CHECK(zeCommandListAppendMemoryFill(up, dSsm, &zpat, 1,
                                        (size_t)48 * NH * DL * DL * 4, nullptr,
                                        0, nullptr));
  };
  auto exec = [&](ze_command_list_handle_t l) -> double {
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &l, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    return now_ns() - t0;
  };
  // One chunk-run: upload input + chunk controls, then upload+exec per layer.
  // After layer 3 of the LAST chunk, snapshot rope'd K (dKn) + the layer-3
  // cache for the handoff slot-placement check (immediate list is sync).
  const int LC = NCH - 1; // probed (last) chunk for continuity/snapshots
  std::vector<float> hKn3((size_t)M * KVW);
  std::vector<uint16_t> hKc3((size_t)4 * TC * 256);
  auto run_chunk = [&](int ch, const std::vector<float> &hx) -> double {
    CHECK(zeCommandListAppendMemoryCopy(up, dXa, hx.data(), MH * 4, nullptr, 0,
                                        nullptr));
    upload_chunk_ctrl(ch);
    double t = 0;
    for (int L = 0; L < 64; ++L) {
      upload_layer(L); // weights must precede THEIR list (shared buffers)
      t += exec(lists[ch][L]);
      if (ch == LC && L == 3) {
        CHECK(zeCommandListAppendMemoryCopy(up, hKn3.data(), dKn,
                                            hKn3.size() * 4, nullptr, 0,
                                            nullptr));
        CHECK(zeCommandListAppendMemoryCopy(up, hKc3.data(), dMKcL[3],
                                            hKc3.size() * 2, nullptr, 0,
                                            nullptr));
      }
    }
    return t;
  };
  auto read_out = [&](std::vector<float> &h) {
    h.assign(MH, 0);
    CHECK(zeCommandListAppendMemoryCopy(up, h.data(), dXa, MH * 4, nullptr, 0,
                                        nullptr));
  };
  auto differs = [&](const std::vector<float> &a,
                     const std::vector<float> &b) -> bool {
    for (size_t j = 0; j < a.size(); ++j) {
      uint32_t x, y;
      std::memcpy(&x, &a[j], 4);
      std::memcpy(&y, &b[j], 4);
      if (x != y)
        return true;
    }
    return false;
  };

  // ---- record NCHx64 per-layer lists (addresses only; weights per exec)
  // Chunk ch: base = ch*M, valid width W = (ch+1)*M. S rows stride TC.
  int mmA = M, tmax = TC, n256 = 256;
  int rowsSM = M * NQ;
  for (int ch = 0; ch < NCH; ++ch) {
    int base = ch * M, W = (ch + 1) * M, nBcQK = (W + 15) / 16;
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
    // ChunkGemm with explicit per-layer resident weight buffers.
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
      cgemmW(dWguL[L], dWguSL[L], I, H, dHh, dG17, 40);
      cgemmW(dWgu2L[L], dWgu2SL[L], I, H, dHh, dU17, 40);
      setarg(rg, kh[SILU], 0, sizeof(void *), &dG17);
      setarg(rg, kh[SILU], 1, sizeof(void *), &dU17);
      setarg(rg, kh[SILU], 2, sizeof(void *), &dG17);
      launch(rg, kh[SILU], MI);
      cvtYa(dG17h, dG17, MI);
      cgemmW(dWdL[L], dWdSL[L], H, I, dG17h, dMix, 136);
      resYa(dXo_, dTmp_, dMix);
    };
    if (!full) {
      normYa(dH, dXi, dInN);
      cvtYa(dHh, dH, M * H);
      cgemmW(dWqL[L], dWqSL[L], C, H, dHh, dQKV, 40);
      cgemmW(dWzL[L], dWzSL[L], V6, H, dHh, dZ, 40);
      cgemmW(dWbL[L], dWbSL[L], NH, H, dHh, dB, 40);
      cgemmW(dWaL[L], dWaSL[L], NH, H, dHh, dA, 40);
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
      cgemmW(dWoL[L], dWoSL[L], H, V6, dAtthL, dMix, 48);
      resYa(dTmp, dXi, dMix);
      mlpTail(dXo, dTmp);
    } else {
      normYa(dH, dXi, dInN);
      cvtYa(dHh, dH, M * H);
      cgemmW(dWqL[L], dWqSL[L], QW, H, dHh, dQKV, 40);
      cgemmW(dWkL[L], dWkSL[L], KVW, H, dHh, dK16, 40);
      cgemmW(dWvL[L], dWvSL[L], KVW, H, dHh, dV16, 40);
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
        int pp = base, ww = W, ss = TC;
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
        int kk = W; // K = valid cols; KMAX = row stride TC (tmax)
        setarg(rg, kh[FCWV], 0, sizeof(void *), &dWsmh);
        setarg(rg, kh[FCWV], 1, sizeof(void *), &dMVcL[L]);
        setarg(rg, kh[FCWV], 2, sizeof(void *), &dCore);
        setarg(rg, kh[FCWV], 3, sizeof(int), &kk);
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
      cgemmW(dWoL[L], dWoSL[L], H, QN, dAtthF, dMix, 48);
      resYa(dTmp, dXi, dMix);
      mlpTail(dXo, dTmp);
    }
    CHECK(zeCommandListClose(rg));
    lists[ch].push_back(rg);
    if ((L & 15) == 0 || L == 63)
      std::printf("chunk %d layer %d recorded (%s)\n", ch, L,
                  full ? "full" : "linear");
  }
  if (stream) {
    // Production single-pass: exec this chunk now, then destroy its lists
    // (bounds live lists at 64 for unbounded NCH).
    if (ch == 0)
      zero_states();
    tms[ch] = run_chunk(ch, hX[ch]);
    for (auto l : lists[ch])
      zeCommandListDestroy(l);
    lists[ch].clear();
    std::printf("chunk %d executed+destroyed\n", ch);
  }
  }


  bool cont = false, det = false;
  if (!stream) {
  // Run 1: all chunks in order with states carrying (production path).
  zero_states();
  for (int ch = 0; ch < NCH; ++ch)
    tms[ch] = run_chunk(ch, hX[ch]);
  }
  std::vector<float> hOutB;
  read_out(hOutB);
  {
    FILE *o = std::fopen("/tmp/chunk64mc_out_b.bin", "wb");
    if (o) {
      std::fwrite(hOutB.data(), 4, MH, o);
      std::fclose(o);
    }
    o = std::fopen("/tmp/chunk64mc_kn3.bin", "wb");
    if (o) {
      std::fwrite(hKn3.data(), 4, hKn3.size(), o);
      std::fclose(o);
    }
    o = std::fopen("/tmp/chunk64mc_kc3.bin", "wb");
    if (o) {
      std::fwrite(hKc3.data(), 2, hKc3.size(), o);
      std::fclose(o);
    }
  }
  // Full state dump for file-handoff to decode (CHUNK_DUMP_CACHES=<dir>):
  // decode-layout KV/SSM arenas + last hidden row + meta. Sequential
  // processes (no VRAM sharing); decode imports with strided H2D per layer.
  if (const char *dd = std::getenv("CHUNK_DUMP_CACHES")) {
    char pp[512];
    auto dumpDev = [&](const char *nm, void *dev, size_t n) {
      std::snprintf(pp, sizeof pp, "%s/%s", dd, nm);
      std::vector<uint8_t> hb(n);
      if (zeCommandListAppendMemoryCopy(up, hb.data(), dev, n, nullptr, 0,
                                        nullptr) != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr, "dump copy %s failed\n", pp);
        std::exit(1);
      }
      FILE *o = std::fopen(pp, "wb");
      if (!o) {
        std::fprintf(stderr, "dump open %s failed\n", pp);
        std::exit(1);
      }
      size_t nw = std::fwrite(hb.data(), 1, n, o);
      std::fclose(o);
      if (nw != n) {
        std::fprintf(stderr, "dump short %s\n", pp);
        std::exit(1);
      }
      std::printf("dumped %s %zu bytes\n", pp, n);
    };
    dumpDev("kc.bin", dKc, (size_t)16 * TC * 4 * 256 * 2);
    dumpDev("vc.bin", dVc, (size_t)16 * TC * 4 * 256 * 2);
    dumpDev("conv.bin", dConv, (size_t)48 * C * 3 * 4);
    dumpDev("ssm.bin", dSsm, (size_t)48 * NH * DL * DL * 4);
    dumpDev("hidlast.bin", (char *)dXa + (size_t)(M - 1) * H * 4,
            (size_t)H * 4);
    std::snprintf(pp, sizeof pp, "%s/meta.txt", dd);
    FILE *o = std::fopen(pp, "w");
    if (o) {
      std::fprintf(o, "P=%d M=%d NCH=%d TC=%d\n", TC, M, NCH, TC);
      std::fclose(o);
    }
  }
  double sum = 0, mx = 0;
  for (float v : hOutB) {
    sum += v;
    mx = std::max(mx, (double)std::fabs(v));
  }
  bool finite = std::isfinite(sum);
  std::printf("chunk64mc M=%d NCH=%d ms/chunk", M, NCH);
  for (int ch = 0; ch < NCH; ++ch)
    std::printf(" %.1f", tms[ch] / 1e6);
  std::printf(" sum %.4f maxabs %.4f %s\n", sum, mx,
              finite ? "FINITE" : "NONFINITE");
  if (!stream) {
  // Run 2: last chunk alone from zeroed states — must DIFFER (state live).
  zero_states();
  run_chunk(LC, hX[LC]);
  std::vector<float> hOutBalone;
  read_out(hOutBalone);
  cont = differs(hOutB, hOutBalone);
  std::printf("chunk64mc continuity %s\n",
              cont ? "last-after-prefix differs from last-alone" : "NO-DIFF!");
  // Run 3: all chunks again — bitwise vs run 1 (determinism).
  zero_states();
  for (int ch = 0; ch < NCH; ++ch)
    run_chunk(ch, hX[ch]);
  std::vector<float> hOutB2;
  read_out(hOutB2);
  det = !differs(hOutB, hOutB2);
  std::printf("chunk64mc determinism %s\n", det ? "BITWISE" : "MISMATCH");
  }
  double tTot = 0;
  for (double t : tms)
    tTot += t;
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"chunk64mc\":\"64-layer N-chunk "
                "prefill (real weights, M=%d, NCH=%d)\","
                "\"mode\":\"%s\",\"ms_all_chunks\":%.1f,\"out_sum\":%.4f,"
                "\"out_maxabs\":%.4f,\"finite\":%s,\"continuity\":%s,"
                "\"reset_deterministic\":%s}",
                M, NCH, stream ? "stream" : "validate", tTot / 1e6, sum, mx,
                finite ? "true" : "false",
                cont ? "true" : "false", det ? "true" : "false");
  if (argc > 22) {
    FILE *o = std::fopen(argv[22], "w");
    if (o) {
      std::fprintf(o, "%s\n", js);
      std::fclose(o);
    }
  } else {
    std::printf("%s\n", js);
  }
  return (finite && (stream || (cont && det))) ? 0 : 1;
}
