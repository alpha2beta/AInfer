// AInfer T5.3 loop adoption: full 64-layer decode loop as RECORDED raw-L0
// command lists. One list per layer (2 templates x per-layer weight, cache,
// small-weight addresses baked at init) + embed list + logits/argmax tail,
// replayed per token with only the DecodeControl contents changing (token,
// position, active_length via 16 B immediate updates). Steady state builds
// zero command lists and performs zero allocations.
// The certified SYCL loop (tools/decode/decode) remains the parity oracle.
// Usage: decode_l0 <model.binfer> <P> <G> <spvdir> [report.json]
//        [--ids=..] [--max-new=..]
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "../sample/sampler.h" // T7.1 host sampler (temp/top-k/top-p/penalty)

#define CHECK(expr)                                                            \
  do {                                                                         \
    ze_result_t _r = (expr);                                                   \
    if (_r != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "L0 error %d at %s:%d (%s)\n", (int)_r, __FILE__,   \
                   __LINE__, #expr);                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

struct Entry {
  char name[64];
  uint64_t d_off, d_bytes, sc_off, sc_bytes;
  uint32_t crc;
};
struct DecodeControl {
  int token_id, position, active_length, selected_token;
};
static uint64_t rd64(std::ifstream &f) {
  uint64_t v;
  f.read((char *)&v, 8);
  return v;
}
static uint32_t rd32(std::ifstream &f) {
  uint32_t v;
  f.read((char *)&v, 4);
  return v;
}
static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}
static double now_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

int main(int argc, char **argv) {
  std::vector<int> opt_ids;
  int opt_max_new = -1;
  SampleParams samp; // T7.1 defaults = certified greedy (temp 0)
  const char *spvdir = nullptr;
  std::vector<const char *> pos;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--ids=", 6) == 0) {
      const char *q = argv[i] + 6;
      while (*q) {
        opt_ids.push_back(atoi(q));
        while (*q && *q != ',')
          ++q;
        if (*q == ',')
          ++q;
      }
    } else if (std::strncmp(argv[i], "--max-new=", 10) == 0) {
      opt_max_new = atoi(argv[i] + 10);
    } else if (std::strncmp(argv[i], "--temp=", 7) == 0) {
      samp.temp = (float)atof(argv[i] + 7);
    } else if (std::strncmp(argv[i], "--top-k=", 8) == 0) {
      samp.top_k = atoi(argv[i] + 8);
    } else if (std::strncmp(argv[i], "--top-p=", 8) == 0) {
      samp.top_p = (float)atof(argv[i] + 8);
    } else if (std::strncmp(argv[i], "--seed=", 7) == 0) {
      samp.seed = (uint64_t)atoll(argv[i] + 7);
    } else if (std::strncmp(argv[i], "--rep-penalty=", 14) == 0) {
      samp.rep_penalty = (float)atof(argv[i] + 14);
    } else if (std::strncmp(argv[i], "--spvdir=", 9) == 0) {
      spvdir = argv[i] + 9;
    } else {
      pos.push_back(argv[i]);
    }
  }
  if (pos.size() < 4)
    spvdir = nullptr;
  else
    spvdir = pos[3];
  if (pos.size() < 4 || !spvdir) {
    std::fprintf(stderr, "usage: decode_l0 <model> <P> <G> <spvdir> [report] "
                         "[--ids=..] [--max-new=..]\n");
    return 2;
  }
  const char *path = pos[0];
  int P = atoi(pos[1]), G = atoi(pos[2]);
  if (!opt_ids.empty())
    P = (int)opt_ids.size();
  if (opt_max_new >= 0)
    G = opt_max_new;
  // T7.4 64K sizing validation: AINFER_MAXCTX may GROW the context sizing
  // (buffers, tables, strides) while running a short prompt. Grow-only so the
  // T4.5 guard (MAXCTX finalized from effective P/G before use) stays intact.
  int MAXCTX = P + G + 8;
  if (const char *mx = std::getenv("AINFER_MAXCTX")) {
    int m = std::atoi(mx);
    if (m > MAXCTX)
      MAXCTX = m;
  }
  const int H = 5120, C = 10240, V6 = 6144, I = 17408, NH = 48, D = 128,
            QW = 12288, KVW = 1024, QN = 6144, V = 248320;

  std::ifstream f(path, std::ios::binary);
  if (!f)
    return 2;
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "BINFER\x00\x01", 8) != 0) {
    std::fprintf(stderr, "bad magic (not a .binfer file)\n");
    return 2;
  }
  uint32_t ver = rd32(f), flags = rd32(f);
  uint64_t n;
  f.read((char *)&n, 8);
  uint64_t table_off = rd64(f);
  (void)ver;
  (void)flags;
  f.seekg((std::streamoff)table_off);
  uint64_t dir_off = 0, dir_bytes = 0;
  for (int i = 0; i < 5; ++i) {
    uint32_t sid = rd32(f);
    uint64_t off = rd64(f), nb = rd64(f);
    rd32(f);
    f.seekg(8, std::ios::cur);
    if (sid == 5) {
      dir_off = off;
      dir_bytes = nb;
    }
  }
  (void)dir_bytes;
  std::vector<Entry> ents(n);
  f.seekg((std::streamoff)dir_off);
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
  f.seekg(0, std::ios::end);
  {
    uint64_t fsize = (uint64_t)f.tellg();
    f.clear();
    for (auto &e : ents) {
      if (e.d_bytes == 0 || e.d_off + e.d_bytes > fsize - 32) {
        std::fprintf(stderr, "payload span out of range: %s\n", e.name);
        return 2;
      }
      if (e.sc_bytes && e.sc_off + e.sc_bytes > fsize - 32) {
        std::fprintf(stderr, "scale span out of range: %s\n", e.name);
        return 2;
      }
    }
  }

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
    return 2;
  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(drv, &cdesc, &ctx));
  ze_device_mem_alloc_desc_t mdesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  auto alloc = [&](size_t sz) -> void * {
    void *p = nullptr;
    if (zeMemAllocDevice(ctx, &mdesc, sz, 4096, dev, &p) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "alloc %zu failed\n", sz);
      std::exit(1);
    }
    return p;
  };
  uint64_t pay_lo = UINT64_MAX, pay_hi = 0, sc_lo = UINT64_MAX, sc_hi = 0;
  for (auto &e : ents) {
    pay_lo = e.d_off < pay_lo ? e.d_off : pay_lo;
    pay_hi = e.d_off + e.d_bytes > pay_hi ? e.d_off + e.d_bytes : pay_hi;
    if (e.sc_bytes) {
      sc_lo = e.sc_off < sc_lo ? e.sc_off : sc_lo;
      sc_hi = e.sc_off + e.sc_bytes > sc_hi ? e.sc_off + e.sc_bytes : sc_hi;
    }
  }
  void *payArena = alloc(pay_hi - pay_lo);
  void *scArena = alloc(sc_hi - sc_lo);
  ze_command_list_handle_t up = nullptr;
  ze_command_queue_desc_t idesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &idesc, &up));
  {
    const size_t CH = 1u << 28;
    std::vector<char> staging(CH);
    auto upspan = [&](uint64_t foff, void *arena, uint64_t alo, uint64_t len) {
      uint64_t done = 0;
      while (done < len) {
        size_t c = (size_t)((len - done > CH) ? CH : (len - done));
        f.clear();
        f.seekg((std::streamoff)(foff + done));
        f.read(staging.data(), c);
        ze_result_t r = zeCommandListAppendMemoryCopy(
            up, (char *)arena + (foff + done - alo), staging.data(), c,
            nullptr, 0, nullptr);
        if (r != ZE_RESULT_SUCCESS) {
          std::fprintf(stderr, "upload failed\n");
          std::exit(1);
        }
        done += c;
      }
    };
    upspan(pay_lo, payArena, pay_lo, pay_hi - pay_lo);
    upspan(sc_lo, scArena, sc_lo, sc_hi - sc_lo);
  }

  // ---- L0-context scratch / caches / small mirrors / rope / control ----
  void *dX = alloc(H * 4), *dH = alloc(H * 4), *dTmp = alloc(H * 4),
       *dMix = alloc(H * 4);
  void *dQKV = alloc(C * 4), *dMx = alloc(C * 4);
  void *dZ = alloc(V6 * 4), *dQ48 = alloc(V6 * 4), *dK48 = alloc(V6 * 4),
       *dV48 = alloc(V6 * 4), *dAtt = alloc(V6 * 4);
  void *dQ16 = alloc(QW * 4), *dK16 = alloc(KVW * 4), *dV16 = alloc(KVW * 4),
       *dKn = alloc(KVW * 4);
  void *dQn = alloc(QN * 4), *dGate = alloc(QN * 4);
  void *dWts = alloc((size_t)24 * MAXCTX * 4);
  void *dG17 = alloc(I * 4), *dU17 = alloc(I * 4), *dQ8 = alloc(I),
       *dSq = alloc(136 * 4);
  void *dB = alloc(NH * 4), *dA = alloc(NH * 4), *dBt = alloc(NH * 4),
       *dG48 = alloc(NH * 4);
  void *dLogits = alloc((size_t)V * 4);
  void *dPV = alloc(64 * 4), *dPI = alloc(64 * 4), *dOutT = alloc(4);
  void *dCtrl = alloc(sizeof(DecodeControl));
  void *dKc = alloc((size_t)16 * MAXCTX * 4 * 256 * 2),
       *dVc = alloc((size_t)16 * MAXCTX * 4 * 256 * 2);
  void *dConv = alloc((size_t)48 * C * 3 * 4),
       *dS = alloc((size_t)48 * NH * D * D * 4);
  void *dInN = alloc((size_t)64 * H * 4), *dPostN = alloc((size_t)64 * H * 4),
       *dQNW = alloc((size_t)16 * 256 * 4), *dKNW = alloc((size_t)16 * 256 * 4),
       *dLinN = alloc((size_t)48 * 128 * 4), *dAL = alloc((size_t)48 * 48 * 4),
       *dDT = alloc((size_t)48 * 48 * 4),
       *dConvW = alloc((size_t)48 * C * 4 * 4), *dFinN = alloc(H * 4);
  void *dCos = alloc(64 * MAXCTX * 4), *dSin = alloc(64 * MAXCTX * 4);
  const char *LP = "model.language_model.layers.";
  auto is_full = [](int L) { return (L % 4) == 3; };
  auto nm = [&](int L, const char *s) {
    return std::string(LP) + std::to_string(L) + "." + s;
  };
  auto load_small = [&](const std::string &nn, void *dst, size_t cnt) {
    const Entry *e = find(nn);
    if (!e) {
      std::fprintf(stderr, "missing %s\n", nn.c_str());
      std::exit(1);
    }
    std::vector<uint16_t> raw(cnt);
    f.clear();
    f.seekg((std::streamoff)e->d_off);
    f.read((char *)raw.data(), cnt * 2);
    std::vector<float> out(cnt);
    for (size_t i = 0; i < cnt; ++i)
      out[i] = bf16_to_f32(raw[i]);
    CHECK(zeCommandListAppendMemoryCopy(up, dst, out.data(), cnt * 4, nullptr,
                                        0, nullptr));
  };
  for (int L = 0; L < 64; ++L) {
    load_small(nm(L, "input_layernorm.weight"), (char *)dInN + (size_t)L * H * 4,
               H);
    load_small(nm(L, "post_attention_layernorm.weight"),
               (char *)dPostN + (size_t)L * H * 4, H);
    if (is_full(L)) {
      int slot = L / 4;
      load_small(nm(L, "self_attn.q_norm.weight"),
                 (char *)dQNW + (size_t)slot * 256 * 4, 256);
      load_small(nm(L, "self_attn.k_norm.weight"),
                 (char *)dKNW + (size_t)slot * 256 * 4, 256);
    } else {
      int sl = L - (L + 1) / 4;
      load_small(nm(L, "linear_attn.norm.weight"),
                 (char *)dLinN + (size_t)sl * 128 * 4, 128);
      load_small(nm(L, "linear_attn.A_log"), (char *)dAL + (size_t)sl * 48 * 4,
                 48);
      load_small(nm(L, "linear_attn.dt_bias"), (char *)dDT + (size_t)sl * 48 * 4,
                 48);
      load_small(nm(L, "linear_attn.conv1d.weight"),
                 (char *)dConvW + (size_t)sl * C * 4 * 4, (size_t)C * 4);
    }
  }
  load_small("model.language_model.norm.weight", dFinN, H);
  {
    std::vector<float> cs(64 * MAXCTX), sn(64 * MAXCTX);
    for (int t = 0; t < MAXCTX; ++t)
      for (int i = 0; i < 64; ++i) {
        double inv = 1.0 / std::pow(10000000.0, (double)(2 * (i % 32)) / 64.0);
        double ang = (double)t * inv;
        cs[t * 64 + i] = (float)std::cos(ang);
        sn[t * 64 + i] = (float)std::sin(ang);
      }
    CHECK(zeCommandListAppendMemoryCopy(up, dCos, cs.data(), cs.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dSin, sn.data(), sn.size() * 4,
                                        nullptr, 0, nullptr));
  }
  { // zeroed persistent caches (BF16 zeros are zero bytes)
    // T7.4: device-side fill — at 64K the KV zero set is 4 GiB and must never
    // cross PCIe as host uploads. Synchronous immediate list: inline completion.
    const uint8_t z = 0;
    CHECK(zeCommandListAppendMemoryFill(
        up, dKc, &z, 1, (size_t)16 * MAXCTX * 4 * 256 * 2, nullptr, 0,
        nullptr));
    CHECK(zeCommandListAppendMemoryFill(
        up, dVc, &z, 1, (size_t)16 * MAXCTX * 4 * 256 * 2, nullptr, 0,
        nullptr));
    CHECK(zeCommandListAppendMemoryFill(up, dConv, &z, 1,
                                        (size_t)48 * C * 3 * 4, nullptr, 0,
                                        nullptr));
    CHECK(zeCommandListAppendMemoryFill(up, dS, &z, 1,
                                        (size_t)48 * NH * D * D * 4, nullptr,
                                        0, nullptr));
  }
  const Entry *Eemb = find("model.language_model.embed_tokens.weight");
  const void *embP = (const char *)payArena + (Eemb->d_off - pay_lo);

  // ---- SPIR-V modules (one per entry, fixed names under spvdir) ----
  struct Mod {
    const char *file, *entry;
  };
  Mod mods[] = {
      {"norm.spv", "_ZTS8RMSNormW"},     {"gemv.spv", "_ZTS8Int4Gemv"},
      {"splitqk.spv", "_ZTS7SplitQK"},   {"batchnorm.spv", "_ZTS9BatchNorm"},
      {"rope.spv", "_ZTS9RopeApply"},    {"kvappend.spv", "_ZTS8KvAppend"},
      {"attn.spv", "_ZTS8AttnCore"},     {"scalesmax.spv", "_ZTS9ScalesMax"},
      {"quantize.spv", "_ZTS8Quantize"}, {"silumul.spv", "_ZTS7SiluMul"},
      {"resaddf.spv", "_ZTS7ResAddF"},   {"ssmconv.spv", "_ZTS7SsmConv"},
      {"ssmrecur.spv", "_ZTS8SsmRecur"}, {"splitrepeat.spv", "_ZTS11SplitRepeat"},
      {"l2normqk.spv", "_ZTS8L2NormQK"}, {"betag.spv", "_ZTS5BetaG"},
      {"rmsinv.spv", "_ZTS6RmsInv"},     {"normgated.spv", "_ZTS9NormGated"},
      {"embed.spv", "_ZTS5Embed"},       {"argmax1.spv", "_ZTS8ArgmaxS1"},
      {"argmax2.spv", "_ZTS8ArgmaxS2"},
  };
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
    RES,
    CONV,
    RECUR,
    SPLIT2,
    L2,
    BETA,
    RMSI,
    GATE,
    EMBED,
    ARG1,
    ARG2,
    NK
  };
  ze_kernel_handle_t kh[NK] = {nullptr};
  for (int i = 0; i < NK; ++i) {
    std::string mp =
        std::string(spvdir) + "/" + mods[i].file;
    FILE *sf = std::fopen(mp.c_str(), "rb");
    if (!sf) {
      std::fprintf(stderr, "no spv %s\n", mp.c_str());
      return 2;
    }
    std::fseek(sf, 0, SEEK_END);
    size_t nn = std::ftell(sf);
    std::fseek(sf, 0, SEEK_SET);
    std::vector<uint8_t> spv(nn);
    if (std::fread(spv.data(), 1, nn, sf) != nn)
      return 2;
    std::fclose(sf);
    ze_module_handle_t mod = nullptr;
    ze_module_desc_t mddesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                               nullptr,
                               ZE_MODULE_FORMAT_IL_SPIRV,
                               spv.size(),
                               spv.data(),
                               nullptr,
                               nullptr};
    CHECK(zeModuleCreate(ctx, dev, &mddesc, &mod, nullptr));
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           mods[i].entry};
    if (zeKernelCreate(mod, &kd, &kh[i]) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "entry %s not found\n", mods[i].entry);
      return 1;
    }
  }
  CHECK(zeKernelSetGroupSize(kh[ARG1], 256, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[ARG2], 256, 1, 1));
  for (int i = 0; i < NK; ++i)
    if (i != ARG1 && i != ARG2)
      CHECK(zeKernelSetGroupSize(kh[i], 1, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[NORM], 256, 1, 1)); // T6.1 parallel norm

  ze_command_queue_handle_t qq = nullptr;
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandQueueCreate(ctx, dev, &qdesc, &qq));
  ze_fence_handle_t fence = nullptr;
  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK(zeFenceCreate(qq, &fdesc, &fence));
  auto newList = [&]() -> ze_command_list_handle_t {
    ze_command_list_handle_t r = nullptr;
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr,
                                 0, 0};
    if (zeCommandListCreate(ctx, dev, &ld, &r) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "list create failed\n");
      std::exit(1);
    }
    return r;
  };
  auto setarg = [&](ze_kernel_handle_t k, uint32_t idx, size_t sz,
                    const void *p) {
    if (zeKernelSetArgumentValue(k, idx, sz, p) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "setArg %u failed\n", idx);
      std::exit(1);
    }
  };
  struct Rec {
    ze_command_list_handle_t h;
  };
  auto launch = [&](Rec &R, ze_kernel_handle_t k, uint32_t count) {
    ze_group_count_t gc = {count, 1, 1};
    ze_result_t r1 =
        zeCommandListAppendLaunchKernel(R.h, k, &gc, nullptr, 0, nullptr);
    ze_result_t r2 = zeCommandListAppendBarrier(R.h, nullptr, 0, nullptr);
    if (r1 != ZE_RESULT_SUCCESS || r2 != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "append failed\n");
      std::exit(1);
    }
  };
  auto gemvE = [&](Rec &R, const Entry *e, int M, int KK, void *Y) {
    const void *dP = (const uint8_t *)payArena + (e->d_off - pay_lo);
    const void *dSc = (const char *)scArena + (e->sc_off - sc_lo);
    setarg(kh[GEMV], 0, sizeof(void *), &Y);
    setarg(kh[GEMV], 1, sizeof(void *), &dP);
    setarg(kh[GEMV], 2, sizeof(void *), &dSc);
    setarg(kh[GEMV], 3, sizeof(void *), &dQ8);
    setarg(kh[GEMV], 4, sizeof(void *), &dSq);
    setarg(kh[GEMV], 5, sizeof(int), &KK);
    launch(R, kh[GEMV], M);
  };
  auto xq = [&](Rec &R, void *X, int nn) {
    int gg = nn / 128;
    setarg(kh[SCALES], 0, sizeof(void *), &X);
    setarg(kh[SCALES], 1, sizeof(void *), &dSq);
    setarg(kh[SCALES], 2, sizeof(int), &gg);
    launch(R, kh[SCALES], gg);
    setarg(kh[QUANT], 0, sizeof(void *), &X);
    setarg(kh[QUANT], 1, sizeof(void *), &dSq);
    setarg(kh[QUANT], 2, sizeof(void *), &dQ8);
    launch(R, kh[QUANT], nn);
  };
  auto rnorm = [&](Rec &R, void *Y, void *X, void *W) {
    int nn = H;
    setarg(kh[NORM], 0, sizeof(void *), &Y);
    setarg(kh[NORM], 1, sizeof(void *), &X);
    setarg(kh[NORM], 2, sizeof(void *), &W);
    setarg(kh[NORM], 3, sizeof(int), &nn);
    setarg(kh[NORM], 4, (size_t)256 * 8, nullptr); // 2 KB SLM partials
    launch(R, kh[NORM], 1);
  };
  auto res = [&](Rec &R, void *Y, void *A, void *B) {
    setarg(kh[RES], 0, sizeof(void *), &Y);
    setarg(kh[RES], 1, sizeof(void *), &A);
    setarg(kh[RES], 2, sizeof(void *), &B);
    launch(R, kh[RES], H);
  };
  std::vector<int> ids = {248045, 846, 198, 3710, 369, 220, 23, 19,
                          348,    220, 18, 593, 220, 17, 30, 248046,
                          198,    248045, 74455, 198, 248068, 271, 248069, 271};
  if (!opt_ids.empty()) {
    ids = opt_ids;
  } else if ((int)ids.size() > P) {
    ids.resize(P);
  } else {
    while ((int)ids.size() < P)
      ids.push_back(198);
  }
  ids.reserve((size_t)P + (size_t)G);

  // ---- record: embed list ----
  Rec embR{newList()};
  {
    setarg(kh[EMBED], 0, sizeof(void *), &dX);
    setarg(kh[EMBED], 1, sizeof(void *), &embP);
    setarg(kh[EMBED], 2, sizeof(void *), &dCtrl);
    launch(embR, kh[EMBED], H);
    CHECK(zeCommandListClose(embR.h));
  }
  // ---- record: 64 layer lists ----
  std::vector<Rec> layers;
  int n256 = 256, tmax = MAXCTX;
  for (int L = 0; L < 64; ++L) {
    Rec R{newList()};
    void *inN = (char *)dInN + (size_t)L * H * 4;
    void *postN = (char *)dPostN + (size_t)L * H * 4;
    rnorm(R, dH, dX, inN);
    xq(R, dH, H);
    if (is_full(L)) {
      int slot = L / 4;
      gemvE(R, find(nm(L, "self_attn.q_proj.weight")), QW, H, dQ16);
      gemvE(R, find(nm(L, "self_attn.k_proj.weight")), KVW, H, dK16);
      gemvE(R, find(nm(L, "self_attn.v_proj.weight")), KVW, H, dV16);
      setarg(kh[SPLIT], 0, sizeof(void *), &dQ16);
      setarg(kh[SPLIT], 1, sizeof(void *), &dQn);
      setarg(kh[SPLIT], 2, sizeof(void *), &dGate);
      launch(R, kh[SPLIT], QN);
      void *qnW = (char *)dQNW + (size_t)slot * 256 * 4;
      void *knW = (char *)dKNW + (size_t)slot * 256 * 4;
      setarg(kh[BNORM], 0, sizeof(void *), &dQn);
      setarg(kh[BNORM], 1, sizeof(void *), &dQn);
      setarg(kh[BNORM], 2, sizeof(void *), &qnW);
      setarg(kh[BNORM], 3, sizeof(int), &n256);
      launch(R, kh[BNORM], 24);
      setarg(kh[BNORM], 0, sizeof(void *), &dKn);
      setarg(kh[BNORM], 1, sizeof(void *), &dK16);
      setarg(kh[BNORM], 2, sizeof(void *), &knW);
      setarg(kh[BNORM], 3, sizeof(int), &n256);
      launch(R, kh[BNORM], 4);
      setarg(kh[ROPE], 0, sizeof(void *), &dQn);
      setarg(kh[ROPE], 1, sizeof(void *), &dKn);
      setarg(kh[ROPE], 2, sizeof(void *), &dCos);
      setarg(kh[ROPE], 3, sizeof(void *), &dSin);
      setarg(kh[ROPE], 4, sizeof(void *), &dCtrl);
      launch(R, kh[ROPE], 28);
      void *kcS = (char *)dKc + ((size_t)slot * MAXCTX) * 4 * 256 * 2;
      void *vcS = (char *)dVc + ((size_t)slot * MAXCTX) * 4 * 256 * 2;
      setarg(kh[KV], 0, sizeof(void *), &kcS);
      setarg(kh[KV], 1, sizeof(void *), &vcS);
      setarg(kh[KV], 2, sizeof(void *), &dKn);
      setarg(kh[KV], 3, sizeof(void *), &dV16);
      setarg(kh[KV], 4, sizeof(void *), &dCtrl);
      setarg(kh[KV], 5, sizeof(int), &tmax);
      launch(R, kh[KV], KVW);
      setarg(kh[ATTN], 0, sizeof(void *), &dAtt);
      setarg(kh[ATTN], 1, sizeof(void *), &dQn);
      setarg(kh[ATTN], 2, sizeof(void *), &kcS);
      setarg(kh[ATTN], 3, sizeof(void *), &vcS);
      setarg(kh[ATTN], 4, sizeof(void *), &dGate);
      setarg(kh[ATTN], 5, sizeof(void *), &dCtrl);
      setarg(kh[ATTN], 6, sizeof(int), &tmax);
      setarg(kh[ATTN], 7, sizeof(void *), &dWts);
      launch(R, kh[ATTN], 24);
      xq(R, dAtt, V6);
      gemvE(R, find(nm(L, "self_attn.o_proj.weight")), H, V6, dMix);
    } else {
      int sl = L - (L + 1) / 4;
      gemvE(R, find(nm(L, "linear_attn.in_proj_qkv.weight")), C, H, dQKV);
      gemvE(R, find(nm(L, "linear_attn.in_proj_z.weight")), V6, H, dZ);
      gemvE(R, find(nm(L, "linear_attn.in_proj_b.weight")), NH, H, dB);
      gemvE(R, find(nm(L, "linear_attn.in_proj_a.weight")), NH, H, dA);
      void *csS = (char *)dConv + (size_t)sl * C * 3 * 4;
      void *cwS = (char *)dConvW + (size_t)sl * C * 4 * 4;
      setarg(kh[CONV], 0, sizeof(void *), &dMx);
      setarg(kh[CONV], 1, sizeof(void *), &dQKV);
      setarg(kh[CONV], 2, sizeof(void *), &csS);
      setarg(kh[CONV], 3, sizeof(void *), &cwS);
      launch(R, kh[CONV], C);
      setarg(kh[SPLIT2], 0, sizeof(void *), &dMx);
      setarg(kh[SPLIT2], 1, sizeof(void *), &dQ48);
      setarg(kh[SPLIT2], 2, sizeof(void *), &dK48);
      setarg(kh[SPLIT2], 3, sizeof(void *), &dV48);
      launch(R, kh[SPLIT2], V6);
      setarg(kh[L2], 0, sizeof(void *), &dQ48);
      setarg(kh[L2], 1, sizeof(void *), &dK48);
      launch(R, kh[L2], 96);
      void *alS = (char *)dAL + (size_t)sl * 48 * 4;
      void *dtS = (char *)dDT + (size_t)sl * 48 * 4;
      setarg(kh[BETA], 0, sizeof(void *), &dBt);
      setarg(kh[BETA], 1, sizeof(void *), &dG48);
      setarg(kh[BETA], 2, sizeof(void *), &dB);
      setarg(kh[BETA], 3, sizeof(void *), &dA);
      setarg(kh[BETA], 4, sizeof(void *), &alS);
      setarg(kh[BETA], 5, sizeof(void *), &dtS);
      launch(R, kh[BETA], NH);
      void *sS = (char *)dS + (size_t)sl * NH * D * D * 4;
      int dd = D;
      setarg(kh[RECUR], 0, sizeof(void *), &dMx);
      setarg(kh[RECUR], 1, sizeof(void *), &dQ48);
      setarg(kh[RECUR], 2, sizeof(void *), &dK48);
      setarg(kh[RECUR], 3, sizeof(void *), &dV48);
      setarg(kh[RECUR], 4, sizeof(void *), &sS);
      setarg(kh[RECUR], 5, sizeof(void *), &dBt);
      setarg(kh[RECUR], 6, sizeof(void *), &dG48);
      setarg(kh[RECUR], 7, sizeof(int), &dd);
      launch(R, kh[RECUR], NH);
      void *ngS = (char *)dLinN + (size_t)sl * 128 * 4;
      setarg(kh[RMSI], 0, sizeof(void *), &dBt);
      setarg(kh[RMSI], 1, sizeof(void *), &dMx);
      launch(R, kh[RMSI], NH);
      setarg(kh[GATE], 0, sizeof(void *), &dAtt);
      setarg(kh[GATE], 1, sizeof(void *), &dMx);
      setarg(kh[GATE], 2, sizeof(void *), &dZ);
      setarg(kh[GATE], 3, sizeof(void *), &ngS);
      setarg(kh[GATE], 4, sizeof(void *), &dBt);
      launch(R, kh[GATE], V6);
      xq(R, dAtt, V6);
      gemvE(R, find(nm(L, "linear_attn.out_proj.weight")), H, V6, dMix);
    }
    res(R, dTmp, dX, dMix);
    rnorm(R, dH, dTmp, postN);
    xq(R, dH, H);
    gemvE(R, find(nm(L, "mlp.gate_proj.weight")), I, H, dG17);
    gemvE(R, find(nm(L, "mlp.up_proj.weight")), I, H, dU17);
    setarg(kh[SILU], 0, sizeof(void *), &dG17);
    setarg(kh[SILU], 1, sizeof(void *), &dU17);
    setarg(kh[SILU], 2, sizeof(void *), &dG17);
    launch(R, kh[SILU], I);
    xq(R, dG17, I);
    gemvE(R, find(nm(L, "mlp.down_proj.weight")), H, I, dMix);
    res(R, dX, dTmp, dMix);
    CHECK(zeCommandListClose(R.h));
    layers.push_back(R);
  }
  // ---- record: tail (final norm + lm_head + argmax) ----
  Rec tail{newList()};
  {
    rnorm(tail, dH, dX, dFinN);
    xq(tail, dH, H);
    gemvE(tail, find("lm_head.weight"), V, H, dLogits);
    setarg(kh[ARG1], 0, sizeof(void *), &dLogits);
    setarg(kh[ARG1], 1, sizeof(void *), &dPV);
    setarg(kh[ARG1], 2, sizeof(void *), &dPI);
    setarg(kh[ARG1], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG1], 4, (size_t)256 * 4, nullptr);
    launch(tail, kh[ARG1], 64);
    // NOTE: nd_range group size was fixed at handle level (256); regular-list
    // group count carries the scale (64 then 1). Matches T3.9 design.
    setarg(kh[ARG2], 0, sizeof(void *), &dPV);
    setarg(kh[ARG2], 1, sizeof(void *), &dPI);
    setarg(kh[ARG2], 2, sizeof(void *), &dOutT);
    setarg(kh[ARG2], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG2], 4, (size_t)256 * 4, nullptr);
    launch(tail, kh[ARG2], 1);
    CHECK(zeCommandListClose(tail.h));
  }
  std::fprintf(stderr, "[l0] recorded 64 layer lists + embed + tail\n");

  // ---- steady-state token loop: control updates + replays only ----
  // T6.1 profiling (env-gated, zero impact by default): every execute and
  // host segment is timed into per-class buckets; summary at end.
  const bool prof = std::getenv("AINFER_PROFILE") != nullptr;
  // T5.5 token-only steady state (wired 2026-09-11; previously the knob only
  // existed on the SYCL path): AINFER_TOP5=0 skips the 1 MB logits D2H +
  // partial_sort + stats + printf per gen step. Sampler still forces the
  // readback when temp > 0. Default ON preserves reporting behavior.
  const char *top5e = std::getenv("AINFER_TOP5");
  const bool wantTop5 = !(top5e && top5e[0] == '0');
  const bool needLogits = wantTop5 || samp.temp > 0;
  std::map<std::string, std::vector<double>> buckets;
  auto exec = [&](ze_command_list_handle_t hl, const char *label) {
    double t0 = prof ? now_ns() : 0;
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &hl, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    if (prof)
      buckets[label].push_back(now_ns() - t0);
  };
  auto seg = [&](const char *label, double t0) {
    if (prof)
      buckets[label].push_back(now_ns() - t0);
  };
  std::vector<int> generated;
  std::vector<std::vector<int>> tops5;
  std::vector<std::vector<float>> tops5v;
  // T5.2 discipline: report/output capacities reserved up front so the
  // steady-state loop performs zero host allocations (diagnostic dump and
  // profiling buckets excluded — env-gated, off by default).
  generated.reserve((size_t)G + 1);
  tops5.reserve((size_t)G + 1);
  tops5v.reserve((size_t)G + 1);
  ids.reserve((size_t)P + (size_t)G);
  std::vector<float> dbgStates; // AINFER_DUMP0 per-layer slots (step 0)
  bool stopped_eos = false;
  std::vector<float> lg(V);
  std::vector<int> lidx(V);
  std::vector<float> lgS(V); // T7.1 sampler scratch (penalty mutates a copy)
  uint64_t samp_rng = samp.seed;
  for (int step = 0; step < P + G; ++step) {
    int pos = step;
    if (pos >= MAXCTX) {
      std::fprintf(stderr, "context overflow: pos %d >= MAXCTX %d\n", pos,
                   MAXCTX);
      return 3;
    }
    if ((int)ids.size() <= step) {
      std::fprintf(stderr, "internal: missing id at step %d\n", step);
      return 3;
    }
    DecodeControl c{ids[step], pos, pos + 1, -1};
    double tCtl = prof ? now_ns() : 0;
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                        nullptr));
    seg("control", tCtl);
    exec(embR.h, "embed");
    // Diagnostic per-layer dump (step 0 only): mirrors decode.cpp's dump
    // slots so the recorded loop diffs layer-by-layer vs the SYCL oracle.
    const char *dbgDump = std::getenv("AINFER_DUMP0");
    if (dbgDump && step == 0)
      dbgStates.assign(64 * H, 0.0f);
    for (int L = 0; L < 64; ++L) {
      exec(layers[L].h, is_full(L) ? "attn_layer" : "linear_layer");
      if (dbgDump && step == 0)
        CHECK(zeCommandListAppendMemoryCopy(up, dbgStates.data() + (size_t)L * H,
                                            dX, (size_t)H * 4, nullptr, 0,
                                            nullptr));
    }
    if (step >= P - 1) {
      exec(tail.h, "tail");
      int tok = -1;
      double tRb = prof ? now_ns() : 0;
      CHECK(zeCommandListAppendMemoryCopy(up, &tok, dOutT, 4, nullptr, 0,
                                          nullptr));
      if (needLogits) {
        CHECK(zeCommandListAppendMemoryCopy(up, lg.data(), dLogits,
                                            (size_t)V * 4, nullptr, 0,
                                            nullptr));
      }
      seg("readback", tRb);
      double tTk = prof ? now_ns() : 0;
      if (wantTop5) {
        for (int i = 0; i < V; ++i)
          lidx[i] = i;
        std::partial_sort(lidx.begin(), lidx.begin() + 5, lidx.end(),
                          [&](int a, int b) { return lg[a] > lg[b]; });
        seg("topk", tTk);
        double mn = 0, sd = 0;
        for (float v : lg)
          mn += v;
        mn /= lg.size();
        for (float v : lg)
          sd += (v - mn) * (v - mn);
        sd = std::sqrt(sd / lg.size());
        std::printf("TOP5@s%d: ", step);
        for (int i = 0; i < 5; ++i)
          std::printf("%d(%.3f) ", lidx[i], lg[lidx[i]]);
        std::printf("| mean %.3f std %.3f\n", mn, sd);
        tops5.emplace_back(lidx.begin(), lidx.begin() + 5);
        tops5v.emplace_back();
        for (int i = 0; i < 5; ++i)
          tops5v.back().push_back(lg[lidx[i]]);
      }
      if (samp.temp > 0) {
        // T7.1 sampled step: penalty over prompt+output so far, then draw.
        // Logits are already host-side (top5 path); no extra transfer.
        lgS = lg;
        apply_rep_penalty(lgS, ids.data(), (int)ids.size(), samp.rep_penalty);
        tok = sample_token(lgS.data(), V, samp, samp_rng);
        std::printf("sampled (t=%.2f,k=%d,p=%.2f,rep=%.2f,seed=%llu)\n",
                    samp.temp, samp.top_k, samp.top_p, samp.rep_penalty,
                    (unsigned long long)samp.seed);
      }
      generated.push_back(tok);
      if (step >= P - 1 && step < P + G - 1)
        ids.push_back(tok);
      std::printf("step %d pos %d -> token %d\n", step, pos, tok);
      if (step >= P - 1 && (tok == 248046 || tok == 248044)) {
        stopped_eos = true;
        break;
      }
    }
  }

  char sbuf[256];
  std::snprintf(sbuf, sizeof sbuf,
                "{\"device\":\"B60\",\"backend\":\"raw-L0\",\"sampling\":{"
                "\"temp\":%.3f,\"top_k\":%d,\"top_p\":%.3f,\"rep_penalty\":"
                "%.3f,\"seed\":%llu},\"prompt\":[",
                samp.temp, samp.top_k, samp.top_p, samp.rep_penalty,
                (unsigned long long)samp.seed);
  std::string json = sbuf;
  for (size_t i = 0; i < ids.size() && i < (size_t)P; ++i)
    json += (i ? "," : "") + std::to_string(ids[i]);
  json += "],\"generated\":[";
  for (size_t i = 0; i < generated.size(); ++i)
    json += (i ? "," : "") + std::to_string(generated[i]);
  json += "],\"stopped_eos\":";
  json += stopped_eos ? "true" : "false";
  json += ",\"top5_per_step\":[";
  for (size_t s = 0; s < tops5.size(); ++s) {
    json += (s ? "," : "") + std::string("{\"step\":") +
            std::to_string(P - 1 + (int)s) + ",\"top5\":[";
    for (int i = 0; i < 5; ++i)
      json += (i ? "," : "") + std::to_string(tops5[s][i]);
    json += "],\"top5v\":[";
    char vb[64];
    for (int i = 0; i < 5; ++i) {
      std::snprintf(vb, sizeof vb, "%s%.4f", i ? "," : "", tops5v[s][i]);
      json += vb;
    }
    json += "]}";
  }
  json += "]}";
  FILE *o = stdout;
  if (pos.size() > 4) {
    o = std::fopen(pos[4] ? pos[4] : "/dev/stdout", "w");
    if (!o)
      return 1;
  }
  std::fprintf(o, "%s\n", json.c_str());
  if (o != stdout)
    std::fclose(o);
  std::printf("generated: ");
  for (int t : generated)
    std::printf("%d ", t);
  std::printf("\n");
  if (prof) {
    // Per-class profile: count, total ms, mean ms, median ms.
    std::fprintf(stderr, "%-12s %6s %10s %10s %10s\n", "class", "n",
                 "total_ms", "mean_ms", "med_ms");
    for (auto &kv : buckets) {
      auto &v = kv.second;
      std::sort(v.begin(), v.end());
      double tot = 0;
      for (double x : v)
        tot += x;
      std::fprintf(stderr, "%-12s %6zu %10.2f %10.3f %10.3f\n",
                   kv.first.c_str(), v.size(), tot / 1e6,
                   tot / v.size() / 1e6, v[v.size() / 2] / 1e6);
    }
  }
  const char *dbgDump2 = std::getenv("AINFER_DUMP0");
  if (dbgDump2 && !dbgStates.empty()) {
    FILE *df = std::fopen(dbgDump2, "wb");
    if (df) {
      std::fwrite(dbgStates.data(), 4, dbgStates.size(), df);
      std::fclose(df);
      std::printf("dumped %zu floats -> %s\n", dbgStates.size(), dbgDump2);
    }
  }
  return 0;
}
