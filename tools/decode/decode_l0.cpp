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
  const char *impdir = nullptr; // T7.4 cache import: prefill state dir
  int preChunks = 0;            // T7.4 in-process prefill: N 256-tok chunks
  bool opt_mtp = (std::getenv("AINFER_MTP") != nullptr && std::getenv("AINFER_MTP")[0] == '1');
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
    } else if (std::strncmp(argv[i], "--import-caches=", 16) == 0) {
      impdir = argv[i] + 16;
    } else if (std::strcmp(argv[i], "--import-caches") == 0 && i + 1 < argc) {
      impdir = argv[++i]; // space-separated form (equals form also works)
    } else if (std::strncmp(argv[i], "--prefill-chunks=", 17) == 0) {
      preChunks = atoi(argv[i] + 17);
    } else if (std::strcmp(argv[i], "--prefill-chunks") == 0 && i + 1 < argc) {
      preChunks = atoi(argv[++i]); // space-separated form also works
    } else if (std::strcmp(argv[i], "--mtp") == 0) {
      opt_mtp = true;
    } else if (std::strncmp(argv[i], "--ids-file=", 11) == 0) {
      FILE *ff = std::fopen(argv[i] + 11, "r");
      if (!ff) {
        std::fprintf(stderr, "no ids file %s\n", argv[i] + 11);
        return 2;
      }
      char *line = nullptr;
      size_t cap = 0;
      if (getline(&line, &cap, ff) > 0) {
        for (const char *q = line; *q;) {
          opt_ids.push_back(atoi(q));
          while (*q && *q != ',')
            ++q;
          if (*q == ',')
            ++q;
        }
      }
      free(line);
      std::fclose(ff);
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
                         "[--ids=..] [--ids-file=..] [--max-new=..] "
                         "[--import-caches=..] [--prefill-chunks=N] [--mtp]\n");
    return 2;
  }
  if (pos.size() > 5) {
    // Stray positionals are almost always misparsed flags (this once
    // silently voided an import run: --import-caches <dir> without '=').
    std::fprintf(stderr, "unexpected positional arg %s\n", pos[5]);
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
  // T7.4 hybrid-attention experiment (AINFER_ATTN=hybrid): GEMM-form decode
  // attention (QK-DPAS + softmax + WV-DPAS) replacing AttnCore per full
  // layer. Buffers sized by MAXCTX (full-width baked N/K; garbage cols
  // beyond active T are unread or zero-weighted by construction).
  const char *hybe = std::getenv("AINFER_ATTN");
  const bool hyb = hybe && std::strcmp(hybe, "hybrid") == 0;
  void *dQnh = alloc(QN * 2), *dS8 = alloc((size_t)24 * MAXCTX * 4),
       *dW8 = alloc((size_t)24 * MAXCTX * 4),
       *dWh8 = alloc((size_t)24 * MAXCTX * 2),
       // WV partial slabs are 24x256: WvGemm writes O rows with hardcoded
       // stride 256 (like the harness), NOT KMAX stride. A 24xMAXCTX slab
       // here once misfiled every kv row and overflowed the slabs.
       *dWVO = alloc((size_t)4 * 24 * 256 * 4),
       *dWTmp = alloc((size_t)24 * 256 * 4);
  if (hyb)
    std::fprintf(stderr, "[l0] attention: hybrid GEMM-form (experimental)\n");
  void *dG17 = alloc(I * 4), *dU17 = alloc(I * 4), *dQ8 = alloc(I),
       *dSq = alloc(136 * 4);
  void *dB = alloc(NH * 4), *dA = alloc(NH * 4), *dBt = alloc(NH * 4),
       *dG48 = alloc(NH * 4);
  void *dLogits = alloc((size_t)V * 4);
  void *dPV = alloc(64 * 4), *dPI = alloc(64 * 4), *dOutT = alloc(4);
  void *dCtrl = alloc(sizeof(DecodeControl));
  // T6.3 INT8 KV (AINFER_KV8=1): per-token symmetric INT8 caches + fp32
  // row scales (halves 64K KV 4.0 -> 2.0 GiB + scales). Only the active
  // precision is allocated; per-layer pointers branch once below.
  const char *kv8e = std::getenv("AINFER_KV8");
  const bool kv8 = kv8e && kv8e[0] == '1';
  void *dKc = nullptr, *dVc = nullptr;
  void *dKc8 = nullptr, *dVc8 = nullptr, *dKscl = nullptr, *dVscl = nullptr;
  if (kv8) {
    dKc8 = alloc((size_t)16 * MAXCTX * 4 * 256);
    dVc8 = alloc((size_t)16 * MAXCTX * 4 * 256);
    dKscl = alloc((size_t)16 * MAXCTX * 4 * 4);
    dVscl = alloc((size_t)16 * MAXCTX * 4 * 4);
    std::fprintf(stderr, "[l0] KV precision: INT8 per-token\n");
  } else {
    dKc = alloc((size_t)16 * MAXCTX * 4 * 256 * 2),
    dVc = alloc((size_t)16 * MAXCTX * 4 * 256 * 2);
  }
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
  { // zeroed persistent caches (BF16/int8 zeros are zero bytes)
    // T7.4: device-side fill — at 64K the KV zero set is 4 GiB and must never
    // cross PCIe as host uploads. Synchronous immediate list: inline completion.
    const uint8_t z = 0;
    if (kv8) {
      CHECK(zeCommandListAppendMemoryFill(
          up, dKc8, &z, 1, (size_t)16 * MAXCTX * 4 * 256, nullptr, 0,
          nullptr));
      CHECK(zeCommandListAppendMemoryFill(
          up, dVc8, &z, 1, (size_t)16 * MAXCTX * 4 * 256, nullptr, 0,
          nullptr));
      // Zero scales are safe: dequant yields exact 0 for untouched slots,
      // and every appended row overwrites its scales.
      CHECK(zeCommandListAppendMemoryFill(
          up, dKscl, &z, 1, (size_t)16 * MAXCTX * 4 * 4, nullptr, 0,
          nullptr));
      CHECK(zeCommandListAppendMemoryFill(
          up, dVscl, &z, 1, (size_t)16 * MAXCTX * 4 * 4, nullptr, 0,
          nullptr));
    } else {
      CHECK(zeCommandListAppendMemoryFill(
          up, dKc, &z, 1, (size_t)16 * MAXCTX * 4 * 256 * 2, nullptr, 0,
          nullptr));
      CHECK(zeCommandListAppendMemoryFill(
          up, dVc, &z, 1, (size_t)16 * MAXCTX * 4 * 256 * 2, nullptr, 0,
          nullptr));
    }
    CHECK(zeCommandListAppendMemoryFill(up, dConv, &z, 1,
                                        (size_t)48 * C * 3 * 4, nullptr, 0,
                                        nullptr));
    CHECK(zeCommandListAppendMemoryFill(up, dS, &z, 1,
                                        (size_t)48 * NH * D * D * 4, nullptr,
                                        0, nullptr));
  }
  // T7.2 MTP draft engine (gated behind --mtp or AINFER_MTP=1)
  void *dPreE = nullptr, *dPreH = nullptr, *dMInN = nullptr, *dMPostN = nullptr,
       *dMNorm = nullptr, *dMQNW = nullptr, *dMKNW = nullptr;
  void *dDraftE = nullptr, *dDraftEn = nullptr, *dDraftHn = nullptr,
       *dDraftIn = nullptr, *dDraftX = nullptr, *dDraftLogits = nullptr,
       *dDraftOutT = nullptr, *dCtrlDraft = nullptr;
  void *dMKc = nullptr, *dMVc = nullptr;

  // Dual-token verification buffers (T7.2 speculative decode)
  void *dCtrl0 = nullptr, *dCtrl1 = nullptr;
  void *dX0 = nullptr, *dX1 = nullptr;
  void *dH0 = nullptr, *dH1 = nullptr;
  void *dTmp0 = nullptr, *dTmp1 = nullptr;
  void *dMix0 = nullptr, *dMix1 = nullptr;
  void *dQKV0 = nullptr, *dQKV1 = nullptr;
  void *dMx0 = nullptr, *dMx1 = nullptr;
  void *dZ0 = nullptr, *dZ1 = nullptr;
  void *dQ48_0 = nullptr, *dQ48_1 = nullptr;
  void *dK48_0 = nullptr, *dK48_1 = nullptr;
  void *dV48_0 = nullptr, *dV48_1 = nullptr;
  void *dAtt0 = nullptr, *dAtt1 = nullptr;
  void *dQ16_0 = nullptr, *dQ16_1 = nullptr;
  void *dK16_0 = nullptr, *dK16_1 = nullptr;
  void *dV16_0 = nullptr, *dV16_1 = nullptr;
  void *dKn0 = nullptr, *dKn1 = nullptr;
  void *dQn0 = nullptr, *dQn1 = nullptr;
  void *dGate0 = nullptr, *dGate1 = nullptr;
  void *dG17_0 = nullptr, *dG17_1 = nullptr;
  void *dU17_0 = nullptr, *dU17_1 = nullptr;
  void *dQ8_0 = nullptr, *dQ8_1 = nullptr;
  void *dSq_0 = nullptr, *dSq_1 = nullptr;
  void *dB0 = nullptr, *dB1 = nullptr;
  void *dA0 = nullptr, *dA1 = nullptr;
  void *dBt0 = nullptr, *dBt1 = nullptr;
  void *dG48_0 = nullptr, *dG48_1 = nullptr;
  void *dLogits0 = nullptr, *dLogits1 = nullptr;
  void *dPV0 = nullptr, *dPV1 = nullptr;
  void *dPI0 = nullptr, *dPI1 = nullptr;
  void *dOutT0 = nullptr, *dOutT1 = nullptr;
  void *dConvSpec = nullptr, *dSSpec = nullptr;
  // Depth-2 triple lane (T7.2 chained drafts): mirrors the _0/_1 sets.
  void *dCtrl2 = nullptr;
  void *dX2 = nullptr;
  void *dH2 = nullptr;
  void *dTmp2 = nullptr;
  void *dMix2 = nullptr;
  void *dQKV2 = nullptr;
  void *dMx2 = nullptr;
  void *dZ2 = nullptr;
  void *dQ48_2 = nullptr;
  void *dK48_2 = nullptr;
  void *dV48_2 = nullptr;
  void *dAtt2 = nullptr;
  void *dQ16_2 = nullptr;
  void *dK16_2 = nullptr;
  void *dV16_2 = nullptr;
  void *dKn2 = nullptr;
  void *dQn2 = nullptr;
  void *dGate2 = nullptr;
  void *dG17_2 = nullptr;
  void *dU17_2 = nullptr;
  void *dQ8_2 = nullptr;
  void *dSq_2 = nullptr;
  void *dB2 = nullptr;
  void *dA2 = nullptr;
  void *dBt2 = nullptr;
  void *dG48_2 = nullptr;
  void *dLogits2 = nullptr;
  void *dPV2 = nullptr;
  void *dPI2 = nullptr;
  void *dOutT2 = nullptr;
  // Depth-2 second specular level + chained-draft state (T7.2 chained).
  void *dConvSpec2 = nullptr, *dSSpec2 = nullptr;
  void *dChH = nullptr, *dCtrlDraft2 = nullptr;

  if (opt_mtp) {
    const uint8_t z = 0;
    dPreE = alloc(H * 4);
    dPreH = alloc(H * 4);
    dMInN = alloc(H * 4);
    dMPostN = alloc(H * 4);
    dMNorm = alloc(H * 4);
    dMQNW = alloc(256 * 4);
    dMKNW = alloc(256 * 4);
    load_small("mtp.pre_fc_norm_embedding.weight", dPreE, H);
    load_small("mtp.pre_fc_norm_hidden.weight", dPreH, H);
    load_small("mtp.layers.0.input_layernorm.weight", dMInN, H);
    load_small("mtp.layers.0.post_attention_layernorm.weight", dMPostN, H);
    load_small("mtp.norm.weight", dMNorm, H);
    load_small("mtp.layers.0.self_attn.q_norm.weight", dMQNW, 256);
    load_small("mtp.layers.0.self_attn.k_norm.weight", dMKNW, 256);

    dDraftE = alloc(H * 4);
    dDraftEn = alloc(H * 4);
    dDraftHn = alloc(H * 4);
    dDraftIn = alloc((size_t)10240 * 4);
    dDraftX = alloc(H * 4);
    dDraftLogits = alloc((size_t)V * 4);
    dDraftOutT = alloc(4);
    dCtrlDraft = alloc(sizeof(DecodeControl));
    dMKc = alloc((size_t)MAXCTX * 4 * 256 * 2);
    dMVc = alloc((size_t)MAXCTX * 4 * 256 * 2);
    CHECK(zeCommandListAppendMemoryFill(up, dMKc, &z, 1,
                                        (size_t)MAXCTX * 4 * 256 * 2,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryFill(up, dMVc, &z, 1,
                                        (size_t)MAXCTX * 4 * 256 * 2,
                                        nullptr, 0, nullptr));

    dCtrl0 = alloc(sizeof(DecodeControl));
    dCtrl1 = alloc(sizeof(DecodeControl));
    dX0 = alloc(H * 4);
    dX1 = alloc(H * 4);
    dH0 = alloc(H * 4);
    dH1 = alloc(H * 4);
    dTmp0 = alloc(H * 4);
    dTmp1 = alloc(H * 4);
    dMix0 = alloc(H * 4);
    dMix1 = alloc(H * 4);
    dQKV0 = alloc(C * 4);
    dQKV1 = alloc(C * 4);
    dMx0 = alloc(C * 4);
    dMx1 = alloc(C * 4);
    dZ0 = alloc(V6 * 4);
    dZ1 = alloc(V6 * 4);
    dQ48_0 = alloc(V6 * 4);
    dQ48_1 = alloc(V6 * 4);
    dK48_0 = alloc(V6 * 4);
    dK48_1 = alloc(V6 * 4);
    dV48_0 = alloc(V6 * 4);
    dV48_1 = alloc(V6 * 4);
    dAtt0 = alloc(V6 * 4);
    dAtt1 = alloc(V6 * 4);
    dQ16_0 = alloc(QW * 4);
    dQ16_1 = alloc(QW * 4);
    dK16_0 = alloc(KVW * 4);
    dK16_1 = alloc(KVW * 4);
    dV16_0 = alloc(KVW * 4);
    dV16_1 = alloc(KVW * 4);
    dKn0 = alloc(KVW * 4);
    dKn1 = alloc(KVW * 4);
    dQn0 = alloc(QN * 4);
    dQn1 = alloc(QN * 4);
    dGate0 = alloc(QN * 4);
    dGate1 = alloc(QN * 4);
    dG17_0 = alloc(I * 4);
    dG17_1 = alloc(I * 4);
    dU17_0 = alloc(I * 4);
    dU17_1 = alloc(I * 4);
    dQ8_0 = alloc(I);
    dQ8_1 = alloc(I);
    dSq_0 = alloc(136 * 4);
    dSq_1 = alloc(136 * 4);
    dB0 = alloc(NH * 4);
    dB1 = alloc(NH * 4);
    dA0 = alloc(NH * 4);
    dA1 = alloc(NH * 4);
    dBt0 = alloc(NH * 4);
    dBt1 = alloc(NH * 4);
    dG48_0 = alloc(NH * 4);
    dG48_1 = alloc(NH * 4);
    dLogits0 = alloc((size_t)V * 4);
    dLogits1 = alloc((size_t)V * 4);
    dPV0 = alloc(64 * 4);
    dPV1 = alloc(64 * 4);
    dPI0 = alloc(64 * 4);
    dPI1 = alloc(64 * 4);
    dOutT0 = alloc(4);
    dOutT1 = alloc(4);
    dConvSpec = alloc((size_t)48 * C * 3 * 4);
    dSSpec = alloc((size_t)48 * NH * D * D * 4);
    CHECK(zeCommandListAppendMemoryFill(up, dConvSpec, &z, 1,
                                        (size_t)48 * C * 3 * 4, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryFill(up, dSSpec, &z, 1,
                                        (size_t)48 * NH * D * D * 4, nullptr, 0, nullptr));
    dCtrl2 = alloc(sizeof(DecodeControl));
    dX2 = alloc(H * 4);
    dH2 = alloc(H * 4);
    dTmp2 = alloc(H * 4);
    dMix2 = alloc(H * 4);
    dQKV2 = alloc(C * 4);
    dMx2 = alloc(C * 4);
    dZ2 = alloc(V6 * 4);
    dQ48_2 = alloc(V6 * 4);
    dK48_2 = alloc(V6 * 4);
    dV48_2 = alloc(V6 * 4);
    dAtt2 = alloc(V6 * 4);
    dQ16_2 = alloc(QW * 4);
    dK16_2 = alloc(KVW * 4);
    dV16_2 = alloc(KVW * 4);
    dKn2 = alloc(KVW * 4);
    dQn2 = alloc(QN * 4);
    dGate2 = alloc(QN * 4);
    dG17_2 = alloc(I * 4);
    dU17_2 = alloc(I * 4);
    dQ8_2 = alloc(I);
    dSq_2 = alloc(136 * 4);
    dB2 = alloc(NH * 4);
    dA2 = alloc(NH * 4);
    dBt2 = alloc(NH * 4);
    dG48_2 = alloc(NH * 4);
    dLogits2 = alloc((size_t)V * 4);
    dPV2 = alloc(64 * 4);
    dPI2 = alloc(64 * 4);
    dOutT2 = alloc(4);
    dConvSpec2 = alloc((size_t)48 * C * 3 * 4);
    dSSpec2 = alloc((size_t)48 * NH * D * D * 4);
    CHECK(zeCommandListAppendMemoryFill(up, dConvSpec2, &z, 1,
                                        (size_t)48 * C * 3 * 4, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryFill(up, dSSpec2, &z, 1,
                                        (size_t)48 * NH * D * D * 4, nullptr, 0, nullptr));
    dChH = alloc(H * 4);
    dCtrlDraft2 = alloc(sizeof(DecodeControl));
    std::fprintf(stderr, "[l0] MTP draft engine + 2-token verification: enabled\n");
  }
  // T7.4 cache import (file handoff from chunk prefill): overwrites the
  // zeroed arenas with prefilled KV (BF16) + SSM state + final hidden.
  // Dump layout = decode layout at stride P (16 slots back-to-back);
  // import re-strides KV per layer into MAXCTX arenas. INT8-KV import is
  // future work (prefill writes BF16); KV8 + import is rejected.
  int impP = 0;
  if (impdir) {
    if (kv8) {
      std::fprintf(stderr, "import-caches needs BF16 KV (no --kv8)\n");
      return 2;
    }
    char mp[512];
    std::snprintf(mp, sizeof mp, "%s/meta.txt", impdir);
    FILE *mf = std::fopen(mp, "r");
    int mP = 0, mM = 0, mN = 0, mTC = 0;
    if (!mf ||
        std::fscanf(mf, "P=%d M=%d NCH=%d TC=%d", &mP, &mM, &mN, &mTC) != 4) {
      std::fprintf(stderr, "import-caches: bad meta %s\n", mp);
      return 2;
    }
    std::fclose(mf);
    if (mP > P) {
      std::fprintf(stderr, "import-caches: dump P=%d > prompt P=%d\n", mP, P);
      return 2;
    }
    // mP <= P: first mP ids are cached (skipped); ids[mP..P-1] loop-decode
    // as the question; tail runs at P-1 as usual.
    if (mP > MAXCTX) {
      std::fprintf(stderr, "import-caches: P=%d > MAXCTX=%d\n", mP, MAXCTX);
      return 2;
    }
    auto impRaw = [&](const char *nm, std::vector<uint8_t> &hb) {
      std::snprintf(mp, sizeof mp, "%s/%s", impdir, nm);
      FILE *fi = std::fopen(mp, "rb");
      if (!fi) {
        std::fprintf(stderr, "import-caches: no %s\n", mp);
        std::exit(2);
      }
      std::fseek(fi, 0, SEEK_END);
      long nb = std::ftell(fi);
      std::fseek(fi, 0, SEEK_SET);
      hb.assign(nb > 0 ? (size_t)nb : 0, 0);
      size_t nr = hb.empty() ? 0 : std::fread(hb.data(), 1, hb.size(), fi);
      std::fclose(fi);
      if (nr != hb.size()) {
        std::fprintf(stderr, "import-caches: short %s\n", mp);
        std::exit(2);
      }
    };
    std::vector<uint8_t> hkc, hvc;
    impRaw("kc.bin", hkc);
    impRaw("vc.bin", hvc);
    size_t lay = (size_t)mP * 4 * 256 * 2;
    if (hkc.size() != lay * 16 || hvc.size() != lay * 16) {
      std::fprintf(stderr, "import-caches: KV size %zu want %zu\n",
                   hkc.size(), lay * 16);
      return 2;
    }
    for (int l = 0; l < 16; ++l) {
      CHECK(zeCommandListAppendMemoryCopy(
          up, (char *)dKc + (size_t)l * MAXCTX * 4 * 256 * 2,
          hkc.data() + (size_t)l * lay, lay, nullptr, 0, nullptr));
      CHECK(zeCommandListAppendMemoryCopy(
          up, (char *)dVc + (size_t)l * MAXCTX * 4 * 256 * 2,
          hvc.data() + (size_t)l * lay, lay, nullptr, 0, nullptr));
    }
    std::vector<uint8_t> hcv, hss;
    impRaw("conv.bin", hcv);
    impRaw("ssm.bin", hss);
    if (hcv.size() != (size_t)48 * C * 3 * 4 ||
        hss.size() != (size_t)48 * NH * D * D * 4) {
      std::fprintf(stderr, "import-caches: SSM size\n");
      return 2;
    }
    CHECK(zeCommandListAppendMemoryCopy(up, dConv, hcv.data(), hcv.size(),
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dS, hss.data(), hss.size(),
                                        nullptr, 0, nullptr));
    std::vector<uint8_t> hhid;
    impRaw("hidlast.bin", hhid);
    if (hhid.size() != (size_t)H * 4) {
      std::fprintf(stderr, "import-caches: hidden size\n");
      return 2;
    }
    CHECK(zeCommandListAppendMemoryCopy(up, dX, hhid.data(), hhid.size(),
                                        nullptr, 0, nullptr));
    impP = mP;
    std::fprintf(stderr, "[l0] imported prefill caches P=%d from %s\n", impP,
                 impdir);
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
      {"kvappendi8.spv", "_ZTS10KvAppendI8"},
      {"attni8.spv", "_ZTS10AttnCoreI8"},
      {"qkgemm.spv", "_ZTS6QkGemm"},     {"softmaxrow.spv", "_ZTS10SoftmaxRow"},
      {"cvtf32f16.spv", "_ZTS9CvtF32F16"}, {"wvgemm.spv", "_ZTS6WvGemm"},
      {"gatemul.spv", "_ZTS7GateMul"}, // resaddf already loaded as RES
      {"chunkgemm.spv", "_ZTS9ChunkGemm"},
      {"chunkssmconv.spv", "_ZTS12ChunkSsmConv"},
      {"chunkssmrecur.spv", "_ZTS13ChunkSsmRecur"},
      {"chunkrope.spv", "_ZTS9ChunkRope"},
      {"chunkkvappend.spv", "_ZTS13ChunkKvAppend"},
      {"chunkqkgemm.spv", "_ZTS11ChunkQkGemm"},
      {"chunksoftmaxrow.spv", "_ZTS15ChunkSoftmaxRow"},
      {"chunkwvgemm.spv", "_ZTS11ChunkWvGemm"},
      {"concat.spv", "_ZTS7Concat2"},
      {"gemvm2.spv", "_ZTS10Int4GemvM2"},
      {"gemvm3.spv", "_ZTS10Int4GemvM3"},
      {"chunkgemmdb.spv", "_ZTS11ChunkGemmDB"},
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
    KV8,
    ATTN8,
    QKG,
    SMR,
    CVT2,
    WVG,
    GMUL,
    CGEMM,
    CCONV,
    CRECUR,
    CROPE,
    CKV,
    CQK,
    CSM,
    CWV,
    CONCAT,
    GEMVM2,
    GEMVM3,
    CGEMMDB,
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
  CHECK(zeKernelSetGroupSize(kh[QKG], 16, 1, 1)); // DPAS kernels: SG16 groups
  CHECK(zeKernelSetGroupSize(kh[WVG], 16, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[CGEMM], 16, 1, 1)); // chunk prefill: same
  CHECK(zeKernelSetGroupSize(kh[CGEMMDB], 16, 1, 1)); // paired-slice GEMM: same
  CHECK(zeKernelSetGroupSize(kh[CQK], 16, 1, 1));
  CHECK(zeKernelSetGroupSize(kh[CWV], 16, 1, 1));

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
  auto gemvM2 = [&](Rec &R, const Entry *e, int M, int KK, void *Y0, void *Y1) {
    const void *dP = (const uint8_t *)payArena + (e->d_off - pay_lo);
    const void *dSc = (const char *)scArena + (e->sc_off - sc_lo);
    setarg(kh[GEMVM2], 0, sizeof(void *), &Y0);
    setarg(kh[GEMVM2], 1, sizeof(void *), &Y1);
    setarg(kh[GEMVM2], 2, sizeof(void *), &dP);
    setarg(kh[GEMVM2], 3, sizeof(void *), &dSc);
    setarg(kh[GEMVM2], 4, sizeof(void *), &dQ8_0);
    setarg(kh[GEMVM2], 5, sizeof(void *), &dSq_0);
    setarg(kh[GEMVM2], 6, sizeof(void *), &dQ8_1);
    setarg(kh[GEMVM2], 7, sizeof(void *), &dSq_1);
    setarg(kh[GEMVM2], 8, sizeof(int), &KK);
    launch(R, kh[GEMVM2], M);
  };
  // Depth-2 triple GEMV (T7.2 chained drafts): same single weight stream,
  // third activation lane from the lane-2 quant buffers (xq2 into dQ8_2 /
  // dSq_2 before each call, mirroring lanes 0/1).
  auto gemvM3 = [&](Rec &R, const Entry *e, int M, int KK, void *Y0, void *Y1,
                    void *Y2) {
    const void *dP = (const uint8_t *)payArena + (e->d_off - pay_lo);
    const void *dSc = (const char *)scArena + (e->sc_off - sc_lo);
    setarg(kh[GEMVM3], 0, sizeof(void *), &Y0);
    setarg(kh[GEMVM3], 1, sizeof(void *), &Y1);
    setarg(kh[GEMVM3], 2, sizeof(void *), &Y2);
    setarg(kh[GEMVM3], 3, sizeof(void *), &dP);
    setarg(kh[GEMVM3], 4, sizeof(void *), &dSc);
    setarg(kh[GEMVM3], 5, sizeof(void *), &dQ8_0);
    setarg(kh[GEMVM3], 6, sizeof(void *), &dSq_0);
    setarg(kh[GEMVM3], 7, sizeof(void *), &dQ8_1);
    setarg(kh[GEMVM3], 8, sizeof(void *), &dSq_1);
    setarg(kh[GEMVM3], 9, sizeof(void *), &dQ8_2);
    setarg(kh[GEMVM3], 10, sizeof(void *), &dSq_2);
    setarg(kh[GEMVM3], 11, sizeof(int), &KK);
    launch(R, kh[GEMVM3], M);
  };
  auto xq2 = [&](Rec &R, void *X, int nn, void *dstQ8, void *dstSq) {
    int gg = nn / 128;
    setarg(kh[SCALES], 0, sizeof(void *), &X);
    setarg(kh[SCALES], 1, sizeof(void *), &dstSq);
    setarg(kh[SCALES], 2, sizeof(int), &gg);
    launch(R, kh[SCALES], gg);
    setarg(kh[QUANT], 0, sizeof(void *), &X);
    setarg(kh[QUANT], 1, sizeof(void *), &dstSq);
    setarg(kh[QUANT], 2, sizeof(void *), &dstQ8);
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
      if (kv8 && !hyb) {
        void *kcS = (char *)dKc8 + ((size_t)slot * MAXCTX) * 4 * 256;
        void *vcS = (char *)dVc8 + ((size_t)slot * MAXCTX) * 4 * 256;
        void *ksS = (char *)dKscl + ((size_t)slot * MAXCTX) * 4 * 4;
        void *vsS = (char *)dVscl + ((size_t)slot * MAXCTX) * 4 * 4;
        setarg(kh[KV8], 0, sizeof(void *), &kcS);
        setarg(kh[KV8], 1, sizeof(void *), &vcS);
        setarg(kh[KV8], 2, sizeof(void *), &ksS);
        setarg(kh[KV8], 3, sizeof(void *), &vsS);
        setarg(kh[KV8], 4, sizeof(void *), &dKn);
        setarg(kh[KV8], 5, sizeof(void *), &dV16);
        setarg(kh[KV8], 6, sizeof(void *), &dCtrl);
        setarg(kh[KV8], 7, sizeof(int), &tmax);
        // KvAppendI8 is one WI per HEAD (d loop inside) — NOT KVW=1024 like
        // the element-wise BF16 kernel. Launching 1024 overran the heaps
        // 256x (hh up to 1023) and trashed neighboring slots.
        launch(R, kh[KV8], 4);
        setarg(kh[ATTN8], 0, sizeof(void *), &dAtt);
        setarg(kh[ATTN8], 1, sizeof(void *), &dQn);
        setarg(kh[ATTN8], 2, sizeof(void *), &kcS);
        setarg(kh[ATTN8], 3, sizeof(void *), &vcS);
        setarg(kh[ATTN8], 4, sizeof(void *), &ksS);
        setarg(kh[ATTN8], 5, sizeof(void *), &vsS);
        setarg(kh[ATTN8], 6, sizeof(void *), &dGate);
        setarg(kh[ATTN8], 7, sizeof(void *), &dCtrl);
        setarg(kh[ATTN8], 8, sizeof(int), &tmax);
        setarg(kh[ATTN8], 9, sizeof(void *), &dWts);
        launch(R, kh[ATTN8], 24);
      } else {
        void *kcS = (char *)dKc + ((size_t)slot * MAXCTX) * 4 * 256 * 2;
        void *vcS = (char *)dVc + ((size_t)slot * MAXCTX) * 4 * 256 * 2;
        setarg(kh[KV], 0, sizeof(void *), &kcS);
        setarg(kh[KV], 1, sizeof(void *), &vcS);
        setarg(kh[KV], 2, sizeof(void *), &dKn);
        setarg(kh[KV], 3, sizeof(void *), &dV16);
        setarg(kh[KV], 4, sizeof(void *), &dCtrl);
        setarg(kh[KV], 5, sizeof(int), &tmax);
        launch(R, kh[KV], KVW);
        if (hyb) {
          // Hybrid GEMM-form attention (experimental): full MAXCTX width
          // baked (garbage cols beyond active T are unread — softmax is
          // Ctrl-bounded — or zero-weighted: V beyond T is zero-filled).
          // Q fp32 -> fp16.
          setarg(kh[CVT2], 0, sizeof(void *), &dQnh);
          setarg(kh[CVT2], 1, sizeof(void *), &dQn);
          launch(R, kh[CVT2], QN);
          // QK: 4 kv-group appends.
          int nnX = (MAXCTX + 15) / 16;
          setarg(kh[QKG], 0, sizeof(void *), &dQnh);
          setarg(kh[QKG], 1, sizeof(void *), &kcS);
          setarg(kh[QKG], 2, sizeof(void *), &dS8);
          setarg(kh[QKG], 3, sizeof(int), &tmax);
          setarg(kh[QKG], 4, sizeof(int), &tmax);
          setarg(kh[QKG], 6, (size_t)8 * 256 * 2, nullptr);
          setarg(kh[QKG], 7, (size_t)16 * 16 * 2, nullptr);
          setarg(kh[QKG], 8, (size_t)8 * 16 * 4, nullptr);
          for (int kv = 0; kv < 4; ++kv) {
            setarg(kh[QKG], 5, sizeof(int), &kv);
            launch(R, kh[QKG], nnX);
          }
          // Softmax (Ctrl-driven) + W fp32 -> fp16.
          setarg(kh[SMR], 0, sizeof(void *), &dW8);
          setarg(kh[SMR], 1, sizeof(void *), &dS8);
          setarg(kh[SMR], 2, sizeof(void *), &dCtrl);
          setarg(kh[SMR], 3, sizeof(int), &tmax);
          launch(R, kh[SMR], 24);
          setarg(kh[CVT2], 0, sizeof(void *), &dWh8);
          setarg(kh[CVT2], 1, sizeof(void *), &dW8);
          launch(R, kh[CVT2], (uint32_t)((size_t)24 * MAXCTX));
          // WV: 16 appends (kv x K-chunk) into partial slabs + combine.
          setarg(kh[WVG], 0, sizeof(void *), &dWh8);
          setarg(kh[WVG], 1, sizeof(void *), &vcS);
          setarg(kh[WVG], 3, sizeof(int), &tmax);
          setarg(kh[WVG], 4, sizeof(int), &tmax);
          setarg(kh[WVG], 7, (size_t)8 * 16 * 2, nullptr);
          setarg(kh[WVG], 8, (size_t)16 * 16 * 2, nullptr);
          setarg(kh[WVG], 9, (size_t)8 * 16 * 4, nullptr);
          for (int kv = 0; kv < 4; ++kv) {
            setarg(kh[WVG], 5, sizeof(int), &kv);
            for (int cc = 0; cc < 4; ++cc) {
              void *po = (char *)dWVO + (size_t)cc * 24 * 256 * 4;
              setarg(kh[WVG], 2, sizeof(void *), &po);
              setarg(kh[WVG], 6, sizeof(int), &cc);
              launch(R, kh[WVG], 16);
            }
          }
          auto res3 = [&](void *Y, void *A, void *B) {
            setarg(kh[RES], 0, sizeof(void *), &Y);
            setarg(kh[RES], 1, sizeof(void *), &A);
            setarg(kh[RES], 2, sizeof(void *), &B);
            launch(R, kh[RES], (uint32_t)(24 * 256));
          };
          void *p0 = dWVO, *p1 = (char *)dWVO + (size_t)24 * 256 * 4,
               *p2 = (char *)dWVO + (size_t)2 * 24 * 256 * 4,
               *p3 = (char *)dWVO + (size_t)3 * 24 * 256 * 4;
          res3(dWTmp, p0, p1);
          res3(dWTmp, dWTmp, p2);
          res3(dAtt, dWTmp, p3);
          // Gate in place (Y==A elementwise-safe, SiluMul precedent).
          setarg(kh[GMUL], 0, sizeof(void *), &dAtt);
          setarg(kh[GMUL], 1, sizeof(void *), &dAtt);
          setarg(kh[GMUL], 2, sizeof(void *), &dGate);
          launch(R, kh[GMUL], QN);
        } else {
          setarg(kh[ATTN], 0, sizeof(void *), &dAtt);
          setarg(kh[ATTN], 1, sizeof(void *), &dQn);
          setarg(kh[ATTN], 2, sizeof(void *), &kcS);
          setarg(kh[ATTN], 3, sizeof(void *), &vcS);
          setarg(kh[ATTN], 4, sizeof(void *), &dGate);
          setarg(kh[ATTN], 5, sizeof(void *), &dCtrl);
          setarg(kh[ATTN], 6, sizeof(int), &tmax);
          setarg(kh[ATTN], 7, sizeof(void *), &dWts);
          launch(R, kh[ATTN], 24);
        }
      }
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
  // ---- record: MTP draft list (T7.2) ----
  Rec mtpDraftR{nullptr};
  if (opt_mtp) {
    mtpDraftR.h = newList();
    // 1. Embed draft token into dDraftE
    setarg(kh[EMBED], 0, sizeof(void *), &dDraftE);
    setarg(kh[EMBED], 1, sizeof(void *), &embP);
    setarg(kh[EMBED], 2, sizeof(void *), &dCtrlDraft);
    launch(mtpDraftR, kh[EMBED], H);

    // 2. Pre-FC norms
    rnorm(mtpDraftR, dDraftEn, dDraftE, dPreE);
    rnorm(mtpDraftR, dDraftHn, dX, dPreH);

    // 3. Concat [dDraftEn, dDraftHn] -> dDraftIn (10240 floats)
    int nnH = H;
    setarg(kh[CONCAT], 0, sizeof(void *), &dDraftIn);
    setarg(kh[CONCAT], 1, sizeof(void *), &dDraftEn);
    setarg(kh[CONCAT], 2, sizeof(void *), &dDraftHn);
    setarg(kh[CONCAT], 3, sizeof(int), &nnH);
    launch(mtpDraftR, kh[CONCAT], 2 * H);

    // 4. FC projection: 5120 x 10240
    xq(mtpDraftR, dDraftIn, 10240);
    gemvE(mtpDraftR, find("mtp.fc.weight"), H, 10240, dDraftX);

    // 5. 1 full attention layer
    rnorm(mtpDraftR, dH, dDraftX, dMInN);
    xq(mtpDraftR, dH, H);
    gemvE(mtpDraftR, find("mtp.layers.0.self_attn.q_proj.weight"), QW, H, dQ16);
    gemvE(mtpDraftR, find("mtp.layers.0.self_attn.k_proj.weight"), KVW, H, dK16);
    gemvE(mtpDraftR, find("mtp.layers.0.self_attn.v_proj.weight"), KVW, H, dV16);
    setarg(kh[SPLIT], 0, sizeof(void *), &dQ16);
    setarg(kh[SPLIT], 1, sizeof(void *), &dQn);
    setarg(kh[SPLIT], 2, sizeof(void *), &dGate);
    launch(mtpDraftR, kh[SPLIT], QN);
    setarg(kh[BNORM], 0, sizeof(void *), &dQn);
    setarg(kh[BNORM], 1, sizeof(void *), &dQn);
    setarg(kh[BNORM], 2, sizeof(void *), &dMQNW);
    setarg(kh[BNORM], 3, sizeof(int), &n256);
    launch(mtpDraftR, kh[BNORM], 24);
    setarg(kh[BNORM], 0, sizeof(void *), &dKn);
    setarg(kh[BNORM], 1, sizeof(void *), &dK16);
    setarg(kh[BNORM], 2, sizeof(void *), &dMKNW);
    setarg(kh[BNORM], 3, sizeof(int), &n256);
    launch(mtpDraftR, kh[BNORM], 4);
    setarg(kh[ROPE], 0, sizeof(void *), &dQn);
    setarg(kh[ROPE], 1, sizeof(void *), &dKn);
    setarg(kh[ROPE], 2, sizeof(void *), &dCos);
    setarg(kh[ROPE], 3, sizeof(void *), &dSin);
    setarg(kh[ROPE], 4, sizeof(void *), &dCtrlDraft);
    launch(mtpDraftR, kh[ROPE], 28);
    setarg(kh[KV], 0, sizeof(void *), &dMKc);
    setarg(kh[KV], 1, sizeof(void *), &dMVc);
    setarg(kh[KV], 2, sizeof(void *), &dKn);
    setarg(kh[KV], 3, sizeof(void *), &dV16);
    setarg(kh[KV], 4, sizeof(void *), &dCtrlDraft);
    setarg(kh[KV], 5, sizeof(int), &tmax);
    launch(mtpDraftR, kh[KV], KVW);
    setarg(kh[ATTN], 0, sizeof(void *), &dAtt);
    setarg(kh[ATTN], 1, sizeof(void *), &dQn);
    setarg(kh[ATTN], 2, sizeof(void *), &dMKc);
    setarg(kh[ATTN], 3, sizeof(void *), &dMVc);
    setarg(kh[ATTN], 4, sizeof(void *), &dGate);
    setarg(kh[ATTN], 5, sizeof(void *), &dCtrlDraft);
    setarg(kh[ATTN], 6, sizeof(int), &tmax);
    setarg(kh[ATTN], 7, sizeof(void *), &dWts);
    launch(mtpDraftR, kh[ATTN], 24);
    xq(mtpDraftR, dAtt, QN);
    gemvE(mtpDraftR, find("mtp.layers.0.self_attn.o_proj.weight"), H, QN, dMix);
    res(mtpDraftR, dTmp, dDraftX, dMix);
    rnorm(mtpDraftR, dH, dTmp, dMPostN);
    xq(mtpDraftR, dH, H);
    gemvE(mtpDraftR, find("mtp.layers.0.mlp.gate_proj.weight"), I, H, dG17);
    gemvE(mtpDraftR, find("mtp.layers.0.mlp.up_proj.weight"), I, H, dU17);
    setarg(kh[SILU], 0, sizeof(void *), &dG17);
    setarg(kh[SILU], 1, sizeof(void *), &dU17);
    setarg(kh[SILU], 2, sizeof(void *), &dG17);
    launch(mtpDraftR, kh[SILU], I);
    xq(mtpDraftR, dG17, I);
    gemvE(mtpDraftR, find("mtp.layers.0.mlp.down_proj.weight"), H, I, dMix);
    res(mtpDraftR, dDraftX, dTmp, dMix);

    // 6. MTP final norm + lm_head + argmax
    rnorm(mtpDraftR, dH, dDraftX, dMNorm);
    xq(mtpDraftR, dH, H);
    gemvE(mtpDraftR, find("lm_head.weight"), V, H, dDraftLogits);
    setarg(kh[ARG1], 0, sizeof(void *), &dDraftLogits);
    setarg(kh[ARG1], 1, sizeof(void *), &dPV);
    setarg(kh[ARG1], 2, sizeof(void *), &dPI);
    setarg(kh[ARG1], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG1], 4, (size_t)256 * 4, nullptr);
    launch(mtpDraftR, kh[ARG1], 64);
    setarg(kh[ARG2], 0, sizeof(void *), &dPV);
    setarg(kh[ARG2], 1, sizeof(void *), &dPI);
    setarg(kh[ARG2], 2, sizeof(void *), &dDraftOutT);
    setarg(kh[ARG2], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG2], 4, (size_t)256 * 4, nullptr);
    launch(mtpDraftR, kh[ARG2], 1);
    CHECK(zeCommandListClose(mtpDraftR.h));
    std::fprintf(stderr, "[l0] recorded MTP draft list\n");
  }

  // ---- record: dual-token verification lists (T7.2 speculative decode) ----
  Rec embM2{nullptr};
  std::vector<Rec> layersM2;
  Rec tailM2{nullptr};
  Rec commitSpecR{nullptr};
  if (opt_mtp) {
    embM2.h = newList();
    setarg(kh[EMBED], 0, sizeof(void *), &dX0);
    setarg(kh[EMBED], 1, sizeof(void *), &embP);
    setarg(kh[EMBED], 2, sizeof(void *), &dCtrl0);
    launch(embM2, kh[EMBED], H);
    setarg(kh[EMBED], 0, sizeof(void *), &dX1);
    setarg(kh[EMBED], 1, sizeof(void *), &embP);
    setarg(kh[EMBED], 2, sizeof(void *), &dCtrl1);
    launch(embM2, kh[EMBED], H);
    CHECK(zeCommandListClose(embM2.h));

    for (int L = 0; L < 64; ++L) {
      Rec R{newList()};
      void *inN = (char *)dInN + (size_t)L * H * 4;
      void *postN = (char *)dPostN + (size_t)L * H * 4;
      rnorm(R, dH0, dX0, inN);
      rnorm(R, dH1, dX1, inN);
      xq2(R, dH0, H, dQ8_0, dSq_0);
      xq2(R, dH1, H, dQ8_1, dSq_1);
      if (is_full(L)) {
        int slot = L / 4;
        gemvM2(R, find(nm(L, "self_attn.q_proj.weight")), QW, H, dQ16_0, dQ16_1);
        gemvM2(R, find(nm(L, "self_attn.k_proj.weight")), KVW, H, dK16_0, dK16_1);
        gemvM2(R, find(nm(L, "self_attn.v_proj.weight")), KVW, H, dV16_0, dV16_1);
        // Token 0 split & bnorm
        setarg(kh[SPLIT], 0, sizeof(void *), &dQ16_0);
        setarg(kh[SPLIT], 1, sizeof(void *), &dQn0);
        setarg(kh[SPLIT], 2, sizeof(void *), &dGate0);
        launch(R, kh[SPLIT], QN);
        // Token 1 split & bnorm
        setarg(kh[SPLIT], 0, sizeof(void *), &dQ16_1);
        setarg(kh[SPLIT], 1, sizeof(void *), &dQn1);
        setarg(kh[SPLIT], 2, sizeof(void *), &dGate1);
        launch(R, kh[SPLIT], QN);

        void *qnW = (char *)dQNW + (size_t)slot * 256 * 4;
        void *knW = (char *)dKNW + (size_t)slot * 256 * 4;
        setarg(kh[BNORM], 0, sizeof(void *), &dQn0);
        setarg(kh[BNORM], 1, sizeof(void *), &dQn0);
        setarg(kh[BNORM], 2, sizeof(void *), &qnW);
        setarg(kh[BNORM], 3, sizeof(int), &n256);
        launch(R, kh[BNORM], 24);
        setarg(kh[BNORM], 0, sizeof(void *), &dKn0);
        setarg(kh[BNORM], 1, sizeof(void *), &dK16_0);
        setarg(kh[BNORM], 2, sizeof(void *), &knW);
        setarg(kh[BNORM], 3, sizeof(int), &n256);
        launch(R, kh[BNORM], 4);

        setarg(kh[BNORM], 0, sizeof(void *), &dQn1);
        setarg(kh[BNORM], 1, sizeof(void *), &dQn1);
        setarg(kh[BNORM], 2, sizeof(void *), &qnW);
        setarg(kh[BNORM], 3, sizeof(int), &n256);
        launch(R, kh[BNORM], 24);
        setarg(kh[BNORM], 0, sizeof(void *), &dKn1);
        setarg(kh[BNORM], 1, sizeof(void *), &dK16_1);
        setarg(kh[BNORM], 2, sizeof(void *), &knW);
        setarg(kh[BNORM], 3, sizeof(int), &n256);
        launch(R, kh[BNORM], 4);

        // RoPE
        setarg(kh[ROPE], 0, sizeof(void *), &dQn0);
        setarg(kh[ROPE], 1, sizeof(void *), &dKn0);
        setarg(kh[ROPE], 2, sizeof(void *), &dCos);
        setarg(kh[ROPE], 3, sizeof(void *), &dSin);
        setarg(kh[ROPE], 4, sizeof(void *), &dCtrl0);
        launch(R, kh[ROPE], 28);

        setarg(kh[ROPE], 0, sizeof(void *), &dQn1);
        setarg(kh[ROPE], 1, sizeof(void *), &dKn1);
        setarg(kh[ROPE], 2, sizeof(void *), &dCos);
        setarg(kh[ROPE], 3, sizeof(void *), &dSin);
        setarg(kh[ROPE], 4, sizeof(void *), &dCtrl1);
        launch(R, kh[ROPE], 28);

        if (kv8) {
          void *kcS = (char *)dKc8 + ((size_t)slot * MAXCTX) * 4 * 256;
          void *vcS = (char *)dVc8 + ((size_t)slot * MAXCTX) * 4 * 256;
          void *ksS = (char *)dKscl + ((size_t)slot * MAXCTX) * 4 * 4;
          void *vsS = (char *)dVscl + ((size_t)slot * MAXCTX) * 4 * 4;
          // Token 0 KV append + AttnCore
          setarg(kh[KV8], 0, sizeof(void *), &kcS);
          setarg(kh[KV8], 1, sizeof(void *), &vcS);
          setarg(kh[KV8], 2, sizeof(void *), &ksS);
          setarg(kh[KV8], 3, sizeof(void *), &vsS);
          setarg(kh[KV8], 4, sizeof(void *), &dKn0);
          setarg(kh[KV8], 5, sizeof(void *), &dV16_0);
          setarg(kh[KV8], 6, sizeof(void *), &dCtrl0);
          setarg(kh[KV8], 7, sizeof(int), &tmax);
          launch(R, kh[KV8], 4);
          setarg(kh[ATTN8], 0, sizeof(void *), &dAtt0);
          setarg(kh[ATTN8], 1, sizeof(void *), &dQn0);
          setarg(kh[ATTN8], 2, sizeof(void *), &kcS);
          setarg(kh[ATTN8], 3, sizeof(void *), &vcS);
          setarg(kh[ATTN8], 4, sizeof(void *), &ksS);
          setarg(kh[ATTN8], 5, sizeof(void *), &vsS);
          setarg(kh[ATTN8], 6, sizeof(void *), &dGate0);
          setarg(kh[ATTN8], 7, sizeof(void *), &dCtrl0);
          setarg(kh[ATTN8], 8, sizeof(int), &tmax);
          setarg(kh[ATTN8], 9, sizeof(void *), &dWts);
          launch(R, kh[ATTN8], 24);

          // Token 1 KV append + AttnCore
          setarg(kh[KV8], 0, sizeof(void *), &kcS);
          setarg(kh[KV8], 1, sizeof(void *), &vcS);
          setarg(kh[KV8], 2, sizeof(void *), &ksS);
          setarg(kh[KV8], 3, sizeof(void *), &vsS);
          setarg(kh[KV8], 4, sizeof(void *), &dKn1);
          setarg(kh[KV8], 5, sizeof(void *), &dV16_1);
          setarg(kh[KV8], 6, sizeof(void *), &dCtrl1);
          setarg(kh[KV8], 7, sizeof(int), &tmax);
          launch(R, kh[KV8], 4);
          setarg(kh[ATTN8], 0, sizeof(void *), &dAtt1);
          setarg(kh[ATTN8], 1, sizeof(void *), &dQn1);
          setarg(kh[ATTN8], 2, sizeof(void *), &kcS);
          setarg(kh[ATTN8], 3, sizeof(void *), &vcS);
          setarg(kh[ATTN8], 4, sizeof(void *), &ksS);
          setarg(kh[ATTN8], 5, sizeof(void *), &vsS);
          setarg(kh[ATTN8], 6, sizeof(void *), &dGate1);
          setarg(kh[ATTN8], 7, sizeof(void *), &dCtrl1);
          setarg(kh[ATTN8], 8, sizeof(int), &tmax);
          setarg(kh[ATTN8], 9, sizeof(void *), &dWts);
          launch(R, kh[ATTN8], 24);
        } else {
          void *kcS = (char *)dKc + ((size_t)slot * MAXCTX) * 4 * 256 * 2;
          void *vcS = (char *)dVc + ((size_t)slot * MAXCTX) * 4 * 256 * 2;
          // Token 0 KV append + AttnCore
          setarg(kh[KV], 0, sizeof(void *), &kcS);
          setarg(kh[KV], 1, sizeof(void *), &vcS);
          setarg(kh[KV], 2, sizeof(void *), &dKn0);
          setarg(kh[KV], 3, sizeof(void *), &dV16_0);
          setarg(kh[KV], 4, sizeof(void *), &dCtrl0);
          setarg(kh[KV], 5, sizeof(int), &tmax);
          launch(R, kh[KV], KVW);
          setarg(kh[ATTN], 0, sizeof(void *), &dAtt0);
          setarg(kh[ATTN], 1, sizeof(void *), &dQn0);
          setarg(kh[ATTN], 2, sizeof(void *), &kcS);
          setarg(kh[ATTN], 3, sizeof(void *), &vcS);
          setarg(kh[ATTN], 4, sizeof(void *), &dGate0);
          setarg(kh[ATTN], 5, sizeof(void *), &dCtrl0);
          setarg(kh[ATTN], 6, sizeof(int), &tmax);
          setarg(kh[ATTN], 7, sizeof(void *), &dWts);
          launch(R, kh[ATTN], 24);

          // Token 1 KV append + AttnCore
          setarg(kh[KV], 0, sizeof(void *), &kcS);
          setarg(kh[KV], 1, sizeof(void *), &vcS);
          setarg(kh[KV], 2, sizeof(void *), &dKn1);
          setarg(kh[KV], 3, sizeof(void *), &dV16_1);
          setarg(kh[KV], 4, sizeof(void *), &dCtrl1);
          setarg(kh[KV], 5, sizeof(int), &tmax);
          launch(R, kh[KV], KVW);
          setarg(kh[ATTN], 0, sizeof(void *), &dAtt1);
          setarg(kh[ATTN], 1, sizeof(void *), &dQn1);
          setarg(kh[ATTN], 2, sizeof(void *), &kcS);
          setarg(kh[ATTN], 3, sizeof(void *), &vcS);
          setarg(kh[ATTN], 4, sizeof(void *), &dGate1);
          setarg(kh[ATTN], 5, sizeof(void *), &dCtrl1);
          setarg(kh[ATTN], 6, sizeof(int), &tmax);
          setarg(kh[ATTN], 7, sizeof(void *), &dWts);
          launch(R, kh[ATTN], 24);
        }
        xq2(R, dAtt0, V6, dQ8_0, dSq_0);
        xq2(R, dAtt1, V6, dQ8_1, dSq_1);
        gemvM2(R, find(nm(L, "self_attn.o_proj.weight")), H, V6, dMix0, dMix1);
      } else {
        int sl = L - (L + 1) / 4;
        gemvM2(R, find(nm(L, "linear_attn.in_proj_qkv.weight")), C, H, dQKV0, dQKV1);
        gemvM2(R, find(nm(L, "linear_attn.in_proj_z.weight")), V6, H, dZ0, dZ1);
        gemvM2(R, find(nm(L, "linear_attn.in_proj_b.weight")), NH, H, dB0, dB1);
        gemvM2(R, find(nm(L, "linear_attn.in_proj_a.weight")), NH, H, dA0, dA1);

        void *csS = (char *)dConv + (size_t)sl * C * 3 * 4;
        void *cwS = (char *)dConvW + (size_t)sl * C * 4 * 4;
        void *sS = (char *)dS + (size_t)sl * NH * D * D * 4;
        void *alS = (char *)dAL + (size_t)sl * 48 * 4;
        void *dtS = (char *)dDT + (size_t)sl * 48 * 4;
        void *ngS = (char *)dLinN + (size_t)sl * 128 * 4;
        void *csSpec = (char *)dConvSpec + (size_t)sl * C * 3 * 4;
        void *sSpec = (char *)dSSpec + (size_t)sl * NH * D * D * 4;
        int dd = D;

        // Token 0: mutates primary state csS, sS
        setarg(kh[CONV], 0, sizeof(void *), &dMx0);
        setarg(kh[CONV], 1, sizeof(void *), &dQKV0);
        setarg(kh[CONV], 2, sizeof(void *), &csS);
        setarg(kh[CONV], 3, sizeof(void *), &cwS);
        launch(R, kh[CONV], C);
        setarg(kh[SPLIT2], 0, sizeof(void *), &dMx0);
        setarg(kh[SPLIT2], 1, sizeof(void *), &dQ48_0);
        setarg(kh[SPLIT2], 2, sizeof(void *), &dK48_0);
        setarg(kh[SPLIT2], 3, sizeof(void *), &dV48_0);
        launch(R, kh[SPLIT2], V6);
        setarg(kh[L2], 0, sizeof(void *), &dQ48_0);
        setarg(kh[L2], 1, sizeof(void *), &dK48_0);
        launch(R, kh[L2], 96);
        setarg(kh[BETA], 0, sizeof(void *), &dBt0);
        setarg(kh[BETA], 1, sizeof(void *), &dG48_0);
        setarg(kh[BETA], 2, sizeof(void *), &dB0);
        setarg(kh[BETA], 3, sizeof(void *), &dA0);
        setarg(kh[BETA], 4, sizeof(void *), &alS);
        setarg(kh[BETA], 5, sizeof(void *), &dtS);
        launch(R, kh[BETA], NH);
        setarg(kh[RECUR], 0, sizeof(void *), &dMx0);
        setarg(kh[RECUR], 1, sizeof(void *), &dQ48_0);
        setarg(kh[RECUR], 2, sizeof(void *), &dK48_0);
        setarg(kh[RECUR], 3, sizeof(void *), &dV48_0);
        setarg(kh[RECUR], 4, sizeof(void *), &sS);
        setarg(kh[RECUR], 5, sizeof(void *), &dBt0);
        setarg(kh[RECUR], 6, sizeof(void *), &dG48_0);
        setarg(kh[RECUR], 7, sizeof(int), &dd);
        launch(R, kh[RECUR], NH);
        setarg(kh[RMSI], 0, sizeof(void *), &dBt0);
        setarg(kh[RMSI], 1, sizeof(void *), &dMx0);
        launch(R, kh[RMSI], NH);
        setarg(kh[GATE], 0, sizeof(void *), &dAtt0);
        setarg(kh[GATE], 1, sizeof(void *), &dMx0);
        setarg(kh[GATE], 2, sizeof(void *), &dZ0);
        setarg(kh[GATE], 3, sizeof(void *), &ngS);
        setarg(kh[GATE], 4, sizeof(void *), &dBt0);
        launch(R, kh[GATE], V6);

        // Copy primary state (updated by Token 0) into specular state
        CHECK(zeCommandListAppendMemoryCopy(R.h, csSpec, csS, (size_t)C * 3 * 4, nullptr, 0, nullptr));
        CHECK(zeCommandListAppendMemoryCopy(R.h, sSpec, sS, (size_t)NH * D * D * 4, nullptr, 0, nullptr));
        CHECK(zeCommandListAppendBarrier(R.h, nullptr, 0, nullptr));

        // Token 1: mutates specular state csSpec, sSpec
        setarg(kh[CONV], 0, sizeof(void *), &dMx1);
        setarg(kh[CONV], 1, sizeof(void *), &dQKV1);
        setarg(kh[CONV], 2, sizeof(void *), &csSpec);
        setarg(kh[CONV], 3, sizeof(void *), &cwS);
        launch(R, kh[CONV], C);
        setarg(kh[SPLIT2], 0, sizeof(void *), &dMx1);
        setarg(kh[SPLIT2], 1, sizeof(void *), &dQ48_1);
        setarg(kh[SPLIT2], 2, sizeof(void *), &dK48_1);
        setarg(kh[SPLIT2], 3, sizeof(void *), &dV48_1);
        launch(R, kh[SPLIT2], V6);
        setarg(kh[L2], 0, sizeof(void *), &dQ48_1);
        setarg(kh[L2], 1, sizeof(void *), &dK48_1);
        launch(R, kh[L2], 96);
        setarg(kh[BETA], 0, sizeof(void *), &dBt1);
        setarg(kh[BETA], 1, sizeof(void *), &dG48_1);
        setarg(kh[BETA], 2, sizeof(void *), &dB1);
        setarg(kh[BETA], 3, sizeof(void *), &dA1);
        setarg(kh[BETA], 4, sizeof(void *), &alS);
        setarg(kh[BETA], 5, sizeof(void *), &dtS);
        launch(R, kh[BETA], NH);
        setarg(kh[RECUR], 0, sizeof(void *), &dMx1);
        setarg(kh[RECUR], 1, sizeof(void *), &dQ48_1);
        setarg(kh[RECUR], 2, sizeof(void *), &dK48_1);
        setarg(kh[RECUR], 3, sizeof(void *), &dV48_1);
        setarg(kh[RECUR], 4, sizeof(void *), &sSpec);
        setarg(kh[RECUR], 5, sizeof(void *), &dBt1);
        setarg(kh[RECUR], 6, sizeof(void *), &dG48_1);
        setarg(kh[RECUR], 7, sizeof(int), &dd);
        launch(R, kh[RECUR], NH);
        setarg(kh[RMSI], 0, sizeof(void *), &dBt1);
        setarg(kh[RMSI], 1, sizeof(void *), &dMx1);
        launch(R, kh[RMSI], NH);
        setarg(kh[GATE], 0, sizeof(void *), &dAtt1);
        setarg(kh[GATE], 1, sizeof(void *), &dMx1);
        setarg(kh[GATE], 2, sizeof(void *), &dZ1);
        setarg(kh[GATE], 3, sizeof(void *), &ngS);
        setarg(kh[GATE], 4, sizeof(void *), &dBt1);
        launch(R, kh[GATE], V6);

        xq2(R, dAtt0, V6, dQ8_0, dSq_0);
        xq2(R, dAtt1, V6, dQ8_1, dSq_1);
        gemvM2(R, find(nm(L, "linear_attn.out_proj.weight")), H, V6, dMix0, dMix1);
      }
      res(R, dTmp0, dX0, dMix0);
      res(R, dTmp1, dX1, dMix1);
      rnorm(R, dH0, dTmp0, postN);
      rnorm(R, dH1, dTmp1, postN);
      xq2(R, dH0, H, dQ8_0, dSq_0);
      xq2(R, dH1, H, dQ8_1, dSq_1);
      gemvM2(R, find(nm(L, "mlp.gate_proj.weight")), I, H, dG17_0, dG17_1);
      gemvM2(R, find(nm(L, "mlp.up_proj.weight")), I, H, dU17_0, dU17_1);
      setarg(kh[SILU], 0, sizeof(void *), &dG17_0);
      setarg(kh[SILU], 1, sizeof(void *), &dU17_0);
      setarg(kh[SILU], 2, sizeof(void *), &dG17_0);
      launch(R, kh[SILU], I);
      setarg(kh[SILU], 0, sizeof(void *), &dG17_1);
      setarg(kh[SILU], 1, sizeof(void *), &dU17_1);
      setarg(kh[SILU], 2, sizeof(void *), &dG17_1);
      launch(R, kh[SILU], I);
      xq2(R, dG17_0, I, dQ8_0, dSq_0);
      xq2(R, dG17_1, I, dQ8_1, dSq_1);
      gemvM2(R, find(nm(L, "mlp.down_proj.weight")), H, I, dMix0, dMix1);
      res(R, dX0, dTmp0, dMix0);
      res(R, dX1, dTmp1, dMix1);
      CHECK(zeCommandListClose(R.h));
      layersM2.push_back(R);
    }

    // Tail for dual-token verification
    tailM2.h = newList();
    rnorm(tailM2, dH0, dX0, dFinN);
    rnorm(tailM2, dH1, dX1, dFinN);
    xq2(tailM2, dH0, H, dQ8_0, dSq_0);
    xq2(tailM2, dH1, H, dQ8_1, dSq_1);
    gemvM2(tailM2, find("lm_head.weight"), V, H, dLogits0, dLogits1);

    // Argmax token 0
    setarg(kh[ARG1], 0, sizeof(void *), &dLogits0);
    setarg(kh[ARG1], 1, sizeof(void *), &dPV0);
    setarg(kh[ARG1], 2, sizeof(void *), &dPI0);
    setarg(kh[ARG1], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG1], 4, (size_t)256 * 4, nullptr);
    launch(tailM2, kh[ARG1], 64);
    setarg(kh[ARG2], 0, sizeof(void *), &dPV0);
    setarg(kh[ARG2], 1, sizeof(void *), &dPI0);
    setarg(kh[ARG2], 2, sizeof(void *), &dOutT0);
    setarg(kh[ARG2], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG2], 4, (size_t)256 * 4, nullptr);
    launch(tailM2, kh[ARG2], 1);

    // Argmax token 1
    setarg(kh[ARG1], 0, sizeof(void *), &dLogits1);
    setarg(kh[ARG1], 1, sizeof(void *), &dPV1);
    setarg(kh[ARG1], 2, sizeof(void *), &dPI1);
    setarg(kh[ARG1], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG1], 4, (size_t)256 * 4, nullptr);
    launch(tailM2, kh[ARG1], 64);
    setarg(kh[ARG2], 0, sizeof(void *), &dPV1);
    setarg(kh[ARG2], 1, sizeof(void *), &dPI1);
    setarg(kh[ARG2], 2, sizeof(void *), &dOutT1);
    setarg(kh[ARG2], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG2], 4, (size_t)256 * 4, nullptr);
    launch(tailM2, kh[ARG2], 1);
    CHECK(zeCommandListClose(tailM2.h));

    // Commit specular state list
    commitSpecR.h = newList();
    CHECK(zeCommandListAppendMemoryCopy(commitSpecR.h, dConv, dConvSpec,
                                        (size_t)48 * C * 3 * 4, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(commitSpecR.h, dS, dSSpec,
                                        (size_t)48 * NH * D * D * 4, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(commitSpecR.h));
    std::fprintf(stderr, "[l0] recorded 64 dual-verification layer lists + embedM2 + tailM2 + commitSpec\n");
  }

  // ---- record: triple-token verification lists (T7.2 chained drafts) ----
  // Structural mirror of the M2 block with a third lane (_2 suffix). Lane 2
  // appends full-attention KV into the PRIMARY slots (slot-indexed: a
  // rejected draft's slots are overwritten by the next round, exactly like
  // lane 1) and mutates the SECOND specular level for linear layers
  // (spec2 = copy of spec1 post-lane-1, then lane-2 update).
  Rec embM3{nullptr};
  std::vector<Rec> layersM3;
  Rec tailM3{nullptr};
  Rec commitSpec2R{nullptr};
  Rec mtpDraft2R{nullptr};
  if (opt_mtp) {
    embM3.h = newList();
    setarg(kh[EMBED], 0, sizeof(void *), &dX0);
    setarg(kh[EMBED], 1, sizeof(void *), &embP);
    setarg(kh[EMBED], 2, sizeof(void *), &dCtrl0);
    launch(embM3, kh[EMBED], H);
    setarg(kh[EMBED], 0, sizeof(void *), &dX1);
    setarg(kh[EMBED], 1, sizeof(void *), &embP);
    setarg(kh[EMBED], 2, sizeof(void *), &dCtrl1);
    launch(embM3, kh[EMBED], H);
    setarg(kh[EMBED], 0, sizeof(void *), &dX2);
    setarg(kh[EMBED], 1, sizeof(void *), &embP);
    setarg(kh[EMBED], 2, sizeof(void *), &dCtrl2);
    launch(embM3, kh[EMBED], H);
    CHECK(zeCommandListClose(embM3.h));

    for (int L = 0; L < 64; ++L) {
      Rec R{newList()};
      void *inN = (char *)dInN + (size_t)L * H * 4;
      void *postN = (char *)dPostN + (size_t)L * H * 4;
      rnorm(R, dH0, dX0, inN);
      rnorm(R, dH1, dX1, inN);
      rnorm(R, dH2, dX2, inN);
      xq2(R, dH0, H, dQ8_0, dSq_0);
      xq2(R, dH1, H, dQ8_1, dSq_1);
      xq2(R, dH2, H, dQ8_2, dSq_2);
      if (is_full(L)) {
        int slot = L / 4;
        gemvM3(R, find(nm(L, "self_attn.q_proj.weight")), QW, H, dQ16_0, dQ16_1, dQ16_2);
        gemvM3(R, find(nm(L, "self_attn.k_proj.weight")), KVW, H, dK16_0, dK16_1, dK16_2);
        gemvM3(R, find(nm(L, "self_attn.v_proj.weight")), KVW, H, dV16_0, dV16_1, dV16_2);
        setarg(kh[SPLIT], 0, sizeof(void *), &dQ16_0);
        setarg(kh[SPLIT], 1, sizeof(void *), &dQn0);
        setarg(kh[SPLIT], 2, sizeof(void *), &dGate0);
        launch(R, kh[SPLIT], QN);
        setarg(kh[SPLIT], 0, sizeof(void *), &dQ16_1);
        setarg(kh[SPLIT], 1, sizeof(void *), &dQn1);
        setarg(kh[SPLIT], 2, sizeof(void *), &dGate1);
        launch(R, kh[SPLIT], QN);
        setarg(kh[SPLIT], 0, sizeof(void *), &dQ16_2);
        setarg(kh[SPLIT], 1, sizeof(void *), &dQn2);
        setarg(kh[SPLIT], 2, sizeof(void *), &dGate2);
        launch(R, kh[SPLIT], QN);

        void *qnW = (char *)dQNW + (size_t)slot * 256 * 4;
        void *knW = (char *)dKNW + (size_t)slot * 256 * 4;
        setarg(kh[BNORM], 0, sizeof(void *), &dQn0);
        setarg(kh[BNORM], 1, sizeof(void *), &dQn0);
        setarg(kh[BNORM], 2, sizeof(void *), &qnW);
        setarg(kh[BNORM], 3, sizeof(int), &n256);
        launch(R, kh[BNORM], 24);
        setarg(kh[BNORM], 0, sizeof(void *), &dKn0);
        setarg(kh[BNORM], 1, sizeof(void *), &dK16_0);
        setarg(kh[BNORM], 2, sizeof(void *), &knW);
        setarg(kh[BNORM], 3, sizeof(int), &n256);
        launch(R, kh[BNORM], 4);
        setarg(kh[BNORM], 0, sizeof(void *), &dQn1);
        setarg(kh[BNORM], 1, sizeof(void *), &dQn1);
        setarg(kh[BNORM], 2, sizeof(void *), &qnW);
        setarg(kh[BNORM], 3, sizeof(int), &n256);
        launch(R, kh[BNORM], 24);
        setarg(kh[BNORM], 0, sizeof(void *), &dKn1);
        setarg(kh[BNORM], 1, sizeof(void *), &dK16_1);
        setarg(kh[BNORM], 2, sizeof(void *), &knW);
        setarg(kh[BNORM], 3, sizeof(int), &n256);
        launch(R, kh[BNORM], 4);
        setarg(kh[BNORM], 0, sizeof(void *), &dQn2);
        setarg(kh[BNORM], 1, sizeof(void *), &dQn2);
        setarg(kh[BNORM], 2, sizeof(void *), &qnW);
        setarg(kh[BNORM], 3, sizeof(int), &n256);
        launch(R, kh[BNORM], 24);
        setarg(kh[BNORM], 0, sizeof(void *), &dKn2);
        setarg(kh[BNORM], 1, sizeof(void *), &dK16_2);
        setarg(kh[BNORM], 2, sizeof(void *), &knW);
        setarg(kh[BNORM], 3, sizeof(int), &n256);
        launch(R, kh[BNORM], 4);

        setarg(kh[ROPE], 0, sizeof(void *), &dQn0);
        setarg(kh[ROPE], 1, sizeof(void *), &dKn0);
        setarg(kh[ROPE], 2, sizeof(void *), &dCos);
        setarg(kh[ROPE], 3, sizeof(void *), &dSin);
        setarg(kh[ROPE], 4, sizeof(void *), &dCtrl0);
        launch(R, kh[ROPE], 28);
        setarg(kh[ROPE], 0, sizeof(void *), &dQn1);
        setarg(kh[ROPE], 1, sizeof(void *), &dKn1);
        setarg(kh[ROPE], 2, sizeof(void *), &dCos);
        setarg(kh[ROPE], 3, sizeof(void *), &dSin);
        setarg(kh[ROPE], 4, sizeof(void *), &dCtrl1);
        launch(R, kh[ROPE], 28);
        setarg(kh[ROPE], 0, sizeof(void *), &dQn2);
        setarg(kh[ROPE], 1, sizeof(void *), &dKn2);
        setarg(kh[ROPE], 2, sizeof(void *), &dCos);
        setarg(kh[ROPE], 3, sizeof(void *), &dSin);
        setarg(kh[ROPE], 4, sizeof(void *), &dCtrl2);
        launch(R, kh[ROPE], 28);

        if (kv8) {
          void *kcS = (char *)dKc8 + ((size_t)slot * MAXCTX) * 4 * 256;
          void *vcS = (char *)dVc8 + ((size_t)slot * MAXCTX) * 4 * 256;
          void *ksS = (char *)dKscl + ((size_t)slot * MAXCTX) * 4 * 4;
          void *vsS = (char *)dVscl + ((size_t)slot * MAXCTX) * 4 * 4;
          setarg(kh[KV8], 0, sizeof(void *), &kcS);
          setarg(kh[KV8], 1, sizeof(void *), &vcS);
          setarg(kh[KV8], 2, sizeof(void *), &ksS);
          setarg(kh[KV8], 3, sizeof(void *), &vsS);
          setarg(kh[KV8], 4, sizeof(void *), &dKn0);
          setarg(kh[KV8], 5, sizeof(void *), &dV16_0);
          setarg(kh[KV8], 6, sizeof(void *), &dCtrl0);
          setarg(kh[KV8], 7, sizeof(int), &tmax);
          launch(R, kh[KV8], 4);
          setarg(kh[ATTN8], 0, sizeof(void *), &dAtt0);
          setarg(kh[ATTN8], 1, sizeof(void *), &dQn0);
          setarg(kh[ATTN8], 2, sizeof(void *), &kcS);
          setarg(kh[ATTN8], 3, sizeof(void *), &vcS);
          setarg(kh[ATTN8], 4, sizeof(void *), &ksS);
          setarg(kh[ATTN8], 5, sizeof(void *), &vsS);
          setarg(kh[ATTN8], 6, sizeof(void *), &dGate0);
          setarg(kh[ATTN8], 7, sizeof(void *), &dCtrl0);
          setarg(kh[ATTN8], 8, sizeof(int), &tmax);
          setarg(kh[ATTN8], 9, sizeof(void *), &dWts);
          launch(R, kh[ATTN8], 24);
          setarg(kh[KV8], 0, sizeof(void *), &kcS);
          setarg(kh[KV8], 1, sizeof(void *), &vcS);
          setarg(kh[KV8], 2, sizeof(void *), &ksS);
          setarg(kh[KV8], 3, sizeof(void *), &vsS);
          setarg(kh[KV8], 4, sizeof(void *), &dKn1);
          setarg(kh[KV8], 5, sizeof(void *), &dV16_1);
          setarg(kh[KV8], 6, sizeof(void *), &dCtrl1);
          setarg(kh[KV8], 7, sizeof(int), &tmax);
          launch(R, kh[KV8], 4);
          setarg(kh[ATTN8], 0, sizeof(void *), &dAtt1);
          setarg(kh[ATTN8], 1, sizeof(void *), &dQn1);
          setarg(kh[ATTN8], 2, sizeof(void *), &kcS);
          setarg(kh[ATTN8], 3, sizeof(void *), &vcS);
          setarg(kh[ATTN8], 4, sizeof(void *), &ksS);
          setarg(kh[ATTN8], 5, sizeof(void *), &vsS);
          setarg(kh[ATTN8], 6, sizeof(void *), &dGate1);
          setarg(kh[ATTN8], 7, sizeof(void *), &dCtrl1);
          setarg(kh[ATTN8], 8, sizeof(int), &tmax);
          setarg(kh[ATTN8], 9, sizeof(void *), &dWts);
          launch(R, kh[ATTN8], 24);
          setarg(kh[KV8], 0, sizeof(void *), &kcS);
          setarg(kh[KV8], 1, sizeof(void *), &vcS);
          setarg(kh[KV8], 2, sizeof(void *), &ksS);
          setarg(kh[KV8], 3, sizeof(void *), &vsS);
          setarg(kh[KV8], 4, sizeof(void *), &dKn2);
          setarg(kh[KV8], 5, sizeof(void *), &dV16_2);
          setarg(kh[KV8], 6, sizeof(void *), &dCtrl2);
          setarg(kh[KV8], 7, sizeof(int), &tmax);
          launch(R, kh[KV8], 4);
          setarg(kh[ATTN8], 0, sizeof(void *), &dAtt2);
          setarg(kh[ATTN8], 1, sizeof(void *), &dQn2);
          setarg(kh[ATTN8], 2, sizeof(void *), &kcS);
          setarg(kh[ATTN8], 3, sizeof(void *), &vcS);
          setarg(kh[ATTN8], 4, sizeof(void *), &ksS);
          setarg(kh[ATTN8], 5, sizeof(void *), &vsS);
          setarg(kh[ATTN8], 6, sizeof(void *), &dGate2);
          setarg(kh[ATTN8], 7, sizeof(void *), &dCtrl2);
          setarg(kh[ATTN8], 8, sizeof(int), &tmax);
          setarg(kh[ATTN8], 9, sizeof(void *), &dWts);
          launch(R, kh[ATTN8], 24);
        } else {
          void *kcS = (char *)dKc + ((size_t)slot * MAXCTX) * 4 * 256 * 2;
          void *vcS = (char *)dVc + ((size_t)slot * MAXCTX) * 4 * 256 * 2;
          setarg(kh[KV], 0, sizeof(void *), &kcS);
          setarg(kh[KV], 1, sizeof(void *), &vcS);
          setarg(kh[KV], 2, sizeof(void *), &dKn0);
          setarg(kh[KV], 3, sizeof(void *), &dV16_0);
          setarg(kh[KV], 4, sizeof(void *), &dCtrl0);
          setarg(kh[KV], 5, sizeof(int), &tmax);
          launch(R, kh[KV], KVW);
          setarg(kh[ATTN], 0, sizeof(void *), &dAtt0);
          setarg(kh[ATTN], 1, sizeof(void *), &dQn0);
          setarg(kh[ATTN], 2, sizeof(void *), &kcS);
          setarg(kh[ATTN], 3, sizeof(void *), &vcS);
          setarg(kh[ATTN], 4, sizeof(void *), &dGate0);
          setarg(kh[ATTN], 5, sizeof(void *), &dCtrl0);
          setarg(kh[ATTN], 6, sizeof(int), &tmax);
          setarg(kh[ATTN], 7, sizeof(void *), &dWts);
          launch(R, kh[ATTN], 24);
          setarg(kh[KV], 0, sizeof(void *), &kcS);
          setarg(kh[KV], 1, sizeof(void *), &vcS);
          setarg(kh[KV], 2, sizeof(void *), &dKn1);
          setarg(kh[KV], 3, sizeof(void *), &dV16_1);
          setarg(kh[KV], 4, sizeof(void *), &dCtrl1);
          setarg(kh[KV], 5, sizeof(int), &tmax);
          launch(R, kh[KV], KVW);
          setarg(kh[ATTN], 0, sizeof(void *), &dAtt1);
          setarg(kh[ATTN], 1, sizeof(void *), &dQn1);
          setarg(kh[ATTN], 2, sizeof(void *), &kcS);
          setarg(kh[ATTN], 3, sizeof(void *), &vcS);
          setarg(kh[ATTN], 4, sizeof(void *), &dGate1);
          setarg(kh[ATTN], 5, sizeof(void *), &dCtrl1);
          setarg(kh[ATTN], 6, sizeof(int), &tmax);
          setarg(kh[ATTN], 7, sizeof(void *), &dWts);
          launch(R, kh[ATTN], 24);
          setarg(kh[KV], 0, sizeof(void *), &kcS);
          setarg(kh[KV], 1, sizeof(void *), &vcS);
          setarg(kh[KV], 2, sizeof(void *), &dKn2);
          setarg(kh[KV], 3, sizeof(void *), &dV16_2);
          setarg(kh[KV], 4, sizeof(void *), &dCtrl2);
          setarg(kh[KV], 5, sizeof(int), &tmax);
          launch(R, kh[KV], KVW);
          setarg(kh[ATTN], 0, sizeof(void *), &dAtt2);
          setarg(kh[ATTN], 1, sizeof(void *), &dQn2);
          setarg(kh[ATTN], 2, sizeof(void *), &kcS);
          setarg(kh[ATTN], 3, sizeof(void *), &vcS);
          setarg(kh[ATTN], 4, sizeof(void *), &dGate2);
          setarg(kh[ATTN], 5, sizeof(void *), &dCtrl2);
          setarg(kh[ATTN], 6, sizeof(int), &tmax);
          setarg(kh[ATTN], 7, sizeof(void *), &dWts);
          launch(R, kh[ATTN], 24);
        }
        xq2(R, dAtt0, V6, dQ8_0, dSq_0);
        xq2(R, dAtt1, V6, dQ8_1, dSq_1);
        xq2(R, dAtt2, V6, dQ8_2, dSq_2);
        gemvM3(R, find(nm(L, "self_attn.o_proj.weight")), H, V6, dMix0, dMix1, dMix2);
      } else {
        int sl = L - (L + 1) / 4;
        gemvM3(R, find(nm(L, "linear_attn.in_proj_qkv.weight")), C, H, dQKV0, dQKV1, dQKV2);
        gemvM3(R, find(nm(L, "linear_attn.in_proj_z.weight")), V6, H, dZ0, dZ1, dZ2);
        gemvM3(R, find(nm(L, "linear_attn.in_proj_b.weight")), NH, H, dB0, dB1, dB2);
        gemvM3(R, find(nm(L, "linear_attn.in_proj_a.weight")), NH, H, dA0, dA1, dA2);

        void *csS = (char *)dConv + (size_t)sl * C * 3 * 4;
        void *cwS = (char *)dConvW + (size_t)sl * C * 4 * 4;
        void *sS = (char *)dS + (size_t)sl * NH * D * D * 4;
        void *alS = (char *)dAL + (size_t)sl * 48 * 4;
        void *dtS = (char *)dDT + (size_t)sl * 48 * 4;
        void *ngS = (char *)dLinN + (size_t)sl * 128 * 4;
        void *csSpec = (char *)dConvSpec + (size_t)sl * C * 3 * 4;
        void *sSpec = (char *)dSSpec + (size_t)sl * NH * D * D * 4;
        void *csSpec2 = (char *)dConvSpec2 + (size_t)sl * C * 3 * 4;
        void *sSpec2 = (char *)dSSpec2 + (size_t)sl * NH * D * D * 4;
        int dd = D;

        // Token 0: mutates primary state csS, sS
        setarg(kh[CONV], 0, sizeof(void *), &dMx0);
        setarg(kh[CONV], 1, sizeof(void *), &dQKV0);
        setarg(kh[CONV], 2, sizeof(void *), &csS);
        setarg(kh[CONV], 3, sizeof(void *), &cwS);
        launch(R, kh[CONV], C);
        setarg(kh[SPLIT2], 0, sizeof(void *), &dMx0);
        setarg(kh[SPLIT2], 1, sizeof(void *), &dQ48_0);
        setarg(kh[SPLIT2], 2, sizeof(void *), &dK48_0);
        setarg(kh[SPLIT2], 3, sizeof(void *), &dV48_0);
        launch(R, kh[SPLIT2], V6);
        setarg(kh[L2], 0, sizeof(void *), &dQ48_0);
        setarg(kh[L2], 1, sizeof(void *), &dK48_0);
        launch(R, kh[L2], 96);
        setarg(kh[BETA], 0, sizeof(void *), &dBt0);
        setarg(kh[BETA], 1, sizeof(void *), &dG48_0);
        setarg(kh[BETA], 2, sizeof(void *), &dB0);
        setarg(kh[BETA], 3, sizeof(void *), &dA0);
        setarg(kh[BETA], 4, sizeof(void *), &alS);
        setarg(kh[BETA], 5, sizeof(void *), &dtS);
        launch(R, kh[BETA], NH);
        setarg(kh[RECUR], 0, sizeof(void *), &dMx0);
        setarg(kh[RECUR], 1, sizeof(void *), &dQ48_0);
        setarg(kh[RECUR], 2, sizeof(void *), &dK48_0);
        setarg(kh[RECUR], 3, sizeof(void *), &dV48_0);
        setarg(kh[RECUR], 4, sizeof(void *), &sS);
        setarg(kh[RECUR], 5, sizeof(void *), &dBt0);
        setarg(kh[RECUR], 6, sizeof(void *), &dG48_0);
        setarg(kh[RECUR], 7, sizeof(int), &dd);
        launch(R, kh[RECUR], NH);
        setarg(kh[RMSI], 0, sizeof(void *), &dBt0);
        setarg(kh[RMSI], 1, sizeof(void *), &dMx0);
        launch(R, kh[RMSI], NH);
        setarg(kh[GATE], 0, sizeof(void *), &dAtt0);
        setarg(kh[GATE], 1, sizeof(void *), &dMx0);
        setarg(kh[GATE], 2, sizeof(void *), &dZ0);
        setarg(kh[GATE], 3, sizeof(void *), &ngS);
        setarg(kh[GATE], 4, sizeof(void *), &dBt0);
        launch(R, kh[GATE], V6);

        // Copy primary state (updated by Token 0) into specular state
        CHECK(zeCommandListAppendMemoryCopy(R.h, csSpec, csS, (size_t)C * 3 * 4, nullptr, 0, nullptr));
        CHECK(zeCommandListAppendMemoryCopy(R.h, sSpec, sS, (size_t)NH * D * D * 4, nullptr, 0, nullptr));
        CHECK(zeCommandListAppendBarrier(R.h, nullptr, 0, nullptr));

        // Token 1: mutates specular state csSpec, sSpec
        setarg(kh[CONV], 0, sizeof(void *), &dMx1);
        setarg(kh[CONV], 1, sizeof(void *), &dQKV1);
        setarg(kh[CONV], 2, sizeof(void *), &csSpec);
        setarg(kh[CONV], 3, sizeof(void *), &cwS);
        launch(R, kh[CONV], C);
        setarg(kh[SPLIT2], 0, sizeof(void *), &dMx1);
        setarg(kh[SPLIT2], 1, sizeof(void *), &dQ48_1);
        setarg(kh[SPLIT2], 2, sizeof(void *), &dK48_1);
        setarg(kh[SPLIT2], 3, sizeof(void *), &dV48_1);
        launch(R, kh[SPLIT2], V6);
        setarg(kh[L2], 0, sizeof(void *), &dQ48_1);
        setarg(kh[L2], 1, sizeof(void *), &dK48_1);
        launch(R, kh[L2], 96);
        setarg(kh[BETA], 0, sizeof(void *), &dBt1);
        setarg(kh[BETA], 1, sizeof(void *), &dG48_1);
        setarg(kh[BETA], 2, sizeof(void *), &dB1);
        setarg(kh[BETA], 3, sizeof(void *), &dA1);
        setarg(kh[BETA], 4, sizeof(void *), &alS);
        setarg(kh[BETA], 5, sizeof(void *), &dtS);
        launch(R, kh[BETA], NH);
        setarg(kh[RECUR], 0, sizeof(void *), &dMx1);
        setarg(kh[RECUR], 1, sizeof(void *), &dQ48_1);
        setarg(kh[RECUR], 2, sizeof(void *), &dK48_1);
        setarg(kh[RECUR], 3, sizeof(void *), &dV48_1);
        setarg(kh[RECUR], 4, sizeof(void *), &sSpec);
        setarg(kh[RECUR], 5, sizeof(void *), &dBt1);
        setarg(kh[RECUR], 6, sizeof(void *), &dG48_1);
        setarg(kh[RECUR], 7, sizeof(int), &dd);
        launch(R, kh[RECUR], NH);
        setarg(kh[RMSI], 0, sizeof(void *), &dBt1);
        setarg(kh[RMSI], 1, sizeof(void *), &dMx1);
        launch(R, kh[RMSI], NH);
        setarg(kh[GATE], 0, sizeof(void *), &dAtt1);
        setarg(kh[GATE], 1, sizeof(void *), &dMx1);
        setarg(kh[GATE], 2, sizeof(void *), &dZ1);
        setarg(kh[GATE], 3, sizeof(void *), &ngS);
        setarg(kh[GATE], 4, sizeof(void *), &dBt1);
        launch(R, kh[GATE], V6);

        // Copy specular state (updated by Token 1) into second level
        CHECK(zeCommandListAppendMemoryCopy(R.h, csSpec2, csSpec, (size_t)C * 3 * 4, nullptr, 0, nullptr));
        CHECK(zeCommandListAppendMemoryCopy(R.h, sSpec2, sSpec, (size_t)NH * D * D * 4, nullptr, 0, nullptr));
        CHECK(zeCommandListAppendBarrier(R.h, nullptr, 0, nullptr));

        // Token 2: mutates second-level specular state csSpec2, sSpec2
        setarg(kh[CONV], 0, sizeof(void *), &dMx2);
        setarg(kh[CONV], 1, sizeof(void *), &dQKV2);
        setarg(kh[CONV], 2, sizeof(void *), &csSpec2);
        setarg(kh[CONV], 3, sizeof(void *), &cwS);
        launch(R, kh[CONV], C);
        setarg(kh[SPLIT2], 0, sizeof(void *), &dMx2);
        setarg(kh[SPLIT2], 1, sizeof(void *), &dQ48_2);
        setarg(kh[SPLIT2], 2, sizeof(void *), &dK48_2);
        setarg(kh[SPLIT2], 3, sizeof(void *), &dV48_2);
        launch(R, kh[SPLIT2], V6);
        setarg(kh[L2], 0, sizeof(void *), &dQ48_2);
        setarg(kh[L2], 1, sizeof(void *), &dK48_2);
        launch(R, kh[L2], 96);
        setarg(kh[BETA], 0, sizeof(void *), &dBt2);
        setarg(kh[BETA], 1, sizeof(void *), &dG48_2);
        setarg(kh[BETA], 2, sizeof(void *), &dB2);
        setarg(kh[BETA], 3, sizeof(void *), &dA2);
        setarg(kh[BETA], 4, sizeof(void *), &alS);
        setarg(kh[BETA], 5, sizeof(void *), &dtS);
        launch(R, kh[BETA], NH);
        setarg(kh[RECUR], 0, sizeof(void *), &dMx2);
        setarg(kh[RECUR], 1, sizeof(void *), &dQ48_2);
        setarg(kh[RECUR], 2, sizeof(void *), &dK48_2);
        setarg(kh[RECUR], 3, sizeof(void *), &dV48_2);
        setarg(kh[RECUR], 4, sizeof(void *), &sSpec2);
        setarg(kh[RECUR], 5, sizeof(void *), &dBt2);
        setarg(kh[RECUR], 6, sizeof(void *), &dG48_2);
        setarg(kh[RECUR], 7, sizeof(int), &dd);
        launch(R, kh[RECUR], NH);
        setarg(kh[RMSI], 0, sizeof(void *), &dBt2);
        setarg(kh[RMSI], 1, sizeof(void *), &dMx2);
        launch(R, kh[RMSI], NH);
        setarg(kh[GATE], 0, sizeof(void *), &dAtt2);
        setarg(kh[GATE], 1, sizeof(void *), &dMx2);
        setarg(kh[GATE], 2, sizeof(void *), &dZ2);
        setarg(kh[GATE], 3, sizeof(void *), &ngS);
        setarg(kh[GATE], 4, sizeof(void *), &dBt2);
        launch(R, kh[GATE], V6);

        xq2(R, dAtt0, V6, dQ8_0, dSq_0);
        xq2(R, dAtt1, V6, dQ8_1, dSq_1);
        xq2(R, dAtt2, V6, dQ8_2, dSq_2);
        gemvM3(R, find(nm(L, "linear_attn.out_proj.weight")), H, V6, dMix0, dMix1, dMix2);
      }
      res(R, dTmp0, dX0, dMix0);
      res(R, dTmp1, dX1, dMix1);
      res(R, dTmp2, dX2, dMix2);
      rnorm(R, dH0, dTmp0, postN);
      rnorm(R, dH1, dTmp1, postN);
      rnorm(R, dH2, dTmp2, postN);
      xq2(R, dH0, H, dQ8_0, dSq_0);
      xq2(R, dH1, H, dQ8_1, dSq_1);
      xq2(R, dH2, H, dQ8_2, dSq_2);
      gemvM3(R, find(nm(L, "mlp.gate_proj.weight")), I, H, dG17_0, dG17_1, dG17_2);
      gemvM3(R, find(nm(L, "mlp.up_proj.weight")), I, H, dU17_0, dU17_1, dU17_2);
      setarg(kh[SILU], 0, sizeof(void *), &dG17_0);
      setarg(kh[SILU], 1, sizeof(void *), &dU17_0);
      setarg(kh[SILU], 2, sizeof(void *), &dG17_0);
      launch(R, kh[SILU], I);
      setarg(kh[SILU], 0, sizeof(void *), &dG17_1);
      setarg(kh[SILU], 1, sizeof(void *), &dU17_1);
      setarg(kh[SILU], 2, sizeof(void *), &dG17_1);
      launch(R, kh[SILU], I);
      setarg(kh[SILU], 0, sizeof(void *), &dG17_2);
      setarg(kh[SILU], 1, sizeof(void *), &dU17_2);
      setarg(kh[SILU], 2, sizeof(void *), &dG17_2);
      launch(R, kh[SILU], I);
      xq2(R, dG17_0, I, dQ8_0, dSq_0);
      xq2(R, dG17_1, I, dQ8_1, dSq_1);
      xq2(R, dG17_2, I, dQ8_2, dSq_2);
      gemvM3(R, find(nm(L, "mlp.down_proj.weight")), H, I, dMix0, dMix1, dMix2);
      res(R, dX0, dTmp0, dMix0);
      res(R, dX1, dTmp1, dMix1);
      res(R, dX2, dTmp2, dMix2);
      CHECK(zeCommandListClose(R.h));
      layersM3.push_back(R);
    }

    // Tail for triple-token verification
    tailM3.h = newList();
    rnorm(tailM3, dH0, dX0, dFinN);
    rnorm(tailM3, dH1, dX1, dFinN);
    rnorm(tailM3, dH2, dX2, dFinN);
    xq2(tailM3, dH0, H, dQ8_0, dSq_0);
    xq2(tailM3, dH1, H, dQ8_1, dSq_1);
    xq2(tailM3, dH2, H, dQ8_2, dSq_2);
    gemvM3(tailM3, find("lm_head.weight"), V, H, dLogits0, dLogits1, dLogits2);

    // Argmax token 0
    setarg(kh[ARG1], 0, sizeof(void *), &dLogits0);
    setarg(kh[ARG1], 1, sizeof(void *), &dPV0);
    setarg(kh[ARG1], 2, sizeof(void *), &dPI0);
    setarg(kh[ARG1], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG1], 4, (size_t)256 * 4, nullptr);
    launch(tailM3, kh[ARG1], 64);
    setarg(kh[ARG2], 0, sizeof(void *), &dPV0);
    setarg(kh[ARG2], 1, sizeof(void *), &dPI0);
    setarg(kh[ARG2], 2, sizeof(void *), &dOutT0);
    setarg(kh[ARG2], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG2], 4, (size_t)256 * 4, nullptr);
    launch(tailM3, kh[ARG2], 1);

    // Argmax token 1
    setarg(kh[ARG1], 0, sizeof(void *), &dLogits1);
    setarg(kh[ARG1], 1, sizeof(void *), &dPV1);
    setarg(kh[ARG1], 2, sizeof(void *), &dPI1);
    setarg(kh[ARG1], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG1], 4, (size_t)256 * 4, nullptr);
    launch(tailM3, kh[ARG1], 64);
    setarg(kh[ARG2], 0, sizeof(void *), &dPV1);
    setarg(kh[ARG2], 1, sizeof(void *), &dPI1);
    setarg(kh[ARG2], 2, sizeof(void *), &dOutT1);
    setarg(kh[ARG2], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG2], 4, (size_t)256 * 4, nullptr);
    launch(tailM3, kh[ARG2], 1);

    // Argmax token 2
    setarg(kh[ARG1], 0, sizeof(void *), &dLogits2);
    setarg(kh[ARG1], 1, sizeof(void *), &dPV2);
    setarg(kh[ARG1], 2, sizeof(void *), &dPI2);
    setarg(kh[ARG1], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG1], 4, (size_t)256 * 4, nullptr);
    launch(tailM3, kh[ARG1], 64);
    setarg(kh[ARG2], 0, sizeof(void *), &dPV2);
    setarg(kh[ARG2], 1, sizeof(void *), &dPI2);
    setarg(kh[ARG2], 2, sizeof(void *), &dOutT2);
    setarg(kh[ARG2], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG2], 4, (size_t)256 * 4, nullptr);
    launch(tailM3, kh[ARG2], 1);
    CHECK(zeCommandListClose(tailM3.h));

    // Commit second-level specular state list (accept-2 path)
    commitSpec2R.h = newList();
    CHECK(zeCommandListAppendMemoryCopy(commitSpec2R.h, dConv, dConvSpec2,
                                        (size_t)48 * C * 3 * 4, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(commitSpec2R.h, dS, dSSpec2,
                                        (size_t)48 * NH * D * D * 4, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(commitSpec2R.h));

    // Chained second draft list: re-applies the MTP module to (embed(d1),
    // saved draft-1 hidden dChH). Reuses the draft buffers (d1 already read
    // to host, dDraftX saved to dChH before exec). Own control dCtrlDraft2
    // bound at record time for RoPE/KV/Attn (MTP KV is slot-indexed like
    // trunk: a rejected d1's slot is overwritten by later appends).
    mtpDraft2R.h = newList();
    int nnH2 = H;
    setarg(kh[EMBED], 0, sizeof(void *), &dDraftE);
    setarg(kh[EMBED], 1, sizeof(void *), &embP);
    setarg(kh[EMBED], 2, sizeof(void *), &dCtrlDraft2);
    launch(mtpDraft2R, kh[EMBED], H);
    rnorm(mtpDraft2R, dDraftEn, dDraftE, dPreE);
    rnorm(mtpDraft2R, dDraftHn, dChH, dPreH);
    setarg(kh[CONCAT], 0, sizeof(void *), &dDraftIn);
    setarg(kh[CONCAT], 1, sizeof(void *), &dDraftEn);
    setarg(kh[CONCAT], 2, sizeof(void *), &dDraftHn);
    setarg(kh[CONCAT], 3, sizeof(int), &nnH2);
    launch(mtpDraft2R, kh[CONCAT], 2 * H);
    xq(mtpDraft2R, dDraftIn, 10240);
    gemvE(mtpDraft2R, find("mtp.fc.weight"), H, 10240, dDraftX);
    rnorm(mtpDraft2R, dH, dDraftX, dMInN);
    xq(mtpDraft2R, dH, H);
    gemvE(mtpDraft2R, find("mtp.layers.0.self_attn.q_proj.weight"), QW, H, dQ16);
    gemvE(mtpDraft2R, find("mtp.layers.0.self_attn.k_proj.weight"), KVW, H, dK16);
    gemvE(mtpDraft2R, find("mtp.layers.0.self_attn.v_proj.weight"), KVW, H, dV16);
    setarg(kh[SPLIT], 0, sizeof(void *), &dQ16);
    setarg(kh[SPLIT], 1, sizeof(void *), &dQn);
    setarg(kh[SPLIT], 2, sizeof(void *), &dGate);
    launch(mtpDraft2R, kh[SPLIT], QN);
    setarg(kh[BNORM], 0, sizeof(void *), &dQn);
    setarg(kh[BNORM], 1, sizeof(void *), &dQn);
    setarg(kh[BNORM], 2, sizeof(void *), &dMQNW);
    setarg(kh[BNORM], 3, sizeof(int), &n256);
    launch(mtpDraft2R, kh[BNORM], 24);
    setarg(kh[BNORM], 0, sizeof(void *), &dKn);
    setarg(kh[BNORM], 1, sizeof(void *), &dK16);
    setarg(kh[BNORM], 2, sizeof(void *), &dMKNW);
    setarg(kh[BNORM], 3, sizeof(int), &n256);
    launch(mtpDraft2R, kh[BNORM], 4);
    setarg(kh[ROPE], 0, sizeof(void *), &dQn);
    setarg(kh[ROPE], 1, sizeof(void *), &dKn);
    setarg(kh[ROPE], 2, sizeof(void *), &dCos);
    setarg(kh[ROPE], 3, sizeof(void *), &dSin);
    setarg(kh[ROPE], 4, sizeof(void *), &dCtrlDraft2);
    launch(mtpDraft2R, kh[ROPE], 28);
    setarg(kh[KV], 0, sizeof(void *), &dMKc);
    setarg(kh[KV], 1, sizeof(void *), &dMVc);
    setarg(kh[KV], 2, sizeof(void *), &dKn);
    setarg(kh[KV], 3, sizeof(void *), &dV16);
    setarg(kh[KV], 4, sizeof(void *), &dCtrlDraft2);
    setarg(kh[KV], 5, sizeof(int), &tmax);
    launch(mtpDraft2R, kh[KV], KVW);
    setarg(kh[ATTN], 0, sizeof(void *), &dAtt);
    setarg(kh[ATTN], 1, sizeof(void *), &dQn);
    setarg(kh[ATTN], 2, sizeof(void *), &dMKc);
    setarg(kh[ATTN], 3, sizeof(void *), &dMVc);
    setarg(kh[ATTN], 4, sizeof(void *), &dGate);
    setarg(kh[ATTN], 5, sizeof(void *), &dCtrlDraft2);
    setarg(kh[ATTN], 6, sizeof(int), &tmax);
    setarg(kh[ATTN], 7, sizeof(void *), &dWts);
    launch(mtpDraft2R, kh[ATTN], 24);
    xq(mtpDraft2R, dAtt, QN);
    gemvE(mtpDraft2R, find("mtp.layers.0.self_attn.o_proj.weight"), H, QN, dMix);
    res(mtpDraft2R, dTmp, dDraftX, dMix);
    rnorm(mtpDraft2R, dH, dTmp, dMPostN);
    xq(mtpDraft2R, dH, H);
    gemvE(mtpDraft2R, find("mtp.layers.0.mlp.gate_proj.weight"), I, H, dG17);
    gemvE(mtpDraft2R, find("mtp.layers.0.mlp.up_proj.weight"), I, H, dU17);
    setarg(kh[SILU], 0, sizeof(void *), &dG17);
    setarg(kh[SILU], 1, sizeof(void *), &dU17);
    setarg(kh[SILU], 2, sizeof(void *), &dG17);
    launch(mtpDraft2R, kh[SILU], I);
    xq(mtpDraft2R, dG17, I);
    gemvE(mtpDraft2R, find("mtp.layers.0.mlp.down_proj.weight"), H, I, dMix);
    res(mtpDraft2R, dDraftX, dTmp, dMix);
    rnorm(mtpDraft2R, dH, dDraftX, dMNorm);
    xq(mtpDraft2R, dH, H);
    gemvE(mtpDraft2R, find("lm_head.weight"), V, H, dDraftLogits);
    setarg(kh[ARG1], 0, sizeof(void *), &dDraftLogits);
    setarg(kh[ARG1], 1, sizeof(void *), &dPV);
    setarg(kh[ARG1], 2, sizeof(void *), &dPI);
    setarg(kh[ARG1], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG1], 4, (size_t)256 * 4, nullptr);
    launch(mtpDraft2R, kh[ARG1], 64);
    setarg(kh[ARG2], 0, sizeof(void *), &dPV);
    setarg(kh[ARG2], 1, sizeof(void *), &dPI);
    setarg(kh[ARG2], 2, sizeof(void *), &dDraftOutT);
    setarg(kh[ARG2], 3, (size_t)256 * 4, nullptr);
    setarg(kh[ARG2], 4, (size_t)256 * 4, nullptr);
    launch(mtpDraft2R, kh[ARG2], 1);
    CHECK(zeCommandListClose(mtpDraft2R.h));
    std::fprintf(stderr, "[l0] recorded 64 triple-verification layer lists + embedM3 + tailM3 + commitSpec2 + draft2\n");
  }

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
  // ---- T7.4 in-process chunked prefill (single-binary production path) ----
  // Flagged by --prefill-chunks N (NCH chunks of M rows, M via CHUNK_M env,
  // default 256; inputs via CHUNK64MC_HOST_IN prefix, same convention as
  // chunk64mc_replay). Records per-chunk 64-layer lists against the SHARED
  // payArena/scArena (weights loaded once — kills the 11 h re-upload
  // traffic AND the >24 GB duplicate-arena trap), appends KV/SSM straight
  // into decode-layout dKc/dVc/dConv/dS, then hands dX + impP to the loop
  // below (same seam as --import-caches, no file round-trip).
  // Score buffer aliased S==Sm (row-wise read-before-write, bitwise-safe);
  // saves 1.5 GiB at 64K (budget 21.4 vs 22.71 GiB heap).
  double prefill_ms = 0;
  if (preChunks > 0) {
    int M = 256;
    if (const char *me = std::getenv("CHUNK_M")) {
      int v = std::atoi(me);
      if (v >= 8 && v <= 512)
        M = v;
    }
    const char *pfx = std::getenv("CHUNK64MC_HOST_IN");
    if (!pfx) {
      std::fprintf(stderr, "--prefill-chunks needs CHUNK64MC_HOST_IN=<prefix>\n");
      return 2;
    }
    const int NCH = preChunks, TC = NCH * M, PNQ = 24;
    const size_t MH = (size_t)M * H;
    if (P < TC) {
      std::fprintf(stderr, "--prefill-chunks: prompt P=%d < TC=%d\n", P, TC);
      return 2;
    }
    // Chunk scratch (decode-layout state arenas are shared, not duplicated).
    void *dXa_ = alloc(MH * 4), *dXb_ = alloc(MH * 4);
    void *dH_ = alloc(MH * 4), *dTmp_ = alloc(MH * 4), *dMix_ = alloc(MH * 4),
         *dHh_ = alloc(MH * 2);
    const size_t MC = (size_t)M * C, MV = (size_t)M * V6, MI = (size_t)M * I,
                 MH48 = (size_t)M * NH, MVQ = (size_t)M * QN;
    void *dQKV_ = alloc((size_t)M * QW * 4);
    void *dMxC_ = alloc(MC * 4), *dMxR_ = alloc(MV * 4);
    void *dZ_ = alloc(MV * 4), *dQ48_ = alloc(MV * 4), *dK48_ = alloc(MV * 4),
         *dV48_ = alloc(MV * 4);
    void *dAttL_ = alloc(MV * 4), *dAtthL_ = alloc(MV * 2);
    void *dB_ = alloc(MH48 * 4), *dA_ = alloc(MH48 * 4),
         *dBt_ = alloc(MH48 * 4), *dG48_ = alloc(MH48 * 4);
    void *dK16_ = alloc((size_t)M * KVW * 4),
         *dV16_ = alloc((size_t)M * KVW * 4),
         *dKn_ = alloc((size_t)M * KVW * 4);
    void *dQn_ = alloc(MVQ * 4), *dGate_ = alloc(MVQ * 4),
         *dAttF_ = alloc(MVQ * 4), *dCore_ = alloc(MVQ * 4),
         *dQnh_ = alloc(MVQ * 2), *dAtthF_ = alloc(MVQ * 2);
    void *dWts_ = alloc((size_t)M * PNQ * TC * 4),
         *dWsmh_ = alloc((size_t)M * PNQ * TC * 2);
    void *dG17_ = alloc(MI * 4), *dU17_ = alloc(MI * 4),
         *dG17h_ = alloc(MI * 2);
    void *dCtrlP = alloc(sizeof(DecodeControl)),
         *dCtrlSmP = alloc(sizeof(DecodeControl));
    auto wsp = [&](int L, const char *s, void **pp, void **ps) {
      const Entry *e = find(nm(L, s));
      if (!e) {
        std::fprintf(stderr, "prefill: missing %s\n", s);
        std::exit(1);
      }
      *pp = (char *)payArena + (e->d_off - pay_lo);
      *ps = (char *)scArena + (e->sc_off - sc_lo);
    };
    auto upChunkCtrl = [&](int ch) {
      int base = ch * M, W = (ch + 1) * M;
      DecodeControl c{9000 + ch, base, base + 1, -1}, csm{0, 0, W, 0};
      CHECK(zeCommandListAppendMemoryCopy(up, dCtrlP, &c, sizeof(c), nullptr,
                                          0, nullptr));
      CHECK(zeCommandListAppendMemoryCopy(up, dCtrlSmP, &csm, sizeof(csm),
                                          nullptr, 0, nullptr));
    };
    std::vector<float> hXc(MH);
    int pfTmax = MAXCTX, pfN256 = 256, pfMmA = M, pfRowsSM = M * PNQ;
    for (int ch = 0; ch < NCH; ++ch) {
      int base = ch * M, W = (ch + 1) * M, nBcQK = (W + 15) / 16;
      char pp[256];
      std::snprintf(pp, sizeof pp, "%s_%d.bin", pfx, ch);
      FILE *fi = std::fopen(pp, "rb");
      if (!fi && ch < 2) {
        // Legacy a/b naming (prefill_arb.py embed output).
        std::snprintf(pp, sizeof pp, "%s_%c.bin", pfx, ch == 0 ? 'a' : 'b');
        fi = std::fopen(pp, "rb");
      }
      if (!fi) {
        std::fprintf(stderr, "prefill: no input %s\n", pp);
        return 2;
      }
      size_t nr = std::fread(hXc.data(), 4, MH, fi);
      std::fclose(fi);
      if (nr != (size_t)MH) {
        std::fprintf(stderr, "prefill: short input %s\n", pp);
        return 2;
      }
      upChunkCtrl(ch);
      for (int L = 0; L < 64; ++L) {
        bool full = (L % 4 == 3);
        int slot = L / 4, sl = L - (L + 1) / 4;
        double tRec = prof ? now_ns() : 0;
        Rec R{newList()};
        void *dXi = (L % 2 == 0) ? dXa_ : dXb_;
        void *dXo = (L % 2 == 0) ? dXb_ : dXa_;
        if (L == 0) {
          CHECK(zeCommandListAppendMemoryCopy(up, dXi, hXc.data(), MH * 4,
                                              nullptr, 0, nullptr));
        }
        auto normYa = [&](void *Y, void *X, void *Wn) {
          int nn = H;
          for (int m = 0; m < M; ++m) {
            void *yy = (char *)Y + (size_t)m * H * 4;
            void *xx = (char *)X + (size_t)m * H * 4;
            setarg(kh[NORM], 0, sizeof(void *), &yy);
            setarg(kh[NORM], 1, sizeof(void *), &xx);
            setarg(kh[NORM], 2, sizeof(void *), &Wn);
            setarg(kh[NORM], 3, sizeof(int), &nn);
            setarg(kh[NORM], 4, (size_t)256 * 8, nullptr);
            launch(R, kh[NORM], 1);
          }
        };
        auto cvtYa = [&](void *Oh, void *X, int nn) {
          setarg(kh[CVT2], 0, sizeof(void *), &Oh);
          setarg(kh[CVT2], 1, sizeof(void *), &X);
          launch(R, kh[CVT2], nn);
        };
        // AINFER_GEMM_DB=1 selects the double-buffered slice-pair GEMM
        // (ChunkGemmDB: 8 DPAS per barrier vs 4). Default keeps ChunkGemm.
        const bool gemmDB =
            (std::getenv("AINFER_GEMM_DB") != nullptr &&
             std::getenv("AINFER_GEMM_DB")[0] == '1');
        auto cgemmW = [&](void *Wp, void *Ws, int nn, int kk, void *Ah,
                          void *Y) {
          int mm = M;
          if (gemmDB) {
            setarg(kh[CGEMMDB], 0, sizeof(void *), &Ah);
            setarg(kh[CGEMMDB], 1, sizeof(void *), &Wp);
            setarg(kh[CGEMMDB], 2, sizeof(void *), &Ws);
            setarg(kh[CGEMMDB], 3, sizeof(void *), &Y);
            setarg(kh[CGEMMDB], 4, sizeof(int), &mm);
            setarg(kh[CGEMMDB], 5, sizeof(int), &kk);
            setarg(kh[CGEMMDB], 6, sizeof(int), &nn);
            setarg(kh[CGEMMDB], 7, (size_t)512 * 2, nullptr);
            setarg(kh[CGEMMDB], 8, (size_t)256 * 2, nullptr);
            setarg(kh[CGEMMDB], 9, (size_t)512 * 2, nullptr);
            setarg(kh[CGEMMDB], 10, (size_t)256 * 2, nullptr);
            setarg(kh[CGEMMDB], 11, (size_t)512 * 4, nullptr);
            launch(R, kh[CGEMMDB], ((mm + 31) / 32) * (nn / 16));
            return;
          }
          setarg(kh[CGEMM], 0, sizeof(void *), &Ah);
          setarg(kh[CGEMM], 1, sizeof(void *), &Wp);
          setarg(kh[CGEMM], 2, sizeof(void *), &Ws);
          setarg(kh[CGEMM], 3, sizeof(void *), &Y);
          setarg(kh[CGEMM], 4, sizeof(int), &mm);
          setarg(kh[CGEMM], 5, sizeof(int), &kk);
          setarg(kh[CGEMM], 6, sizeof(int), &nn);
          setarg(kh[CGEMM], 7, (size_t)512 * 2, nullptr);
          setarg(kh[CGEMM], 8, (size_t)256 * 2, nullptr);
          setarg(kh[CGEMM], 9, (size_t)512 * 4, nullptr);
          launch(R, kh[CGEMM], ((mm + 31) / 32) * (nn / 16));
        };
        auto resYa = [&](void *Y, void *A, void *B) {
          for (int m = 0; m < M; ++m) {
            void *yy = (char *)Y + (size_t)m * H * 4;
            void *aa = (char *)A + (size_t)m * H * 4;
            void *bb = (char *)B + (size_t)m * H * 4;
            setarg(kh[RES], 0, sizeof(void *), &yy);
            setarg(kh[RES], 1, sizeof(void *), &aa);
            setarg(kh[RES], 2, sizeof(void *), &bb);
            launch(R, kh[RES], H);
          }
        };
        auto mlpTail = [&](void *dXo_, void *dTmp_) {
          void *pWg, *pWgS, *pWu, *pWuS, *pWd, *pWdS;
          wsp(L, "mlp.gate_proj.weight", &pWg, &pWgS);
          wsp(L, "mlp.up_proj.weight", &pWu, &pWuS);
          wsp(L, "mlp.down_proj.weight", &pWd, &pWdS);
          normYa(dH_, dTmp_,
                 (char *)dPostN + (size_t)L * H * 4);
          cvtYa(dHh_, dH_, M * H);
          cgemmW(pWg, pWgS, I, H, dHh_, dG17_);
          cgemmW(pWu, pWuS, I, H, dHh_, dU17_);
          setarg(kh[SILU], 0, sizeof(void *), &dG17_);
          setarg(kh[SILU], 1, sizeof(void *), &dU17_);
          setarg(kh[SILU], 2, sizeof(void *), &dG17_);
          launch(R, kh[SILU], MI);
          cvtYa(dG17h_, dG17_, MI);
          cgemmW(pWd, pWdS, H, I, dG17h_, dMix_);
          resYa(dXo_, dTmp_, dMix_);
        };
        if (!full) {
          void *pWq, *pWqS, *pWz, *pWzS, *pWb, *pWbS, *pWa, *pWaS, *pWo, *pWoS;
          wsp(L, "linear_attn.in_proj_qkv.weight", &pWq, &pWqS);
          wsp(L, "linear_attn.in_proj_z.weight", &pWz, &pWzS);
          wsp(L, "linear_attn.in_proj_b.weight", &pWb, &pWbS);
          wsp(L, "linear_attn.in_proj_a.weight", &pWa, &pWaS);
          wsp(L, "linear_attn.out_proj.weight", &pWo, &pWoS);
          normYa(dH_, dXi, (char *)dInN + (size_t)L * H * 4);
          cvtYa(dHh_, dH_, M * H);
          cgemmW(pWq, pWqS, C, H, dHh_, dQKV_);
          cgemmW(pWz, pWzS, V6, H, dHh_, dZ_);
          cgemmW(pWb, pWbS, NH, H, dHh_, dB_);
          cgemmW(pWa, pWaS, NH, H, dHh_, dA_);
          void *csS = (char *)dConv + (size_t)sl * C * 3 * 4;
          void *cwS = (char *)dConvW + (size_t)sl * C * 4 * 4;
          setarg(kh[CCONV], 0, sizeof(void *), &dMxC_);
          setarg(kh[CCONV], 1, sizeof(void *), &dQKV_);
          setarg(kh[CCONV], 2, sizeof(void *), &csS);
          setarg(kh[CCONV], 3, sizeof(void *), &cwS);
          setarg(kh[CCONV], 4, sizeof(int), &C);
          setarg(kh[CCONV], 5, sizeof(int), &pfMmA);
          launch(R, kh[CCONV], C);
          for (int m = 0; m < M; ++m) {
            void *mx = (char *)dMxC_ + (size_t)m * C * 4;
            void *q4 = (char *)dQ48_ + (size_t)m * V6 * 4;
            void *k4 = (char *)dK48_ + (size_t)m * V6 * 4;
            void *v4 = (char *)dV48_ + (size_t)m * V6 * 4;
            setarg(kh[SPLIT2], 0, sizeof(void *), &mx);
            setarg(kh[SPLIT2], 1, sizeof(void *), &q4);
            setarg(kh[SPLIT2], 2, sizeof(void *), &k4);
            setarg(kh[SPLIT2], 3, sizeof(void *), &v4);
            launch(R, kh[SPLIT2], V6);
            setarg(kh[L2], 0, sizeof(void *), &q4);
            setarg(kh[L2], 1, sizeof(void *), &k4);
            launch(R, kh[L2], 96);
            void *b1 = (char *)dB_ + (size_t)m * NH * 4;
            void *a1 = (char *)dA_ + (size_t)m * NH * 4;
            void *bt1 = (char *)dBt_ + (size_t)m * NH * 4;
            void *g1 = (char *)dG48_ + (size_t)m * NH * 4;
            void *alS = (char *)dAL + (size_t)sl * 48 * 4;
            void *dtS = (char *)dDT + (size_t)sl * 48 * 4;
            setarg(kh[BETA], 0, sizeof(void *), &bt1);
            setarg(kh[BETA], 1, sizeof(void *), &g1);
            setarg(kh[BETA], 2, sizeof(void *), &b1);
            setarg(kh[BETA], 3, sizeof(void *), &a1);
            setarg(kh[BETA], 4, sizeof(void *), &alS);
            setarg(kh[BETA], 5, sizeof(void *), &dtS);
            launch(R, kh[BETA], NH);
          }
          void *sS = (char *)dS + (size_t)sl * NH * D * D * 4;
          setarg(kh[CRECUR], 0, sizeof(void *), &dMxR_);
          setarg(kh[CRECUR], 1, sizeof(void *), &dQ48_);
          setarg(kh[CRECUR], 2, sizeof(void *), &dK48_);
          setarg(kh[CRECUR], 3, sizeof(void *), &dV48_);
          setarg(kh[CRECUR], 4, sizeof(void *), &sS);
          setarg(kh[CRECUR], 5, sizeof(void *), &dBt_);
          setarg(kh[CRECUR], 6, sizeof(void *), &dG48_);
          setarg(kh[CRECUR], 7, sizeof(int), &pfMmA);
          launch(R, kh[CRECUR], NH);
          for (int m = 0; m < M; ++m) {
            void *mx = (char *)dMxR_ + (size_t)m * V6 * 4;
            void *bt1 = (char *)dBt_ + (size_t)m * NH * 4;
            void *at = (char *)dAttL_ + (size_t)m * V6 * 4;
            void *zz = (char *)dZ_ + (size_t)m * V6 * 4;
            setarg(kh[RMSI], 0, sizeof(void *), &bt1);
            setarg(kh[RMSI], 1, sizeof(void *), &mx);
            launch(R, kh[RMSI], NH);
            setarg(kh[GATE], 0, sizeof(void *), &at);
            setarg(kh[GATE], 1, sizeof(void *), &mx);
            setarg(kh[GATE], 2, sizeof(void *), &zz);
            void *ngS = (char *)dLinN + (size_t)sl * 128 * 4;
            setarg(kh[GATE], 3, sizeof(void *), &ngS);
            setarg(kh[GATE], 4, sizeof(void *), &bt1);
            launch(R, kh[GATE], V6);
          }
          cvtYa(dAtthL_, dAttL_, M * V6);
          cgemmW(pWo, pWoS, H, V6, dAtthL_, dMix_);
          resYa(dTmp_, dXi, dMix_);
          mlpTail(dXo, dTmp_);
        } else {
          void *pWq, *pWqS, *pWk, *pWkS, *pWv, *pWvS, *pWo, *pWoS;
          wsp(L, "self_attn.q_proj.weight", &pWq, &pWqS);
          wsp(L, "self_attn.k_proj.weight", &pWk, &pWkS);
          wsp(L, "self_attn.v_proj.weight", &pWv, &pWvS);
          wsp(L, "self_attn.o_proj.weight", &pWo, &pWoS);
          normYa(dH_, dXi, (char *)dInN + (size_t)L * H * 4);
          cvtYa(dHh_, dH_, M * H);
          cgemmW(pWq, pWqS, QW, H, dHh_, dQKV_);
          cgemmW(pWk, pWkS, KVW, H, dHh_, dK16_);
          cgemmW(pWv, pWvS, KVW, H, dHh_, dV16_);
          for (int m = 0; m < M; ++m) {
            void *q16 = (char *)dQKV_ + (size_t)m * QW * 4;
            void *k16 = (char *)dK16_ + (size_t)m * KVW * 4;
            void *qn = (char *)dQn_ + (size_t)m * QN * 4;
            void *gt = (char *)dGate_ + (size_t)m * QN * 4;
            void *kn = (char *)dKn_ + (size_t)m * KVW * 4;
            setarg(kh[SPLIT], 0, sizeof(void *), &q16);
            setarg(kh[SPLIT], 1, sizeof(void *), &qn);
            setarg(kh[SPLIT], 2, sizeof(void *), &gt);
            launch(R, kh[SPLIT], QN);
            setarg(kh[BNORM], 0, sizeof(void *), &qn);
            setarg(kh[BNORM], 1, sizeof(void *), &qn);
            void *qnW = (char *)dQNW + (size_t)slot * 256 * 4;
            setarg(kh[BNORM], 2, sizeof(void *), &qnW);
            setarg(kh[BNORM], 3, sizeof(int), &pfN256);
            launch(R, kh[BNORM], 24);
            setarg(kh[BNORM], 0, sizeof(void *), &kn);
            setarg(kh[BNORM], 1, sizeof(void *), &k16);
            void *knW = (char *)dKNW + (size_t)slot * 256 * 4;
            setarg(kh[BNORM], 2, sizeof(void *), &knW);
            setarg(kh[BNORM], 3, sizeof(int), &pfN256);
            launch(R, kh[BNORM], 4);
          }
          void *kcS = (char *)dKc + ((size_t)slot * MAXCTX) * 4 * 256 * 2;
          void *vcS = (char *)dVc + ((size_t)slot * MAXCTX) * 4 * 256 * 2;
          setarg(kh[CROPE], 0, sizeof(void *), &dQn_);
          setarg(kh[CROPE], 1, sizeof(void *), &dKn_);
          setarg(kh[CROPE], 2, sizeof(void *), &dCos);
          setarg(kh[CROPE], 3, sizeof(void *), &dSin);
          setarg(kh[CROPE], 4, sizeof(void *), &dCtrlP);
          setarg(kh[CROPE], 5, sizeof(int), &pfTmax);
          setarg(kh[CROPE], 6, sizeof(int), &pfMmA);
          launch(R, kh[CROPE], M * 28);
          setarg(kh[CKV], 0, sizeof(void *), &kcS);
          setarg(kh[CKV], 1, sizeof(void *), &vcS);
          setarg(kh[CKV], 2, sizeof(void *), &dKn_);
          setarg(kh[CKV], 3, sizeof(void *), &dV16_);
          setarg(kh[CKV], 4, sizeof(void *), &dCtrlP);
          setarg(kh[CKV], 5, sizeof(int), &pfTmax);
          setarg(kh[CKV], 6, sizeof(int), &pfMmA);
          launch(R, kh[CKV], M * 1024);
          cvtYa(dQnh_, dQn_, M * QN);
          {
            int pp = base, ww = W, ss = TC;
            setarg(kh[CQK], 0, sizeof(void *), &dQnh_);
            setarg(kh[CQK], 1, sizeof(void *), &kcS);
            setarg(kh[CQK], 2, sizeof(void *), &dWts_);
            setarg(kh[CQK], 3, sizeof(int), &pp);
            setarg(kh[CQK], 4, sizeof(int), &pfMmA);
            setarg(kh[CQK], 5, sizeof(int), &ww);
            setarg(kh[CQK], 6, sizeof(int), &ss);
            setarg(kh[CQK], 7, (size_t)8 * 256 * 2, nullptr);
            setarg(kh[CQK], 8, (size_t)16 * 16 * 2, nullptr);
            setarg(kh[CQK], 9, (size_t)8 * 16 * 4, nullptr);
            launch(R, kh[CQK], (uint32_t)(M * 4 * nBcQK));
          }
          {
            int rows = M * PNQ;
            setarg(kh[CSM], 0, sizeof(void *), &dWts_);
            setarg(kh[CSM], 1, sizeof(void *), &dWts_);
            setarg(kh[CSM], 2, sizeof(void *), &dCtrlSmP);
            setarg(kh[CSM], 3, sizeof(int), &TC);
            setarg(kh[CSM], 4, sizeof(int), &rows);
            launch(R, kh[CSM], (uint32_t)rows);
          }
          cvtYa(dWsmh_, dWts_, M * PNQ * TC);
          {
            int kk = W; // K = valid cols; KMAX = row stride TC
            setarg(kh[CWV], 0, sizeof(void *), &dWsmh_);
            setarg(kh[CWV], 1, sizeof(void *), &vcS);
            setarg(kh[CWV], 2, sizeof(void *), &dCore_);
            setarg(kh[CWV], 3, sizeof(int), &kk);
            setarg(kh[CWV], 4, sizeof(int), &TC);
            setarg(kh[CWV], 5, (size_t)8 * 16 * 2, nullptr);
            setarg(kh[CWV], 6, (size_t)16 * 16 * 2, nullptr);
            setarg(kh[CWV], 7, (size_t)8 * 16 * 4, nullptr);
            launch(R, kh[CWV], (uint32_t)(M * 64));
          }
          setarg(kh[GMUL], 0, sizeof(void *), &dAttF_);
          setarg(kh[GMUL], 1, sizeof(void *), &dCore_);
          setarg(kh[GMUL], 2, sizeof(void *), &dGate_);
          launch(R, kh[GMUL], (uint32_t)(M * QN));
          cvtYa(dAtthF_, dAttF_, M * QN);
          cgemmW(pWo, pWoS, H, QN, dAtthF_, dMix_);
          resYa(dTmp_, dXi, dMix_);
          mlpTail(dXo, dTmp_);
        }
        CHECK(zeCommandListClose(R.h));
        double t0 = now_ns();
        CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &R.h, fence));
        CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
        CHECK(zeFenceReset(fence));
        double dtms = (now_ns() - t0) / 1e6;
        prefill_ms += dtms;
        if (prof)
          buckets["prefill_rec"].push_back(t0 - tRec);
        if (prof)
          buckets[full ? "prefill_full" : "prefill_lin"].push_back(dtms *
                                                                   1e6);
        if ((L & 15) == 0 || L == 63)
          std::fprintf(stderr, "[prefill] chunk %d layer %d recorded+run (%s)\n",
                       ch, L, full ? "full" : "linear");
        zeCommandListDestroy(R.h);
      }
    }
    // Hand final hidden to the decode loop (same seam as --import-caches).
    CHECK(zeCommandListAppendMemoryCopy(up, dX, (char *)dXa_ + (size_t)(M - 1) * H * 4,
                                        (size_t)H * 4, nullptr, 0, nullptr));
    if (const char *pd = std::getenv("AINFER_PREFILL_DUMP")) {
      // Bitwise gate vs chunk64mc out_b (same inputs => identical hidden).
      std::vector<float> hpx(MH);
      CHECK(zeCommandListAppendMemoryCopy(up, hpx.data(), dXa_, MH * 4,
                                          nullptr, 0, nullptr));
      FILE *pf = std::fopen(pd, "wb");
      if (pf) {
        std::fwrite(hpx.data(), 4, MH, pf);
        std::fclose(pf);
      }
    }
    impP = TC;
    std::fprintf(stderr, "[prefill] %d chunks x %d rows, %.1f ms exec, impP=%d\n",
                 NCH, M, prefill_ms, impP);
  }
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
  // T7.4 import mode: slots 0..impP-1 prefilled externally; skip them and
  // run tail-only at step impP-1 (re-executing layers would duplicate the
  // slot and clobber the injected hidden). ids[impP..P-1] loop-decode as
  // the question when present; tail runs at P-1 as usual.
  const bool impTailOnly = impP > 0;
  int pendingDraft = -1;
  int mtpDraftTotal = 0, mtpDraftAccepted = 0;
  int mtpChainedTotal = 0, mtpChainedAccepted = 0;
  double tGenStart = 0;
  double total_gen_ms = 0;
  double gen_tok_per_sec = 0;
  // Unbuffered per-step trace for long runs (buffered stdout hides pace for
  // tens of minutes at 64K; lesson from the first needle attempt).
  const bool steplog = std::getenv("AINFER_STEPLOG") != nullptr;
  double tRun0 = steplog ? now_ns() : 0;
  (void)tRun0;
  for (int step = impTailOnly ? impP - 1 : 0; step < P + G; ++step) {
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
    if (steplog)
      std::fprintf(stderr, "[step %d pos %d %.1fs]\n", step, pos,
                   (now_ns() - tRun0) / 1e9);
    DecodeControl c{ids[step], pos, pos + 1, -1};
    double tCtl = prof ? now_ns() : 0;
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                        nullptr));
    seg("control", tCtl);
    if (!(impTailOnly && step == impP - 1)) {
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
      if (step >= P - 1 && tGenStart == 0)
        tGenStart = now_ns();
      if (opt_mtp && step >= P - 1) {
        if (tok == 248046 || tok == 248044 || (int)generated.size() >= G) {
          stopped_eos = (tok == 248046 || tok == 248044);
          break;
        }
        // Depth-2 chained drafts: d1 = MTP(confirmed), d2 = MTP(d1, h_mtp1).
        // Helper: (re)draft both levels from a confirmed token + trunk hidden
        // (trunk dX must hold hidden-after-confirmed; dChH saves draft-1
        // hidden for the chained second evaluation).
        int pendingDraft = -1, pendingDraft2 = -1;
        bool need_d2 = false;
        auto draft1only = [&](int conf_tok, int conf_pos) {
          DecodeControl cd{conf_tok, conf_pos, conf_pos + 1, -1};
          CHECK(zeCommandListAppendMemoryCopy(up, dCtrlDraft, &cd, sizeof(cd),
                                              nullptr, 0, nullptr));
          exec(mtpDraftR.h, "mtp_draft");
          CHECK(zeCommandListAppendMemoryCopy(up, &pendingDraft, dDraftOutT, 4,
                                              nullptr, 0, nullptr));
        };
        auto draft2chain = [&](int d1pos) {
          // Chained part only: d1 pending for position d1pos (= conf_pos + 1
          // of the draft-1 that produced it), dDraftX holds its hidden.
          CHECK(zeCommandListAppendMemoryCopy(up, dChH, dDraftX, (size_t)H * 4,
                                              nullptr, 0, nullptr));
          DecodeControl cd2{pendingDraft, d1pos, d1pos + 1, -1};
          CHECK(zeCommandListAppendMemoryCopy(up, dCtrlDraft2, &cd2, sizeof(cd2),
                                              nullptr, 0, nullptr));
          exec(mtpDraft2R.h, "mtp_draft2");
          CHECK(zeCommandListAppendMemoryCopy(up, &pendingDraft2, dDraftOutT, 4,
                                              nullptr, 0, nullptr));
          need_d2 = false;
        };
        auto redraft = [&](int conf_tok, int conf_pos) {
          draft1only(conf_tok, conf_pos);
          draft2chain(conf_pos + 1);
        };
        // Depth policy: AINFER_MTP2=0 forces legacy depth-1 dual-token
        // loop, =1 forces depth-2 triple loop; default (unset) is ADAPTIVE.
        const char *mtp2e = std::getenv("AINFER_MTP2");
        const int force_depth = (mtp2e && mtp2e[0] == '0') ? 2 : (mtp2e ? 3 : 0);
        double trail_tok = 0, d1trail = 0;
        int trail_n = 0, d1n = 0, since_probe = 0, probe_ivl = 8;
        bool last_probe = false;
        auto note_d1 = [&](bool acc) {
          d1trail += acc ? 1.0 : 0.0;
          d1n++;
          if (d1n > 8) {
            d1trail *= 0.5;
            d1n = 4;
          }
        };
        auto want_m3 = [&]() -> bool {
          if (force_depth == 2)
            return false;
          if (force_depth == 3)
            return true;
          // Discovery is free: d1trail updates on EVERY round (M2 and M3),
          // so seed it with cheap M2 rounds — no M3 warmup tax.
          last_probe = false;
          if (d1n < 4) {
            if ((++since_probe % 4) == 0) {
              last_probe = true;
              return true;
            }
            return false;
          }
          double d1a = d1trail / d1n;
          if (d1a < 0.8) {
            if ((++since_probe % probe_ivl) == 0) {
              last_probe = true;
              return true;
            }
            return false;
          }
          if (trail_n >= 2) {
            double rate_m3 = (trail_tok / trail_n) / 0.1168;
            if (rate_m3 < 18.5) {
              if ((++since_probe % probe_ivl) == 0) {
                last_probe = true;
                return true;
              }
              return false;
            }
          }
          return true;
        };
        auto note_trail = [&](int sz0, bool chained_acc) {
          trail_tok += (double)((int)generated.size() - sz0);
          trail_n++;
          if (trail_n > 12) {
            trail_tok *= 0.5;
            trail_n = 6;
          }
          if (last_probe) {
            probe_ivl = chained_acc ? 8 : std::min(32, probe_ivl * 2);
            last_probe = false;
          }
        };
        int confirmed_tok = tok;
        int cur_pos = pos + 1;
        if (force_depth == 2) {
          // Legacy depth-1 loop (M2 lists, single draft per round). NOTE:
          // no local pendingDraft here — draft1only writes the shared outer
          // one (a shadowed local once froze drafts at their first value).
          draft1only(tok, pos);
          while ((int)generated.size() <= G && cur_pos < MAXCTX - 1) {
            int draft_tok = pendingDraft;
            int pos0 = cur_pos;
            int pos1 = cur_pos + 1;
            DecodeControl c0{confirmed_tok, pos0, pos0 + 1, -1};
            DecodeControl c1{draft_tok, pos1, pos1 + 1, -1};
            CHECK(zeCommandListAppendMemoryCopy(up, dCtrl0, &c0, sizeof(c0), nullptr, 0, nullptr));
            CHECK(zeCommandListAppendMemoryCopy(up, dCtrl1, &c1, sizeof(c1), nullptr, 0, nullptr));
            exec(embM2.h, "embed_m2");
            for (int L = 0; L < 64; ++L) {
              exec(layersM2[L].h, is_full(L) ? "attn_m2" : "lin_m2");
            }
            exec(tailM2.h, "tail_m2");
            int tok0 = -1, tok1 = -1;
            CHECK(zeCommandListAppendMemoryCopy(up, &tok0, dOutT0, 4, nullptr, 0, nullptr));
            CHECK(zeCommandListAppendMemoryCopy(up, &tok1, dOutT1, 4, nullptr, 0, nullptr));
            mtpDraftTotal++;
            bool acc2 = (tok0 == draft_tok);
            note_d1(acc2);
            if (acc2) {
              mtpDraftAccepted++;
              exec(commitSpecR.h, "commit_spec");
              generated.push_back(tok0);
              ids.push_back(tok0);
              std::printf("step %d pos %d -> token %d [MTP draft accepted, alpha=%.3f (%d/%d)]\n",
                          (int)ids.size() - 1, pos0, tok0,
                          (float)mtpDraftAccepted / mtpDraftTotal,
                          mtpDraftAccepted, mtpDraftTotal);
              if (tok0 == 248046 || tok0 == 248044) {
                stopped_eos = true;
                break;
              }
              if ((int)generated.size() <= G) {
                generated.push_back(tok1);
                ids.push_back(tok1);
                std::printf("step %d pos %d -> token %d [MTP verified]\n",
                            (int)ids.size() - 1, pos1, tok1);
                if (tok1 == 248046 || tok1 == 248044) {
                  stopped_eos = true;
                  break;
                }
                if ((int)generated.size() <= G && pos1 + 1 < MAXCTX) {
                  CHECK(zeCommandListAppendMemoryCopy(up, dX, dX1, (size_t)H * 4, nullptr, 0, nullptr));
                  draft1only(tok1, pos1);
                  confirmed_tok = tok1;
                  cur_pos = pos1 + 1;
                } else {
                  break;
                }
              } else {
                break;
              }
            } else {
              generated.push_back(tok0);
              ids.push_back(tok0);
              std::printf("step %d pos %d -> token %d [MTP draft %d REJECTED, alpha=%.3f (%d/%d)]\n",
                          (int)ids.size() - 1, pos0, tok0, draft_tok,
                          (float)mtpDraftAccepted / mtpDraftTotal,
                          mtpDraftAccepted, mtpDraftTotal);
              if (tok0 == 248046 || tok0 == 248044) {
                stopped_eos = true;
                break;
              }
              if ((int)generated.size() <= G && pos0 + 1 < MAXCTX) {
                CHECK(zeCommandListAppendMemoryCopy(up, dX, dX0, (size_t)H * 4, nullptr, 0, nullptr));
                draft1only(tok0, pos0);
                confirmed_tok = tok0;
                cur_pos = pos0 + 1;
              } else {
                break;
              }
            }
          }
        } else {
        // Seed draft-1 only; draft-2 is computed on demand by the first
        // M3 round (need_d2). Trunk dX holds hidden-after-tok.
        draft1only(tok, pos);
        need_d2 = true;
        }

        // G-counting matches the base loop (G+1 tokens when no EOS): base
        // overshoots max-new by one, so the MTP guards use <= G (T7.2 fix
        // for the 60-vs-61 stop bug). The M3/adaptive loop is skipped in
        // force-M2 mode (its standalone loop above already ran).
        while (force_depth != 2 && (int)generated.size() <= G &&
               cur_pos < MAXCTX - 2) {
          if (!want_m3()) {
            // Adaptive M2 round: single draft verify via the M2 lists.
            // pendingDraft is fresh (every round end re-drafts d1).
            int draft_tok = pendingDraft;
            int pos0 = cur_pos;
            int pos1 = cur_pos + 1;
            DecodeControl c0{confirmed_tok, pos0, pos0 + 1, -1};
            DecodeControl c1{draft_tok, pos1, pos1 + 1, -1};
            CHECK(zeCommandListAppendMemoryCopy(up, dCtrl0, &c0, sizeof(c0), nullptr, 0, nullptr));
            CHECK(zeCommandListAppendMemoryCopy(up, dCtrl1, &c1, sizeof(c1), nullptr, 0, nullptr));
            exec(embM2.h, "embed_m2");
            for (int L = 0; L < 64; ++L) {
              exec(layersM2[L].h, is_full(L) ? "attn_m2" : "lin_m2");
            }
            exec(tailM2.h, "tail_m2");
            int tok0 = -1, tok1 = -1;
            CHECK(zeCommandListAppendMemoryCopy(up, &tok0, dOutT0, 4, nullptr, 0, nullptr));
            CHECK(zeCommandListAppendMemoryCopy(up, &tok1, dOutT1, 4, nullptr, 0, nullptr));
            mtpDraftTotal++;
            bool acc2 = (tok0 == draft_tok);
            note_d1(acc2);
            if (acc2) {
              mtpDraftAccepted++;
              exec(commitSpecR.h, "commit_spec");
              generated.push_back(tok0);
              ids.push_back(tok0);
              std::printf("step %d pos %d -> token %d [MTP2 draft accepted, alpha=%.3f (%d/%d)]\n",
                          (int)ids.size() - 1, pos0, tok0,
                          (float)mtpDraftAccepted / mtpDraftTotal,
                          mtpDraftAccepted, mtpDraftTotal);
              if (tok0 == 248046 || tok0 == 248044) {
                stopped_eos = true;
                break;
              }
              if ((int)generated.size() <= G) {
                generated.push_back(tok1);
                ids.push_back(tok1);
                std::printf("step %d pos %d -> token %d [MTP2 verified]\n",
                            (int)ids.size() - 1, pos1, tok1);
                if (tok1 == 248046 || tok1 == 248044) {
                  stopped_eos = true;
                  break;
                }
                if ((int)generated.size() <= G && pos1 + 1 < MAXCTX) {
                  CHECK(zeCommandListAppendMemoryCopy(up, dX, dX1, (size_t)H * 4, nullptr, 0, nullptr));
                  draft1only(tok1, pos1);
                  need_d2 = true;
                  confirmed_tok = tok1;
                  cur_pos = pos1 + 1;
                } else {
                  break;
                }
              } else {
                break;
              }
            } else {
              generated.push_back(tok0);
              ids.push_back(tok0);
              std::printf("step %d pos %d -> token %d [MTP2 draft %d REJECTED, alpha=%.3f (%d/%d)]\n",
                          (int)ids.size() - 1, pos0, tok0, draft_tok,
                          (float)mtpDraftAccepted / mtpDraftTotal,
                          mtpDraftAccepted, mtpDraftTotal);
              if (tok0 == 248046 || tok0 == 248044) {
                stopped_eos = true;
                break;
              }
              if ((int)generated.size() <= G && pos0 + 1 < MAXCTX) {
                CHECK(zeCommandListAppendMemoryCopy(up, dX, dX0, (size_t)H * 4, nullptr, 0, nullptr));
                draft1only(tok0, pos0);
                need_d2 = true;
                confirmed_tok = tok0;
                cur_pos = pos0 + 1;
              } else {
                break;
              }
            }
            continue;
          }
          if (need_d2) {
            // Previous round was M2 (only d1 fresh): run the chained part.
            // d1 was drafted from (confirmed_tok, cur_pos - 1).
            draft2chain(cur_pos);
          }
          int sz0 = (int)generated.size();
          int draft_tok = pendingDraft;
          int draft2_tok = pendingDraft2;
          int pos0 = cur_pos;
          int pos1 = cur_pos + 1;
          int pos2 = cur_pos + 2;

          DecodeControl c0{confirmed_tok, pos0, pos0 + 1, -1};
          DecodeControl c1{draft_tok, pos1, pos1 + 1, -1};
          DecodeControl c2{draft2_tok, pos2, pos2 + 1, -1};
          CHECK(zeCommandListAppendMemoryCopy(up, dCtrl0, &c0, sizeof(c0), nullptr, 0, nullptr));
          CHECK(zeCommandListAppendMemoryCopy(up, dCtrl1, &c1, sizeof(c1), nullptr, 0, nullptr));
          CHECK(zeCommandListAppendMemoryCopy(up, dCtrl2, &c2, sizeof(c2), nullptr, 0, nullptr));

          exec(embM3.h, "embed_m3");
          for (int L = 0; L < 64; ++L) {
            exec(layersM3[L].h, is_full(L) ? "attn_m3" : "lin_m3");
          }
          exec(tailM3.h, "tail_m3");

          int tok0 = -1, tok1 = -1, tok2 = -1;
          CHECK(zeCommandListAppendMemoryCopy(up, &tok0, dOutT0, 4, nullptr, 0, nullptr));
          CHECK(zeCommandListAppendMemoryCopy(up, &tok1, dOutT1, 4, nullptr, 0, nullptr));
          CHECK(zeCommandListAppendMemoryCopy(up, &tok2, dOutT2, 4, nullptr, 0, nullptr));

          mtpDraftTotal++;
          bool acc1 = (tok0 == draft_tok);
          note_d1(acc1);
          auto push_tok = [&](int t, int p, const char *tag) {
            generated.push_back(t);
            ids.push_back(t);
            std::printf("step %d pos %d -> token %d [%s alpha=%.3f (%d/%d)]\n",
                        (int)ids.size() - 1, p, t, tag,
                        (float)mtpDraftAccepted / mtpDraftTotal,
                        mtpDraftAccepted, mtpDraftTotal);
          };
          auto is_eos = [&](int t) { return t == 248046 || t == 248044; };
          if (acc1) {
            mtpDraftAccepted++;
            mtpChainedTotal++;
            if (tok1 == draft2_tok) {
              // Accept-2: all three verified. Commit level-2 specular state.
              mtpChainedAccepted++;
              exec(commitSpec2R.h, "commit_spec2");
              push_tok(tok0, pos0, "MTP draft accepted");
              if (is_eos(tok0)) { stopped_eos = true; break; }
              if ((int)generated.size() <= G) {
                push_tok(tok1, pos1, "MTP chained accepted");
                if (is_eos(tok1)) { stopped_eos = true; break; }
                if ((int)generated.size() <= G) {
                  push_tok(tok2, pos2, "MTP verified");
                  if (is_eos(tok2)) { stopped_eos = true; break; }
                  if ((int)generated.size() <= G && pos2 + 1 < MAXCTX - 1) {
                    CHECK(zeCommandListAppendMemoryCopy(up, dX, dX2, (size_t)H * 4, nullptr, 0, nullptr));
                    note_trail(sz0, true);
                    redraft(tok2, pos2);
                    confirmed_tok = tok2;
                    cur_pos = pos2 + 1;
                  } else {
                    break;
                  }
                } else {
                  break;
                }
              } else {
                break;
              }
            } else {
              // Accept-1: tok0, tok1 verified; d2 rejected. Commit level-1.
              exec(commitSpecR.h, "commit_spec");
              push_tok(tok0, pos0, "MTP draft accepted");
              if (is_eos(tok0)) { stopped_eos = true; break; }
              if ((int)generated.size() <= G) {
                push_tok(tok1, pos1, "MTP chained REJECTED");
                if (is_eos(tok1)) { stopped_eos = true; break; }
                if ((int)generated.size() <= G && pos1 + 1 < MAXCTX - 1) {
                  CHECK(zeCommandListAppendMemoryCopy(up, dX, dX1, (size_t)H * 4, nullptr, 0, nullptr));
                  note_trail(sz0, false);
                    redraft(tok1, pos1);
                  confirmed_tok = tok1;
                  cur_pos = pos1 + 1;
                } else {
                  break;
                }
              } else {
                break;
              }
            }
          } else {
            // Accept-0 (reject): tok0 is verified true token; primary state
            // already correct, no commit. tok1/tok2 discarded.
            push_tok(tok0, pos0, "MTP draft REJECTED");
            if (is_eos(tok0)) { stopped_eos = true; break; }
            if ((int)generated.size() <= G && pos0 + 1 < MAXCTX - 1) {
              CHECK(zeCommandListAppendMemoryCopy(up, dX, dX0, (size_t)H * 4, nullptr, 0, nullptr));
              note_trail(sz0, false);
              redraft(tok0, pos0);
              confirmed_tok = tok0;
              cur_pos = pos0 + 1;
            } else {
              break;
            }
          }
        }
        total_gen_ms = (now_ns() - tGenStart) / 1e6;
        gen_tok_per_sec = (generated.size() > 0 && total_gen_ms > 0) ? (generated.size() / (total_gen_ms / 1000.0)) : 0.0;
        std::printf("[MTP Speculative Decode] %zu tokens generated in %.1f ms = %.2f tok/s (%.2f ms/token)\n",
                    generated.size(), total_gen_ms, gen_tok_per_sec,
                    gen_tok_per_sec > 0 ? 1000.0 / gen_tok_per_sec : 0.0);
        break;
      }
      if (step >= P - 1 && (tok == 248046 || tok == 248044)) {
        stopped_eos = true;
        break;
      }
    }
  }
  if (!opt_mtp && tGenStart > 0) {
    total_gen_ms = (now_ns() - tGenStart) / 1e6;
    gen_tok_per_sec = (generated.size() > 0 && total_gen_ms > 0) ? (generated.size() / (total_gen_ms / 1000.0)) : 0.0;
    std::printf("[Normal Decode] %zu tokens generated in %.1f ms = %.2f tok/s (%.2f ms/token)\n",
                generated.size(), total_gen_ms, gen_tok_per_sec,
                gen_tok_per_sec > 0 ? 1000.0 / gen_tok_per_sec : 0.0);
  }
  if (opt_mtp && mtpDraftTotal > 0) {
    std::printf("[MTP summary] %d/%d accepted, alpha = %.3f",
                mtpDraftAccepted, mtpDraftTotal,
                (float)mtpDraftAccepted / mtpDraftTotal);
    if (mtpChainedTotal > 0)
      std::printf(" chained %d/%d = %.3f", mtpChainedAccepted, mtpChainedTotal,
                  (float)mtpChainedAccepted / mtpChainedTotal);
    std::printf("\n");
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
  json += "]";
  if (preChunks > 0) {
    char pb[64];
    std::snprintf(pb, sizeof pb, ",\"prefill_ms\":%.1f", prefill_ms);
    json += pb;
  }
  if (opt_mtp) {
    char mb[192];
    std::snprintf(mb, sizeof mb,
                  ",\"mtp\":{\"total\":%d,\"accepted\":%d,\"alpha\":%.4f,"
                  "\"chained_total\":%d,\"chained_accepted\":%d}",
                  mtpDraftTotal, mtpDraftAccepted,
                  mtpDraftTotal > 0 ? (float)mtpDraftAccepted / mtpDraftTotal : 0.0f,
                  mtpChainedTotal, mtpChainedAccepted);
    json += mb;
  }
  if (total_gen_ms > 0) {
    char db[128];
    std::snprintf(db, sizeof db, ",\"decode_ms\":%.1f,\"decode_tok_per_sec\":%.2f",
                  total_gen_ms, gen_tok_per_sec);
    json += db;
  }
  json += "}";
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
