// AInfer T4.2-device: full 64-layer decode loop with KV + SSM caches.
// Prefill by looping the decode step (correctness-first), then generate.
// Caches: KV for 16 full-attn layers (max_ctx), conv+SSM state for 48 linear.
// Uses device argmax (T3.9). All GEMVs are INT4 dp4a from static arenas.
// Usage: decode <model.binfer> <prompt_len> <gen_len> [report.json]
#include <level_zero/ze_api.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;

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
static sycl::device pick_b60() {
  for (auto p : sycl::platform::get_platforms())
    for (auto d : p.get_devices(sycl::info::device_type::gpu))
      if (d.get_info<sycl::info::device::name>().find("B60") !=
          std::string::npos)
        return d;
  std::fprintf(stderr, "FATAL: no B60\n");
  std::exit(1);
}
static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}

// ---- kernel tags (one per shape-class; params carry per-layer pointers) ----
struct KEmb {};
struct KNorm {};
struct KMax {};
struct KQuant {};
struct KGemv {};
struct KSpl {};
struct KRope {};
struct KKv {};
struct KGqa {};
struct KRes {};
struct KConv {};
struct KRec {};
struct KBG {};
struct KNG {};
struct KSilu {};
struct KA1 {};
struct KA2 {};
struct KSpl2 {};
struct KNorm2 {};
struct KNG2 {};
struct KRes2 {};

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: decode <binfer> <prompt_len> <gen_len> "
                         "[report.json]\n");
    return 2;
  }
  std::vector<int> opt_ids;
  int opt_max_new = -1;
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
    } else {
      pos.push_back(argv[i]);
    }
  }
  if (pos.size() < 3) {
    std::fprintf(stderr, "usage: decode <binfer> <prompt_len> <gen_len> "
                         "[report.json [dump_step [dump_path]]] [--ids=..] [--max-new=..]\n");
    return 2;
  }
  const char *path = pos[0];
  int P = atoi(pos[1]), G = atoi(pos[2]);
  int dump_step = (pos.size() > 4) ? atoi(pos[4]) : -1;
  std::vector<float> dump_states; // 64*5120 when dumping
  std::vector<float> dump_states_b;
  int dump_step2 = 3;
  size_t dump_layer_count[2] = {0, 0};
  // Effective P/G must be final BEFORE sizing caches: --ids= resets P and
  // --max-new= resets G, and the old order sized MAXCTX from the positional
  // args (e.g. G=1) while generating far past it -> KV/RoPE OOB at pos>=MAXCTX
  // (T4.5/T6.3 collapse at pos 79 was exactly this, not quantization drift).
  // NOTE: ids parsing for --ids= happens here so P is final; G likewise.
  {
    if (!opt_ids.empty())
      P = (int)opt_ids.size();
    if (opt_max_new >= 0)
      G = opt_max_new;
  }
  const int MAXCTX = P + G + 8;
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
  std::vector<Entry> entries(n);
  f.seekg((std::streamoff)dir_off);
  for (uint64_t i = 0; i < n; ++i) {
    Entry &e = entries[i];
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
    for (auto &e : entries)
      if (nm == e.name)
        return &e;
    return nullptr;
  };

  // reject truncated/short files: every span must lie inside [0, fsize-32]
  // (trailing sha). Without this, short reads silently feed zeros (T4.5).
  f.seekg(0, std::ios::end);
  {
    uint64_t fsize = (uint64_t)f.tellg();
    f.clear();
    for (auto &e : entries) {
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
  uint64_t pay_lo = UINT64_MAX, pay_hi = 0, sc_lo = UINT64_MAX, sc_hi = 0;
  for (auto &e : entries) {
    pay_lo = e.d_off < pay_lo ? e.d_off : pay_lo;
    pay_hi = e.d_off + e.d_bytes > pay_hi ? e.d_off + e.d_bytes : pay_hi;
    if (e.sc_bytes) {
      sc_lo = e.sc_off < sc_lo ? e.sc_off : sc_lo;
      sc_hi = e.sc_off + e.sc_bytes > sc_hi ? e.sc_off + e.sc_bytes : sc_hi;
    }
  }
  ze_device_mem_alloc_desc_t mdesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  void *payArena = nullptr, *scArena = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)(pay_hi - pay_lo), 4096, dev,
                         &payArena));
  uint64_t scSize = sc_hi - sc_lo;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)scSize, 4096, dev, &scArena));
  ze_command_list_handle_t list = nullptr;
  ze_command_queue_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &ldesc, &list));
  const size_t CH = 1u << 28;
  std::vector<char> staging(CH);
  auto up = [&](uint64_t foff, void *arena, uint64_t alo, uint64_t len) {
    uint64_t done = 0;
    while (done < len) {
      size_t c = (size_t)((len - done > CH) ? CH : (len - done));
      f.clear();
      f.seekg((std::streamoff)(foff + done));
      f.read(staging.data(), c);
      CHECK(zeCommandListAppendMemoryCopy(
          list, (char *)arena + (foff + done - alo), staging.data(), c,
          nullptr, 0, nullptr));
      done += c;
    }
  };
  up(pay_lo, payArena, pay_lo, pay_hi - pay_lo);
  up(sc_lo, scArena, sc_lo, scSize);

  sycl::device sdev = pick_b60();
  sycl::queue q(sdev, {sycl::property::queue::enable_profiling()});
  const char *LP = "model.language_model.layers.";
  // layer types: every 4th is full_attention (index %4 == 3)
  auto is_full = [](int L) { return (L % 4) == 3; };
  auto nm = [&](int L, const char *s) {
    return std::string(LP) + std::to_string(L) + "." + s;
  };

  // ---- caches / work buffers ----
  const int D = 128, NK = 16, NV = 48, KD = 2048, VD = 6144, CONVD = 10240;
  // T5.1 FIX: kvSz already spans all 16 full-layer slots
  // (16 layers x MAXCTX x 4 KV heads x 256); the old `* 16` over-allocated 16x
  // (8.6 GB phantom at 4K context -> would OOM the 24 GB card).
  size_t kvSz = (size_t)16 * MAXCTX * 4 * 256;
  float *dKc = sycl::malloc_device<float>(kvSz, q);
  float *dVc = sycl::malloc_device<float>(kvSz, q);
  float *dConv = sycl::malloc_device<float>((size_t)48 * CONVD * 3, q);
  float *dS = sycl::malloc_device<float>((size_t)48 * NV * D * D, q);
  q.memset(dConv, 0, (size_t)48 * CONVD * 3 * 4).wait();
  q.memset(dS, 0, (size_t)48 * NV * D * D * 4).wait();
  float *dX = sycl::malloc_device<float>(5120, q);
  float *dH = sycl::malloc_device<float>(5120, q);
  int8_t *dQ8 = sycl::malloc_device<int8_t>(17408, q);
  float *dQKV = sycl::malloc_device<float>(10240, q);
  float *dZ = sycl::malloc_device<float>(6144, q);
  float *dB48 = sycl::malloc_device<float>(48, q);
  float *dA48 = sycl::malloc_device<float>(48, q);
  float *dG48 = sycl::malloc_device<float>(48, q);
  float *dBt48 = sycl::malloc_device<float>(48, q);
  float *dMx = sycl::malloc_device<float>(10240, q);
  float *dQ48 = sycl::malloc_device<float>(6144, q);
  float *dK48 = sycl::malloc_device<float>(6144, q);
  float *dV48 = sycl::malloc_device<float>(6144, q);
  float *dQ16 = sycl::malloc_device<float>(12288, q);
  float *dK16 = sycl::malloc_device<float>(1024, q);
  float *dV16 = sycl::malloc_device<float>(1024, q);
  float *dQn = sycl::malloc_device<float>(6144, q);
  float *dKn = sycl::malloc_device<float>(1024, q);
  float *dGate = sycl::malloc_device<float>(6144, q);
  float *dAtt = sycl::malloc_device<float>(6144, q);
  float *dWts = sycl::malloc_device<float>(24 * MAXCTX, q);
  float *dMix = sycl::malloc_device<float>(5120, q);
  float *dTmp = sycl::malloc_device<float>(5120, q);
  float *dG17 = sycl::malloc_device<float>(17408, q);
  float *dU17 = sycl::malloc_device<float>(17408, q);
  float *dLogits = sycl::malloc_device<float>(248320, q);
  float *dPV = sycl::malloc_device<float>(64, q);
  int *dPI = sycl::malloc_device<int>(64, q);
  // T5.5: token handoff allocation. Default (explicit): device int + 4 B D2H
  // copy per gen step. AINFER_TOKEN_SHARED=1: SYCL-shared int, host reads it
  // directly after the queue wait (no copy command). Winner kept per measure.
  const bool tokShared = std::getenv("AINFER_TOKEN_SHARED") != nullptr;
  int *dOutT = tokShared ? sycl::malloc_shared<int>(1, q)
                         : sycl::malloc_device<int>(1, q);
  // (T5.1: per-layer small weights now preloaded in dInNorm/dPostNorm/...;
  // no per-step staging buffers remain)
  float *dCos = sycl::malloc_device<float>(64 * MAXCTX, q);
  float *dSin = sycl::malloc_device<float>(64 * MAXCTX, q);

  // ---- T5.1: preload all small (BF16, non-quantized) tensors ONCE ----
  // Old loop did ~260 file seeks+reads per token (norms/A_log/dt/conv per layer
  // per step). Now: single streaming pass at init, fixed device addresses.
  float *dInNorm = sycl::malloc_device<float>((size_t)64 * 5120, q);
  float *dPostNorm = sycl::malloc_device<float>((size_t)64 * 5120, q);
  float *dQNorm = sycl::malloc_device<float>((size_t)16 * 256, q);
  float *dKNorm = sycl::malloc_device<float>((size_t)16 * 256, q);
  float *dLinNorm = sycl::malloc_device<float>((size_t)48 * 128, q);
  float *dAlogA = sycl::malloc_device<float>((size_t)48 * 48, q);
  float *dDtA = sycl::malloc_device<float>((size_t)48 * 48, q);
  float *dConvW = sycl::malloc_device<float>((size_t)48 * CONVD * 4, q);
  float *dFinNorm = sycl::malloc_device<float>(5120, q);
  auto preload = [&](const std::string &name, float *dst, size_t cnt) {
    const Entry *e = find(name);
    if (!e) {
      std::fprintf(stderr, "FATAL preload: missing %s\n", name.c_str());
      std::exit(1);
    }
    std::vector<uint16_t> raw(cnt);
    f.clear();
    f.seekg((std::streamoff)e->d_off);
    f.read((char *)raw.data(), cnt * 2);
    std::vector<float> out(cnt);
    for (size_t i = 0; i < cnt; ++i)
      out[i] = bf16_to_f32(raw[i]);
    q.memcpy(dst, out.data(), cnt * 4).wait();
  };
  for (int L = 0; L < 64; ++L) {
    preload(nm(L, "input_layernorm.weight"), dInNorm + (size_t)L * 5120, 5120);
    preload(nm(L, "post_attention_layernorm.weight"),
            dPostNorm + (size_t)L * 5120, 5120);
    if (is_full(L)) {
      int slot = L / 4;
      preload(nm(L, "self_attn.q_norm.weight"), dQNorm + (size_t)slot * 256,
              256);
      preload(nm(L, "self_attn.k_norm.weight"), dKNorm + (size_t)slot * 256,
              256);
    } else {
      int sl = L - (L + 1) / 4;
      preload(nm(L, "linear_attn.norm.weight"), dLinNorm + (size_t)sl * 128,
              128);
      preload(nm(L, "linear_attn.A_log"), dAlogA + (size_t)sl * 48, 48);
      preload(nm(L, "linear_attn.dt_bias"), dDtA + (size_t)sl * 48, 48);
      preload(nm(L, "linear_attn.conv1d.weight"),
              dConvW + (size_t)sl * CONVD * 4, (size_t)CONVD * 4);
    }
  }
  preload("model.language_model.norm.weight", dFinNorm, 5120);
  // T5.1 liveness evidence: one-time static footprint report (stderr).
  // All buffers below are allocated once at init; none in the token loop.
  {
    // straight-line accounting (bytes, FP32 unless noted)
    size_t kvB = kvSz * 4 * 2;                 // K+V caches (FP32)
    size_t ssmB = (size_t)48 * NV * D * D * 4; // recurrent state (FP32)
    size_t convB = (size_t)48 * CONVD * 3 * 4; // conv shift state
    size_t tokB = 0;                           // single-token activation scratch
    tokB += (size_t)(5120 + 5120 + 5120 + 5120) * 4; // dX,dH,dMix,dTmp
    tokB += (size_t)(10240 + 6144 + 6144 + 6144 + 6144) * 4; // dQKV,dZ,dQ48,dK48,dV48
    tokB += (size_t)(12288 + 1024 + 1024) * 4; // dQ16,dK16,dV16
    tokB += (size_t)(6144 + 1024) * 4;         // dQn,dKn
    tokB += (size_t)(6144 + 6144) * 4;         // dGate,dAtt
    tokB += (size_t)17408 * 4 * 2;             // dG17,dU17
    tokB += (size_t)248320 * 4;                // dLogits
    tokB += (size_t)48 * 4 * 4;                // dB48,dA48,dG48,dBt48
    tokB += (size_t)10240 * 4;                 // dMx
    tokB += (size_t)136 * 4 + (size_t)17408;   // dSq + dQ8
    tokB += (size_t)24 * MAXCTX * 4;           // dWts
    tokB += (size_t)64 * MAXCTX * 4 * 2;       // dCos,dSin
    tokB += (size_t)64 * 4 + 64 * 4;           // dPV,dPI,dOutT
    size_t smallB = 0;                         // preloaded small weights
    smallB += (size_t)64 * 5120 * 4 * 2;       // in/post norms
    smallB += (size_t)16 * 256 * 4 * 2;        // q/k norms
    smallB += (size_t)48 * 128 * 4;            // lin norms
    smallB += (size_t)48 * 48 * 4 * 2;         // A_log/dt_bias
    smallB += (size_t)48 * CONVD * 4 * 4;      // conv weights
    smallB += (size_t)5120 * 4;                // final norm
    std::fprintf(stderr,
                 "[t51] MAXCTX=%d static MiB: KV=%.1f SSM=%.1f convState=%.1f "
                 "tokScratch=%.1f smallW=%.1f (model arenas separate)\n",
                 MAXCTX, kvB / 1048576.0, ssmB / 1048576.0,
                 convB / 1048576.0, tokB / 1048576.0, smallB / 1048576.0);
  }
  // rope tables for all positions (recomputed on host, uploaded once)
  {
    std::vector<float> cs(64 * MAXCTX), sn(64 * MAXCTX);
    for (int t = 0; t < MAXCTX; ++t)
      for (int i = 0; i < 64; ++i) {
        double inv = 1.0 / std::pow(10000000.0,
                                    (double)(2 * (i % 32)) / 64.0);
        double ang = (double)t * inv;
        cs[t * 64 + i] = (float)std::cos(ang);
        sn[t * 64 + i] = (float)std::sin(ang);
      }
    q.memcpy(dCos, cs.data(), cs.size() * 4).wait();
    q.memcpy(dSin, sn.data(), sn.size() * 4).wait();
  }

  // T6.3: per-group-128 INT8 activation scales (replaces per-tensor sq).
  // Diagnostic (tools/t63/report_actquant.json, real L0 MLP vectors): per-group
  // halves zero-rate (16.2%->8.3%), +6.8dB SNR, 2.2x downstream down_proj error
  // reduction vs per-tensor. Cost: 136 extra scales/GEMV (negligible).
  float *dSq = sycl::malloc_device<float>(136, q); // max K=17408 -> 136 groups
  auto gemvA = [&](const Entry *e, int M, int K, float *Y) {
    const uint8_t *dP = (const uint8_t *)payArena + (e->d_off - pay_lo);
    const uint16_t *dSc =
        (const uint16_t *)((const char *)scArena + (e->sc_off - sc_lo));
    auto ev = q.submit([&](sycl::handler &h) {
      h.parallel_for<KGemv>(sycl::range<1>(M), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
        int m = id[0];
        float acc = 0;
        int Gg = K / 128;
        for (int g = 0; g < Gg; ++g) {
          uint32_t ub = dSc[(size_t)m * Gg + g];
          ub <<= 16;
          float sc;
          __builtin_memcpy(&sc, &ub, 4);
          float sqg = dSq[g];
          size_t base = (size_t)m * K / 2 + (size_t)g * 64;
          esimd::simd<int, 8> acc8(0);
          for (int j = 0; j < 128; j += 32) {
            esimd::simd<unsigned char, 16> bytes =
                esimd::block_load<unsigned char, 16>(dP + base + j / 2);
            esimd::simd<unsigned short, 32> v;
            v.template select<16, 2>(0) =
                esimd::convert<unsigned short>(bytes & 0xFu);
            v.template select<16, 2>(1) =
                esimd::convert<unsigned short>(bytes >> 4);
            esimd::simd<short, 32> sv = esimd::convert<short>(v);
            sv -= (sv & 8) << 1;
            esimd::simd<int8_t, 32> w8 = esimd::convert<int8_t>(sv);
            esimd::simd<int8_t, 32> xq =
                esimd::block_load<int8_t, 32>(dQ8 + g * 128 + j);
            esimd::simd<int, 8> w32 = w8.bit_cast_view<int>();
            esimd::simd<int, 8> x32 = xq.bit_cast_view<int>();
            acc8 = esimd::dp4a<int>(acc8, w32, x32);
          }
          int tmp[8];
          acc8.copy_to(tmp);
          int gs = 0;
          for (int u = 0; u < 8; ++u)
            gs += tmp[u];
          acc += (float)gs * sc * sqg;
        }
        Y[m] = acc;
      });
    });
    ev.wait();
  };
  // Per-group-128 INT8 activation scales, computed DEVICE-side (T5.3 enabler).
  // Old path read the whole activation back to host and re-uploaded 136
  // scales per GEMV (~700 D2H+H2D round trips per token) — that readback is
  // what blocks whole-loop list recording. Max-abs is order-independent and
  // fabs/ternary match the old std::fabs/std::max bitwise (incl. NaN/zero),
  // so dSq — and everything downstream — is bit-identical to the host path.
  auto xscales_of = [&](const float *b, int N) {
    int G = N / 128;
    auto ev = q.submit([&](sycl::handler &h) {
      h.parallel_for<KMax>(sycl::range<1>(G), [=](sycl::id<1> id) {
        int g = id[0];
        float m = 0;
        for (int j = 0; j < 128; ++j) {
          float a = sycl::fabs(b[g * 128 + j]);
          m = a > m ? a : m;
        }
        float s = m / 127.0f;
        dSq[g] = s == 0 ? 1.0f : s;
      });
    });
    ev.wait();
    return G;
  };
  auto quantize = [&](const float *X, int K) {
    auto ev = q.submit([&](sycl::handler &h) {
      h.parallel_for<KQuant>(sycl::range<1>(K), [=](sycl::id<1> id) {
        float s = dSq[id[0] / 128];
        float v = X[id[0]] / s;
        int qi = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
        dQ8[id[0]] = (int8_t)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
      });
    });
    ev.wait();
  };
  auto rmsnorm = [&](const float *X, const float *W, float *Y, int rows, int Dd,
                     bool oneplus) {
    auto ev = q.submit([&](sycl::handler &h) {
      h.parallel_for<KNorm>(sycl::range<1>(rows), [=](sycl::id<1> id) {
        int r = id[0];
        double ss = 0;
        for (int j = 0; j < Dd; ++j) {
          float v = X[r * Dd + j];
          ss += (double)v * v;
        }
        float inv = 1.0f / std::sqrt((float)(ss / Dd) + 1e-6f);
        for (int j = 0; j < Dd; ++j)
          Y[r * Dd + j] =
              X[r * Dd + j] * inv * (oneplus ? (1.0f + W[j]) : W[j]);
      });
    });
    ev.wait();
  };

  // ---- prompt ids: reuse T1.5 corpus-ish deterministic prompt ----
  // "What is 84 * 3 / 2?" encoded with <|im_start|>user ... assistant
  std::vector<int> ids = {248045, 846,  198, 3710, 369, 220, 23, 19,
                          348,    220,  18,  593,  220, 17,  30, 248046,
                          198,    248045, 74455, 198, 248068, 271, 248069, 271};
  // P/G already finalized above (before MAXCTX); only build the id vector here.
  if (!opt_ids.empty()) {
    ids = opt_ids;
  } else if ((int)ids.size() > P) {
    ids.resize(P);
  } else {
    while ((int)ids.size() < P)
      ids.push_back(198);
  }
  ids.reserve((size_t)P + (size_t)G); // T5.2: no growth allocation in gen loop

  // ---- embed lookup kernel (BF16 rows) ----
  const Entry *Eemb = find("model.language_model.embed_tokens.weight");
  const uint16_t *embP = (const uint16_t *)((const char *)payArena +
                                            (Eemb->d_off - pay_lo));
  auto embed = [&](int tok) {
    auto ev = q.submit([&](sycl::handler &h) {
      h.parallel_for<KEmb>(sycl::range<1>(5120), [=](sycl::id<1> id) {
        uint16_t b = embP[(size_t)tok * 5120 + id[0]];
        uint32_t u = (uint32_t)b << 16;
        float x;
        __builtin_memcpy(&x, &u, 4);
        dX[id[0]] = x;
      });
    });
    ev.wait();
  };

  std::vector<int> generated;
  std::vector<std::array<int, 5>> tops5;
  std::vector<std::array<float, 5>> tops5v;
  std::vector<float> logits_host(248320);
  std::vector<int> logit_indices(248320);
  generated.reserve((size_t)G + 1);
  tops5.reserve((size_t)G + 1);
  tops5v.reserve((size_t)G + 1);
  double t0 = 0;
  // measure only the generate phase structure; report per-step ms below
  bool stopped_eos = false;
  const char *top5env = std::getenv("AINFER_TOP5");
  const bool top5diag = !top5env || std::atoi(top5env) != 0;
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
    embed(ids[step]);
    for (int L = 0; L < 64; ++L) {
      // ---- input norm (T5.1: preloaded) ----
      rmsnorm(dX, dInNorm + (size_t)L * 5120, dH, 1, 5120, true);
      if (is_full(L)) {
        // ---- attention block ----
        xscales_of(dH, 5120);
        quantize(dH, 5120);
        gemvA(find(nm(L, "self_attn.q_proj.weight")), 12288, 5120, dQ16);
        gemvA(find(nm(L, "self_attn.k_proj.weight")), 1024, 5120, dK16);
        gemvA(find(nm(L, "self_attn.v_proj.weight")), 1024, 5120, dV16);
        // split q/gate
        {
          auto ev = q.submit([&](sycl::handler &h) {
            h.parallel_for<KSpl>(sycl::range<1>(6144), [=](sycl::id<1> id) {
              int i = id[0], hh = i / 256, d = i % 256;
              dQn[i] = dQ16[(size_t)hh * 512 + d];
              dGate[i] = dQ16[(size_t)hh * 512 + 256 + d];
            });
          });
          ev.wait();
        }
        rmsnorm(dQn, dQNorm + (size_t)(L / 4) * 256, dQn, 24, 256, true);
        rmsnorm(dK16, dKNorm + (size_t)(L / 4) * 256, dKn, 4, 256, true);
        // rope at pos
        {
          auto ev = q.submit([&](sycl::handler &h) {
            h.parallel_for<KRope>(sycl::range<1>(28), [=](sycl::id<1> id) {
              int i = id[0];
              float *X = i < 24 ? dQn + (size_t)i * 256
                               : dKn + (size_t)(i - 24) * 256;
              for (int d = 0; d < 32; ++d) {
                float x0 = X[d], x1 = X[d + 32];
                float c = dCos[pos * 64 + d], s = dSin[pos * 64 + d];
                X[d] = x0 * c - x1 * s;
                X[d + 32] = x0 * s + x1 * c;
              }
            });
          });
          ev.wait();
        }
        int slot = L / 4; // 0..15
        // kv append
        {
          auto ev = q.submit([&](sycl::handler &h) {
            h.parallel_for<KKv>(sycl::range<1>(4 * 256), [=](sycl::id<1> id) {
              int i = id[0], hh = i / 256, d = i % 256;
              dKc[(((size_t)slot * MAXCTX + pos) * 4 + hh) * 256 + d] =
                  dKn[(size_t)hh * 256 + d];
              dVc[(((size_t)slot * MAXCTX + pos) * 4 + hh) * 256 + d] =
                  dV16[(size_t)hh * 256 + d];
            });
          });
          ev.wait();
        }
        // gqa over 0..pos
        {
          auto ev = q.submit([&](sycl::handler &h) {
            h.parallel_for<KGqa>(sycl::range<1>(24), [=](sycl::id<1> id) {
              int hh = id[0], kv = hh / 6, T = pos + 1;
              float mx = -1e30f;
              for (int t = 0; t < T; ++t) {
                float s = 0;
                for (int d = 0; d < 256; ++d)
                  s += dQn[(size_t)hh * 256 + d] *
                       dKc[(((size_t)slot * MAXCTX + t) * 4 + kv) * 256 + d];
                s /= 16.0f;
                dWts[(size_t)hh * MAXCTX + t] = s;
                mx = s > mx ? s : mx;
              }
              float se = 0;
              for (int t = 0; t < T; ++t) {
                float w = std::exp(dWts[(size_t)hh * MAXCTX + t] - mx);
                dWts[(size_t)hh * MAXCTX + t] = w;
                se += w;
              }
              for (int d = 0; d < 256; ++d) {
                float acc = 0;
                for (int t = 0; t < T; ++t)
                  acc += dWts[(size_t)hh * MAXCTX + t] / se *
                         dVc[(((size_t)slot * MAXCTX + t) * 4 + kv) * 256 + d];
                float g = dGate[(size_t)hh * 256 + d];
                dAtt[(size_t)hh * 256 + d] = acc / (1.0f + std::exp(-g));
              }
            });
          });
          ev.wait();
        }
        xscales_of(dAtt, 6144);
        quantize(dAtt, 6144);
        gemvA(find(nm(L, "self_attn.o_proj.weight")), 5120, 6144, dMix);
      } else {
        // ---- linear block ----
        // index among the 48 linear layers (full layers are L%4==3)
        int sl = L - (L + 1) / 4;
        xscales_of(dH, 5120);
        quantize(dH, 5120);
        gemvA(find(nm(L, "linear_attn.in_proj_qkv.weight")), CONVD, 5120,
              dQKV);
        gemvA(find(nm(L, "linear_attn.in_proj_z.weight")), VD, 5120, dZ);
        gemvA(find(nm(L, "linear_attn.in_proj_b.weight")), 48, 5120, dB48);
        gemvA(find(nm(L, "linear_attn.in_proj_a.weight")), 48, 5120, dA48);
        // conv weights (T5.1: preloaded in dConvW)
        float *cs = dConv + (size_t)sl * CONVD * 3;
        const float *CW = dConvW + (size_t)sl * CONVD * 4;
        {
          auto ev = q.submit([&](sycl::handler &h) {
            h.parallel_for<KConv>(sycl::range<1>(CONVD), [=](sycl::id<1> id) {
              int c = id[0];
              float acc = cs[c * 3 + 0] * CW[(size_t)c * 4 + 0] +
                          cs[c * 3 + 1] * CW[(size_t)c * 4 + 1] +
                          cs[c * 3 + 2] * CW[(size_t)c * 4 + 2] +
                          dQKV[c] * CW[(size_t)c * 4 + 3];
              dMx[c] = acc / (1.0f + sycl::exp(-acc));
              cs[c * 3 + 0] = cs[c * 3 + 1];
              cs[c * 3 + 1] = cs[c * 3 + 2];
              cs[c * 3 + 2] = dQKV[c];
            });
          });
          ev.wait();
        }
        // split + repeat
        {
          auto ev = q.submit([&](sycl::handler &h) {
            h.parallel_for<KSpl2>(sycl::range<1>(6144), [=](sycl::id<1> id) {
              int i = id[0], hh = i / 128, d = i % 128, kh = hh / 3;
              dQ48[i] = dMx[(size_t)kh * 128 + d];
              dK48[i] = dMx[(size_t)(2048 + kh * 128) + d];
              dV48[i] = dMx[(size_t)(4096 + hh * 128) + d];
            });
          });
          ev.wait();
        }
        // l2norm + q scale
        {
          auto ev = q.submit([&](sycl::handler &h) {
            h.parallel_for<KNorm2>(sycl::range<1>(96), [=](sycl::id<1> id) {
              int i = id[0];
              float *X = i < 48 ? dQ48 + (size_t)i * 128
                                : dK48 + (size_t)(i - 48) * 128;
              float ss = 0;
              for (int d = 0; d < 128; ++d)
                ss += X[d] * X[d];
              float inv = 1.0f / sycl::sqrt(ss + 1e-6f);
              for (int d = 0; d < 128; ++d)
                X[d] *= inv * (i < 48 ? 0.0883883476f : 1.0f);
            });
          });
          ev.wait();
        }
        // beta, g (T5.1: A_log/dt_bias preloaded in dAlogA/dDtA)
        {
          const float *AL = dAlogA + (size_t)sl * 48;
          const float *DT = dDtA + (size_t)sl * 48;
          auto ev = q.submit([&](sycl::handler &h) {
            h.parallel_for<KBG>(sycl::range<1>(48), [=](sycl::id<1> id) {
              int hh = id[0];
              dBt48[hh] = 1.0f / (1.0f + sycl::exp(-dB48[hh]));
              float sa = dA48[hh] + DT[hh];
              float soft = sa > 20 ? sa : sycl::log(1.0f + sycl::exp(sa));
              dG48[hh] = -sycl::exp(AL[hh]) * soft;
            });
          });
          ev.wait();
        }
        // recurrent (state persists in dS)
        {
          float *Sh = dS + (size_t)sl * NV * D * D;
          auto ev = q.submit([&](sycl::handler &h) {
            h.parallel_for<KRec>(sycl::range<1>(48), [=](sycl::id<1> id) {
              int hh = id[0];
              float *S0 = Sh + (size_t)hh * D * D;
              float *qh = dQ48 + (size_t)hh * 128, *kh = dK48 + (size_t)hh * 128,
                    *vh = dV48 + (size_t)hh * 128;
              float gt = sycl::exp(dG48[hh]), bt = dBt48[hh];
              for (int i = 0; i < D * D; ++i)
                S0[i] *= gt;
              float kv[128];
              for (int v = 0; v < D; ++v) {
                float s = 0;
                for (int k = 0; k < D; ++k)
                  s += S0[k * D + v] * kh[k];
                kv[v] = s;
              }
              for (int v = 0; v < D; ++v)
                kv[v] = (vh[v] - kv[v]) * bt;
              for (int k = 0; k < D; ++k)
                for (int v = 0; v < D; ++v)
                  S0[k * D + v] += kh[k] * kv[v];
              for (int v = 0; v < D; ++v) {
                float s = 0;
                for (int k = 0; k < D; ++k)
                  s += S0[k * D + v] * qh[k];
                dMx[(size_t)hh * 128 + v] = s;
              }
            });
          });
          ev.wait();
        }
        // norm-gated (T5.1: preloaded in dLinNorm)
        const float *NG = dLinNorm + (size_t)sl * 128;
        {
          auto ev = q.submit([&](sycl::handler &h) {
            h.parallel_for<KNG>(sycl::range<1>(48), [=](sycl::id<1> id) {
              int hh = id[0];
              float ss = 0;
              for (int d = 0; d < 128; ++d) {
                float v = dMx[(size_t)hh * 128 + d];
                ss += v * v;
              }
              dBt48[hh] = 1.0f / sycl::sqrt(ss / 128 + 1e-6f);
            });
          });
          ev.wait();
          auto ev2 = q.submit([&](sycl::handler &h) {
            h.parallel_for<KNG2>(sycl::range<1>(6144), [=](sycl::id<1> id) {
              int i = id[0], hh = i / 128, d = i % 128;
              float zv = dZ[(size_t)hh * 128 + d];
              dAtt[i] = NG[d] * dMx[i] * dBt48[hh] *
                        (zv / (1.0f + sycl::exp(-zv)));
            });
          });
          ev2.wait();
        }
        xscales_of(dAtt, 6144);
        quantize(dAtt, 6144);
        gemvA(find(nm(L, "linear_attn.out_proj.weight")), 5120, 6144,
              dMix);
      }
      // residual -> mid
      {
        auto ev = q.submit([&](sycl::handler &h) {
          h.parallel_for<KRes>(sycl::range<1>(5120), [=](sycl::id<1> id) {
            dTmp[id[0]] = dX[id[0]] + dMix[id[0]];
          });
        });
        ev.wait();
      }
      // post norm + mlp (T5.1: preloaded)
      rmsnorm(dTmp, dPostNorm + (size_t)L * 5120, dH, 1, 5120, true);
      xscales_of(dH, 5120);
      quantize(dH, 5120);
      gemvA(find(nm(L, "mlp.gate_proj.weight")), 17408, 5120, dG17);
      gemvA(find(nm(L, "mlp.up_proj.weight")), 17408, 5120, dU17);
      {
        auto ev = q.submit([&](sycl::handler &h) {
          h.parallel_for<KSilu>(sycl::range<1>(17408), [=](sycl::id<1> id) {
            float g = dG17[id[0]];
            dG17[id[0]] = (g / (1.0f + sycl::exp(-g))) * dU17[id[0]];
          });
        });
        ev.wait();
      }
      xscales_of(dG17, 17408);
      quantize(dG17, 17408);
      gemvA(find(nm(L, "mlp.down_proj.weight")), 5120, 17408, dMix);
      {
        auto ev = q.submit([&](sycl::handler &h) {
          h.parallel_for<KRes2>(sycl::range<1>(5120), [=](sycl::id<1> id) {
            dX[id[0]] = dTmp[id[0]] + dMix[id[0]];
          });
        });
        ev.wait();
      }
      if (getenv("AINFER_TRACE") && step >= P - 1) {
        // unbuffered first-NaN-layer trace (stderr)
        std::vector<float> hx(5120);
        q.memcpy(hx.data(), dX, (size_t)5120 * 4).wait();
        int nn = 0;
        for (int i = 0; i < 5120; ++i)
          if (std::isnan(hx[i]))
            ++nn;
        if (nn)
          std::fprintf(stderr, "TRACE step %d L%d dX nan %d/5120 maxabs %.3g\n",
                       step, L, nn, 0.0);
      }
      if (step == dump_step || step == dump_step2) {
        // pre-sized once; indexed write (no growth during run)
        if (dump_states.empty()) {
          dump_states.assign(64 * 5120, 0.0f);
          dump_states_b.assign(64 * 5120, 0.0f);
        }
        auto &vec = (step == dump_step) ? dump_states : dump_states_b;
        size_t base = (size_t)dump_layer_count[step == dump_step2 ? 1 : 0]++;
        q.memcpy(vec.data() + base * 5120, dX, (size_t)5120 * 4).wait();
      }
    } // layers

    // ---- after last prefill step and each gen step: logits + argmax ----
    if (step >= P - 1) {
      rmsnorm(dX, dFinNorm, dH, 1, 5120, true);
      xscales_of(dH, 5120);
      quantize(dH, 5120);
      gemvA(find("lm_head.weight"), 248320, 5120, dLogits);
      {
        auto ev = q.submit([&](sycl::handler &h) {
          sycl::local_accessor<float, 1> bv(256, h);
          sycl::local_accessor<int, 1> bi(256, h);
          h.parallel_for<KA1>(sycl::nd_range<1>(64 * 256, 256),
                              [=](sycl::nd_item<1> it) {
                                int lid = (int)it.get_local_id(0);
                                int gid = (int)it.get_group(0);
                                float best = -INFINITY;
                                int bidx = INT_MAX;
                                for (int i = gid * 256 + lid; i < 248320;
                                     i += 64 * 256) {
                                  float v = dLogits[i];
                                  if (v > best || (v == best && i < bidx)) {
                                    best = v;
                                    bidx = i;
                                  }
                                }
                                bv[lid] = best;
                                bi[lid] = bidx;
                                it.barrier(sycl::access::fence_space::local_space);
                                for (int st = 128; st > 0; st >>= 1) {
                                  if (lid < st) {
                                    if (bv[lid + st] > bv[lid] ||
                                        (bv[lid + st] == bv[lid] &&
                                         bi[lid + st] < bi[lid])) {
                                      bv[lid] = bv[lid + st];
                                      bi[lid] = bi[lid + st];
                                    }
                                  }
                                  it.barrier(sycl::access::fence_space::local_space);
                                }
                                if (lid == 0) {
                                  dPV[gid] = bv[0];
                                  dPI[gid] = bi[0];
                                }
                              });
        });
        ev.wait();
        auto ev2 = q.submit([&](sycl::handler &h) {
          sycl::local_accessor<float, 1> bv2(256, h);
          sycl::local_accessor<int, 1> bi2(256, h);
          h.parallel_for<KA2>(sycl::nd_range<1>(256, 256),
                              [=](sycl::nd_item<1> it) {
                                int lid = (int)it.get_local_id(0);
                                if (lid < 64) {
                                  bv2[lid] = dPV[lid];
                                  bi2[lid] = dPI[lid];
                                } else {
                                  bv2[lid] = -INFINITY;
                                  bi2[lid] = INT_MAX;
                                }
                                it.barrier(sycl::access::fence_space::local_space);
                                for (int st = 128; st > 0; st >>= 1) {
                                  if (lid < st) {
                                    if (bv2[lid + st] > bv2[lid] ||
                                        (bv2[lid + st] == bv2[lid] &&
                                         bi2[lid + st] < bi2[lid])) {
                                      bv2[lid] = bv2[lid + st];
                                      bi2[lid] = bi2[lid + st];
                                    }
                                  }
                                  it.barrier(sycl::access::fence_space::local_space);
                                }
                                if (lid == 0)
                                  dOutT[0] = bi2[0];
                              });
        });
        ev2.wait();
      }
      int tok = -1;
      if (tokShared)
        tok = dOutT[0]; // coherent after ev2.wait() above; no copy command
      else
        q.memcpy(&tok, dOutT, 4).wait();
      // T5.5: full-logits/top5 readback is diagnostic-only. AINFER_TOP5=0
      // selects the steady-state token-only return path (no 1 MB D2H, no host
      // partial_sort); default 1 preserves the certified reporting behavior.
      if (top5diag && step >= P - 1) {
        q.memcpy(logits_host.data(), dLogits, (size_t)248320 * 4).wait();
        for (int i = 0; i < 248320; ++i)
          logit_indices[i] = i;
        std::partial_sort(logit_indices.begin(), logit_indices.begin() + 5,
                          logit_indices.end(), [&](int a, int b) {
                            return logits_host[a] > logits_host[b];
                          });
        double mn = 0, sd = 0;
        for (float v : logits_host)
          mn += v;
        mn /= logits_host.size();
        for (float v : logits_host)
          sd += (v - mn) * (v - mn);
        sd = std::sqrt(sd / logits_host.size());
        std::printf("TOP5@s%d: ", step);
        for (int i = 0; i < 5; ++i)
          std::printf("%d(%.3f) ", logit_indices[i],
                      logits_host[logit_indices[i]]);
        std::printf("| mean %.3f std %.3f\n", mn, sd);
        tops5.emplace_back();
        tops5v.emplace_back();
        for (int i = 0; i < 5; ++i) {
          tops5.back()[i] = logit_indices[i];
          tops5v.back()[i] = logits_host[logit_indices[i]];
        }
        if (std::getenv("AINFER_STATS")) {
          std::vector<float> hx(5120), hh(5120);
          q.memcpy(hx.data(), dX, (size_t)5120 * 4).wait();
          q.memcpy(hh.data(), dH, (size_t)5120 * 4).wait();
          double mx = 0, mn = 0, hxmx = 0;
          int nanx = 0, nanh = 0;
          for (int i = 0; i < 5120; ++i) {
            float v = hx[i];
            if (std::isnan(v))
              ++nanx;
            else
              hxmx = std::max((double)hxmx, (double)std::fabs(v));
            float w = hh[i];
            if (std::isnan(w))
              ++nanh;
            else {
              mx = std::max(mx, (double)std::fabs(w));
              mn += w;
            }
          }
          std::fprintf(stderr,
                       "STATS@s%d dXmax %.3g nan %d | dHlogits-in max %.3g "
                       "mean %.4f nan %d\n",
                       step, hxmx, nanx, mx, mn / 5120, nanh);
        }
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

  std::string json = "{\"device\":\"B60\",\"prompt\":[";
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
  if (argc > 4) {
    o = std::fopen(pos.size() > 3 ? pos[3] : "/dev/stdout", "w");
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
  if (dump_step >= 0) {
    const char *dp = pos.size() > 5 ? pos[5] : "/tmp/dstates.bin";
    FILE *df = std::fopen(dp, "wb");
    if (df) {
      std::fwrite(dump_states.data(), 4, dump_states.size(), df);
      std::fclose(df);
      std::printf("dumped %zu floats -> %s\n", dump_states.size(), dp);
      {
        FILE *df2 = std::fopen("/tmp/dstates_b.bin", "wb");
        if (df2) {
          std::fwrite(dump_states_b.data(), 4, dump_states_b.size(), df2);
          std::fclose(df2);
          std::printf("dumped %zu floats -> /tmp/dstates_b.bin\n",
                      dump_states_b.size());
        }
      }
    }
  }
  return 0;
}
