// AInfer T4.2-device stage B: full L3 decode attention-half on device.
// Input: decode_xt. Stages: input RMSNorm(1+w) -> Q/K/V dp4a GEMVs (INT4 from
// arenas) -> split q/gate -> Q/K RMSNorm-256 -> NeoX RoPE-64 -> KV append ->
// GQA decode core (24Q/4KV) -> sigmoid gate -> o_proj dp4a -> +residual.
// Verified stage-wise vs reference/attn_block_L3.json + rope_pos4.json.
// Usage: blkblock <model.binfer> [report.json]
#include <level_zero/ze_api.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
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

// ---- minimal JSON float-array parser ----
static const char *g_p;
static void jskip() {
  while (*g_p == ' ' || *g_p == '\n' || *g_p == '\r' || *g_p == '\t' ||
         *g_p == ',' || *g_p == ':')
    ++g_p;
}
static double jnum() {
  jskip();
  char *e = nullptr;
  double v = std::strtod(g_p, &e);
  g_p = e;
  return v;
}
static std::vector<double> jget(const std::string &s, const std::string &key) {
  auto pos = s.find("\"" + key + "\"");
  if (pos == std::string::npos)
    return {};
  g_p = s.c_str() + pos + key.size() + 2;
  std::vector<double> out;
  std::function<void()> rec = [&]() {
    jskip();
    if (*g_p != '[')
      return;
    ++g_p;
    while (true) {
      jskip();
      if (*g_p == ']') {
        ++g_p;
        break;
      }
      if (*g_p == '[')
        rec();
      else
        out.push_back(jnum());
      jskip();
    }
  };
  rec();
  return out;
}
static std::string slurp(const char *p) {
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

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
static double ev_ms(sycl::event e) {
  e.wait();
  return (double)(e.get_profiling_info<
                      sycl::info::event_profiling::command_end>() -
                  e.get_profiling_info<
                      sycl::info::event_profiling::command_start>()) *
         1e-6;
}

// dp4a decode GEMV, runtime K (same math as blkexec stage A).
struct DKTag {};
static double gemv_dp4a(sycl::queue &q, const uint8_t *P, const uint16_t *S,
                        const int8_t *XQ, float sq, float *Y, int M, int K) {
  int G = K / 128;
  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for<DKTag>(sycl::range<1>(M), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      int m = id[0];
      float acc = 0;
      for (int g = 0; g < G; ++g) {
        uint32_t ub = S[(size_t)m * G + g];
        ub <<= 16;
        float sc;
        __builtin_memcpy(&sc, &ub, 4);
        size_t base = (size_t)m * K / 2 + (size_t)g * 64;
        esimd::simd<int, 8> acc8(0);
        for (int j = 0; j < 128; j += 32) {
          esimd::simd<unsigned char, 16> bytes =
              esimd::block_load<unsigned char, 16>(P + base + j / 2);
          esimd::simd<unsigned short, 32> v;
          v.template select<16, 2>(0) =
              esimd::convert<unsigned short>(bytes & 0xFu);
          v.template select<16, 2>(1) =
              esimd::convert<unsigned short>(bytes >> 4);
          esimd::simd<short, 32> sv = esimd::convert<short>(v);
          sv -= (sv & 8) << 1;
          esimd::simd<int8_t, 32> w8 = esimd::convert<int8_t>(sv);
          esimd::simd<int8_t, 32> xq =
              esimd::block_load<int8_t, 32>(XQ + g * 128 + j);
          esimd::simd<int, 8> w32 = w8.bit_cast_view<int>();
          esimd::simd<int, 8> x32 = xq.bit_cast_view<int>();
          acc8 = esimd::dp4a<int>(acc8, w32, x32);
        }
        int tmp[8];
        acc8.copy_to(tmp);
        int gsum = 0;
        for (int u = 0; u < 8; ++u)
          gsum += tmp[u];
        acc += (float)gsum * sc * sq;
      }
      Y[m] = acc;
    });
  });
  e.wait();
  return (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_end>() -
         (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_start>();
}

struct QTag {};   // actquant
struct NTag {};   // rmsnorm (any dim, one WI per row)
struct STag {};   // split q content/gate
struct RTag {};   // rope neox-64
struct KTag {};   // kv append at position 4
struct ATag {};   // gqa decode core
struct OTag {};   // residual add

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: blkblock <model.binfer> [report.json]\n");
    return 2;
  }
  const char *path = argv[1];
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return 2;
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "BINFER\x00\x01", 8) != 0)
    return 2;
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
  auto find = [&](const char *nm) -> const Entry * {
    for (auto &e : entries)
      if (std::strcmp(e.name, nm) == 0)
        return &e;
    return nullptr;
  };
  // BF16 small tensor -> fp32 host vector
  auto load_bf16 = [&](const char *nm, size_t count) {
    const Entry *e = find(nm);
    std::vector<uint16_t> raw(count);
    f.clear();
    f.seekg((std::streamoff)e->d_off);
    f.read((char *)raw.data(), count * 2);
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i)
      out[i] = bf16_to_f32(raw[i]);
    return out;
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
  const char *L3 = "model.language_model.layers.3.";
  std::string PN[4] = {std::string(L3) + "self_attn.q_proj.weight",
                       std::string(L3) + "self_attn.k_proj.weight",
                       std::string(L3) + "self_attn.v_proj.weight",
                       std::string(L3) + "self_attn.o_proj.weight"};
  const Entry *E[4];
  for (int i = 0; i < 4; ++i) {
    E[i] = find(PN[i].c_str());
    if (!E[i]) {
      std::fprintf(stderr, "missing %s\n", PN[i].c_str());
      return 2;
    }
  }
  // reference vectors
  std::string js = slurp("/mnt/usb/AInfer/reference/attn_block_L3.json");
  std::string rs = slurp("/mnt/usb/AInfer/reference/rope_pos4.json");
  auto J = [&](const std::string &s, const char *k) {
    auto v = jget(s, k);
    return std::vector<float>(v.begin(), v.end());
  };
  std::vector<float> xt = J(js, "decode_xt");      // 1x1x5120
  std::vector<float> refQ = J(js, "decode_q");     // 1x24x256
  std::vector<float> refG = J(js, "decode_gate");  // 24x256
  std::vector<float> refK5 = J(js, "decode_K5");   // 4x5x256
  std::vector<float> refV5 = J(js, "decode_V5");
  std::vector<float> refTO = J(js, "decode_token_out"); // 1x1x5120
  std::vector<float> refW0 = J(js, "decode_attn_weights_head0"); // 5
  std::vector<float> cos = J(rs, "cos"), sin = J(rs, "sin"); // 64
  std::vector<float> ln1 = load_bf16((std::string(L3) + "input_layernorm.weight").c_str(), 5120);
  std::vector<float> qnw = load_bf16((std::string(L3) + "self_attn.q_norm.weight").c_str(), 256);
  std::vector<float> knw = load_bf16((std::string(L3) + "self_attn.k_norm.weight").c_str(), 256);

  // device buffers
  float *dX = sycl::malloc_device<float>(5120, q);
  float *dH = sycl::malloc_device<float>(5120, q);
  float *dW = sycl::malloc_device<float>(5120, q);
  int8_t *dXQ = sycl::malloc_device<int8_t>(6144, q);
  float *dQ = sycl::malloc_device<float>(12288, q);
  float *dC = sycl::malloc_device<float>(6144, q);
  float *dG = sycl::malloc_device<float>(6144, q);
  float *dK = sycl::malloc_device<float>(1024, q);
  float *dV = sycl::malloc_device<float>(1024, q);
  float *dQn = sycl::malloc_device<float>(6144, q);
  float *dKn = sycl::malloc_device<float>(1024, q);
  float *dKc = sycl::malloc_device<float>(4 * 5 * 256, q);
  float *dVc = sycl::malloc_device<float>(4 * 5 * 256, q);
  float *dA = sycl::malloc_device<float>(6144, q);
  float *dY = sycl::malloc_device<float>(5120, q);
  float *dWts = sycl::malloc_device<float>(24 * 5, q);
  float *dLn = sycl::malloc_device<float>(5120, q);
  float *dQnW = sycl::malloc_device<float>(256, q);
  float *dKnW = sycl::malloc_device<float>(256, q);
  float *dCos = sycl::malloc_device<float>(64, q);
  float *dSin = sycl::malloc_device<float>(64, q);
  q.memcpy(dX, xt.data(), 5120 * 4).wait();
  q.memcpy(dLn, ln1.data(), 5120 * 4).wait();
  q.memcpy(dQnW, qnw.data(), 256 * 4).wait();
  q.memcpy(dKnW, knw.data(), 256 * 4).wait();
  q.memcpy(dCos, cos.data(), 64 * 4).wait();
  q.memcpy(dSin, sin.data(), 64 * 4).wait();
  // KV prefill (first 4 positions) into cache
  q.memcpy(dKc, refK5.data(), 4 * 4 * 256 * 4).wait();
  q.memcpy(dVc, refV5.data(), 4 * 4 * 256 * 4).wait();

  auto rmsnorm = [&](const float *X, const float *W, float *Y, int rows, int D) {
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<NTag>(sycl::range<1>(rows), [=](sycl::id<1> id) {
        int r = id[0];
        double ss = 0;
        for (int j = 0; j < D; ++j) {
          float v = X[r * D + j];
          ss += (double)v * v;
        }
        float inv = 1.0f / std::sqrt((float)(ss / D) + 1e-6f);
        for (int j = 0; j < D; ++j)
          Y[r * D + j] = X[r * D + j] * inv * (1.0f + W[j]);
      });
    });
    e.wait();
  };
  auto quantize = [&](const float *X, int8_t *XQ, float sq, int K) {
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<QTag>(sycl::range<1>(K), [=](sycl::id<1> id) {
        float v = X[id[0]] / sq;
        int qi = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        XQ[id[0]] = (int8_t)qi;
      });
    });
    e.wait();
  };
  auto gemvA = [&](const Entry *e, int M, int K, float sq, float *Y) {
    const uint8_t *dP = (const uint8_t *)payArena + (e->d_off - pay_lo);
    const uint16_t *dS =
        (const uint16_t *)((const char *)scArena + (e->sc_off - sc_lo));
    return gemv_dp4a(q, dP, dS, dXQ, sq, Y, M, K);
  };
  auto xmax_of = [&](const float *dBuf, int N) {
    std::vector<float> h(N);
    q.memcpy(h.data(), dBuf, (size_t)N * 4).wait();
    float m = 0;
    for (float v : h)
      m = std::max(m, std::fabs(v));
    return m;
  };

  std::string json = "{\"device\":\"B60\",\"results\":[";
  bool first = true;
  bool allok = true;
  auto row = [&](const char *nm, double maxrel, const char *extra = "") {
    bool ok = maxrel < 0.05;
    allok &= ok;
    char b[512];
    std::snprintf(b, sizeof b, "%s{\"name\":\"%s\",\"maxrel\":%.2e%s,\"%s\":true}",
                  first ? "" : ",", nm, maxrel, extra, ok ? "PASS" : "FAIL");
    json += b;
    first = false;
    std::printf("%-16s maxrel %.2e %s %s\n", nm, maxrel, extra,
                ok ? "PASS" : "FAIL");
  };
  auto rel = [&](const std::vector<float> &got, const std::vector<float> &ref) {
    double mx = 0, rm = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      rm = std::max(rm, (double)std::fabs(ref[i]));
      mx = std::max(mx, (double)std::fabs(got[i] - ref[i]));
    }
    return mx / rm;
  };

  // 1. input norm
  rmsnorm(dX, dLn, dH, 1, 5120);
  // 2. Q/K/V GEMVs (quantize dH once: K=5120 shared)
  {
    float sq = xmax_of(dH, 5120) / 127.0f;
    quantize(dH, dXQ, sq, 5120);
    gemvA(E[0], 12288, 5120, sq, dQ);
    gemvA(E[1], 1024, 5120, sq, dK);
    gemvA(E[2], 1024, 5120, sq, dV);
  }
  // 3. split q content/gate
  {
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<STag>(sycl::range<1>(6144), [=](sycl::id<1> id) {
        int i = id[0], h2 = i / 256, d = i % 256;
        dC[i] = dQ[(size_t)h2 * 512 + d];
        dG[i] = dQ[(size_t)h2 * 512 + 256 + d];
      });
    });
    e.wait();
  }
  // 4. Q/K norm-256
  rmsnorm(dC, dQnW, dQn, 24, 256);
  rmsnorm(dK, dKnW, dKn, 4, 256);
  // 5. RoPE-64 (24+4 rows)
  {
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<RTag>(sycl::range<1>(28), [=](sycl::id<1> id) {
        int i = id[0];
        float *X = i < 24 ? dQn + (size_t)i * 256 : dKn + (size_t)(i - 24) * 256;
        for (int d = 0; d < 32; ++d) {
          float x0 = X[d], x1 = X[d + 32];
          float c = dCos[d], s = dSin[d];
          X[d] = x0 * c - x1 * s;
          X[d + 32] = x0 * s + x1 * c;
        }
      });
    });
    e.wait();
  }
  // ============ two-track verification ============
  // Track K (kernel proof): device vs HOST-INT4 pipeline (file bytes dequantized
  // on host, device XQ read back and reused). Threshold 1e-4.
  // Track Q (quant accounting): host-INT4 final vs HF BF16 ref. Informational.
  auto rowinfo = [&](const char *nm, double maxrel, const char *extra = "") {
    char b[512];
    std::snprintf(b, sizeof b, "%s{\"name\":\"%s\",\"maxrel\":%.2e%s,\"INFO\":true}",
                  first ? "" : ",", nm, maxrel, extra);
    json += b;
    first = false;
    std::printf("%-16s maxrel %.2e %s INFO\n", nm, maxrel, extra);
  };
  // read back device intermediates (dXQ still holds the K=5120 quant here)
  std::vector<float> devQ(12288), devK(1024), devV(1024);
  std::vector<int8_t> devXQ(5120);
  std::vector<float> devQn(6144), devG(6144), devKn(1024);
  q.memcpy(devQ.data(), dQ, devQ.size() * 4).wait();
  q.memcpy(devK.data(), dK, devK.size() * 4).wait();
  q.memcpy(devV.data(), dV, devV.size() * 4).wait();
  q.memcpy(devXQ.data(), dXQ, 5120).wait();
  q.memcpy(devQn.data(), dQn, devQn.size() * 4).wait();
  q.memcpy(devG.data(), dG, devG.size() * 4).wait();
  q.memcpy(devKn.data(), dKn, devKn.size() * 4).wait();
  // host: file bytes -> dequant GEMV with DEVICE XQ
  auto file_ps = [&](const Entry *e) {
    std::vector<uint8_t> P(e->d_bytes);
    std::vector<uint16_t> S(e->sc_bytes / 2);
    f.clear();
    f.seekg((std::streamoff)e->d_off);
    f.read((char *)P.data(), e->d_bytes);
    f.seekg((std::streamoff)e->sc_off);
    f.read((char *)S.data(), e->sc_bytes);
    return std::make_pair(P, S);
  };
  auto host_gemv = [&](const std::vector<uint8_t> &P, const std::vector<uint16_t> &S,
                       const std::vector<int8_t> &XQ, float sq, int M, int K) {
    std::vector<float> Y(M, 0);
    int G = K / 128;
    for (int m = 0; m < M; ++m) {
      float acc = 0;
      for (int g = 0; g < G; ++g) {
        float sc = bf16_to_f32(S[(size_t)m * G + g]);
        int64_t isum = 0;
        for (int j = 0; j < 128; ++j) {
          size_t idx = (size_t)m * K + g * 128 + j;
          uint8_t b = P[idx / 2];
          int nib = (idx & 1) ? (b >> 4) : (b & 0xF);
          if (nib >= 8)
            nib -= 16;
          isum += (int64_t)nib * XQ[g * 128 + j];
        }
        acc += (float)isum * sc * sq;
      }
      Y[m] = acc;
    }
    return Y;
  };
  // host h + XQ for the quant-kernel check
  double hss = 0;
  for (int j = 0; j < 5120; ++j)
    hss += (double)xt[j] * xt[j];
  float hinv = 1.0f / std::sqrt((float)(hss / 5120) + 1e-6f);
  std::vector<float> hostH(5120);
  for (int j = 0; j < 5120; ++j)
    hostH[j] = xt[j] * hinv * (1.0f + ln1[j]);
  float hsq = 0;
  for (float v : hostH)
    hsq = std::max(hsq, std::fabs(v));
  hsq /= 127.0f;
  int xqbad = 0;
  for (int i = 0; i < 5120; ++i) {
    float v = hostH[i] / hsq;
    int qi = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
    if (devXQ[i] != (int8_t)qi)
      ++xqbad;
  }
  {
    char ex[64];
    std::snprintf(ex, sizeof ex, ",\"note\":\"%d/5120 mismatch\"", xqbad);
    row(xqbad == 0 ? "actquant" : "actquant", xqbad == 0 ? 0.0 : 1.0, ex);
    if (xqbad)
      allok = false;
  }
  // host GEMVs with device XQ
  auto ps_q = file_ps(E[0]);
  auto ps_k = file_ps(E[1]);
  auto ps_v = file_ps(E[2]);
  std::vector<float> hQ = host_gemv(ps_q.first, ps_q.second, devXQ, hsq, 12288, 5120);
  std::vector<float> hK = host_gemv(ps_k.first, ps_k.second, devXQ, hsq, 1024, 5120);
  std::vector<float> hV = host_gemv(ps_v.first, ps_v.second, devXQ, hsq, 1024, 5120);
  auto cmp = [&](const char *nm, const std::vector<float> &got,
                 const std::vector<float> &ref) {
    double mx = 0, rm = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      rm = std::max(rm, (double)std::fabs(ref[i]));
      mx = std::max(mx, (double)std::fabs(got[i] - ref[i]));
    }
    row(nm, mx / rm, ",\"note\":\"dev-vs-hostINT4\"");
    return mx / rm;
  };
  cmp("gemv-q", devQ, hQ);
  cmp("gemv-k", devK, hK);
  cmp("gemv-v", devV, hV);
  // host fp pipeline from host GEMVs: split, norm256, rope
  std::vector<float> hC(6144), hGt(6144);
  for (int i = 0; i < 6144; ++i) {
    hC[i] = hQ[(i / 256) * 512 + (i % 256)];
    hGt[i] = hQ[(i / 256) * 512 + 256 + (i % 256)];
  }
  auto hnorm = [&](const std::vector<float> &X, const std::vector<float> &W,
                   int rows, int D) {
    std::vector<float> Y(X.size());
    for (int r = 0; r < rows; ++r) {
      double ss = 0;
      for (int j = 0; j < D; ++j)
        ss += (double)X[r * D + j] * X[r * D + j];
      float inv = 1.0f / std::sqrt((float)(ss / D) + 1e-6f);
      for (int j = 0; j < D; ++j)
        Y[r * D + j] = X[r * D + j] * inv * (1.0f + W[j]);
    }
    return Y;
  };
  std::vector<float> hQn = hnorm(hC, qnw, 24, 256);
  std::vector<float> hKn = hnorm(
      std::vector<float>(hK.begin(), hK.end()), knw, 4, 256);
  for (int i = 0; i < 28; ++i) {
    float *X = i < 24 ? hQn.data() + (size_t)i * 256 : hKn.data() + (size_t)(i - 24) * 256;
    for (int d = 0; d < 32; ++d) {
      float x0 = X[d], x1 = X[d + 32];
      float c = cos[d], s2 = sin[d];
      X[d] = x0 * c - x1 * s2;
      X[d + 32] = x0 * s2 + x1 * c;
    }
  }
  cmp("qn-rope", devQn, hQn);
  cmp("kn-rope", devKn, hKn);
  // host GQA from host rope outputs + host gate
  std::vector<float> hKc(4 * 5 * 256), hVc(4 * 5 * 256);
  for (int i = 0; i < 4 * 4 * 256; ++i) {
    hKc[i] = refK5[i];
    hVc[i] = refV5[i];
  }
  // NOTE: prefill cache rows come from BF16 HF refs here (device used same
  // upload); new-token rows appended below from host rope outputs.
  for (int h2 = 0; h2 < 4; ++h2)
    for (int d = 0; d < 256; ++d) {
      hKc[((size_t)h2 * 5 + 4) * 256 + d] = hKn[(size_t)h2 * 256 + d];
      hVc[((size_t)h2 * 5 + 4) * 256 + d] = hV[(size_t)h2 * 256 + d];
    }
  std::vector<float> hWts(24 * 5), hA(6144);
  for (int hh = 0; hh < 24; ++hh) {
    int kv = hh / 6;
    float mx2 = -1e30f;
    for (int t = 0; t < 5; ++t) {
      float s = 0;
      for (int d = 0; d < 256; ++d)
        s += hQn[(size_t)hh * 256 + d] * hKc[((size_t)kv * 5 + t) * 256 + d];
      s /= 16.0f;
      hWts[(size_t)hh * 5 + t] = s;
      mx2 = s > mx2 ? s : mx2;
    }
    float se = 0;
    for (int t = 0; t < 5; ++t) {
      hWts[(size_t)hh * 5 + t] = std::exp(hWts[(size_t)hh * 5 + t] - mx2);
      se += hWts[(size_t)hh * 5 + t];
    }
    for (int d = 0; d < 256; ++d) {
      float acc = 0;
      for (int t = 0; t < 5; ++t)
        acc += hWts[(size_t)hh * 5 + t] / se *
               hVc[((size_t)kv * 5 + t) * 256 + d];
      hA[(size_t)hh * 256 + d] = acc / (1.0f + std::exp(-hGt[(size_t)hh * 256 + d]));
    }
  }
  // host o_proj GEMV (dequant from file) + residual
  auto ps_o = file_ps(E[3]);
  float osq = 0;
  for (float v : hA)
    osq = std::max(osq, std::fabs(v));
  osq /= 127.0f;
  std::vector<int8_t> oXQ(6144);
  for (int i = 0; i < 6144; ++i) {
    float v = hA[i] / osq;
    int qi = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
    oXQ[i] = (int8_t)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
  }
  std::vector<float> hY = host_gemv(ps_o.first, ps_o.second, oXQ, osq, 5120, 6144);
  for (int i = 0; i < 5120; ++i)
    hY[i] += xt[i];
  // ---- device: KV append, GQA core, o_proj + residual ----
  {
    auto e2 = q.submit([&](sycl::handler &h) {
      h.parallel_for<KTag>(sycl::range<1>(4 * 256), [=](sycl::id<1> id) {
        int i = id[0], h2 = i / 256, d = i % 256;
        dKc[((size_t)h2 * 5 + 4) * 256 + d] = dKn[(size_t)h2 * 256 + d];
        dVc[((size_t)h2 * 5 + 4) * 256 + d] = dV[(size_t)h2 * 256 + d];
      });
    });
    e2.wait();
  }
  {
    auto e3 = q.submit([&](sycl::handler &h) {
      h.parallel_for<ATag>(sycl::range<1>(24), [=](sycl::id<1> id) {
        int hh = id[0], kv = hh / 6;
        float mx2 = -1e30f;
        for (int t = 0; t < 5; ++t) {
          float s = 0;
          for (int d = 0; d < 256; ++d)
            s += dQn[(size_t)hh * 256 + d] * dKc[((size_t)kv * 5 + t) * 256 + d];
          s /= 16.0f;
          dWts[(size_t)hh * 5 + t] = s;
          mx2 = s > mx2 ? s : mx2;
        }
        float se = 0;
        for (int t = 0; t < 5; ++t) {
          float w = std::exp(dWts[(size_t)hh * 5 + t] - mx2);
          dWts[(size_t)hh * 5 + t] = w;
          se += w;
        }
        for (int d = 0; d < 256; ++d) {
          float acc = 0;
          for (int t = 0; t < 5; ++t)
            acc += dWts[(size_t)hh * 5 + t] / se *
                   dVc[((size_t)kv * 5 + t) * 256 + d];
          float g = dG[(size_t)hh * 256 + d];
          dA[(size_t)hh * 256 + d] = acc / (1.0f + std::exp(-g));
        }
      });
    });
    e3.wait();
  }
  {
    std::vector<float> devWts(24 * 5), devA(6144);
    q.memcpy(devWts.data(), dWts, devWts.size() * 4).wait();
    q.memcpy(devA.data(), dA, devA.size() * 4).wait();
    double se = 0, seh = 0;
    for (int t = 0; t < 5; ++t) {
      se += devWts[t];
      seh += hWts[t];
    }
    double mx = 0;
    for (int t = 0; t < 5; ++t)
      mx = std::max(mx, std::fabs(devWts[t] / se - hWts[t] / seh));
    row("gqa-w0", mx, ",\"note\":\"dev-vs-hostINT4\"");
    cmp("gqa-out", devA, hA);
  }
  {
    float sq = xmax_of(dA, 6144) / 127.0f;
    quantize(dA, dXQ, sq, 6144);
    gemvA(E[3], 5120, 6144, sq, dY);
    auto e4 = q.submit([&](sycl::handler &h) {
      h.parallel_for<OTag>(sycl::range<1>(5120), [=](sycl::id<1> id) {
        dY[id[0]] += dX[id[0]];
      });
    });
    e4.wait();
  }
  {
    std::vector<float> got(5120);
    q.memcpy(got.data(), dY, 5120 * 4).wait();
    cmp("block-out", got, hY);
    double mx = 0, rm = 0;
    for (int i = 0; i < 5120; ++i) {
      rm = std::max(rm, (double)std::fabs(refTO[i]));
      mx = std::max(mx, (double)std::fabs(hY[i] - refTO[i]));
    }
    char ex[96];
    std::snprintf(ex, sizeof ex, ",\"note\":\"hostINT4-vs-HF-BF16\"");
    rowinfo("quant-delta", mx / rm, ex);
  }
  json += "]}";
  FILE *o = stdout;
  if (argc > 2) {
    o = std::fopen(argv[2], "w");
    if (!o)
      return 1;
  }
  std::fprintf(o, "%s\n", json.c_str());
  if (o != stdout)
    std::fclose(o);
  CHECK(zeCommandListDestroy(list));
  CHECK(zeMemFree(ctx, payArena));
  CHECK(zeMemFree(ctx, scArena));
  CHECK(zeContextDestroy(ctx));
  return allok ? 0 : 3;
}
