// AInfer T4.2-device stage A: real-weight INT4 GEMV from static L0 arenas.
// Parses .binfer, allocates the 2 static arenas, streams the full upload,
// then runs the proven dp4a-i8 decode kernel READING P/S DIRECTLY FROM THE
// ARENAS at entry offsets (no synthetic weights anywhere in the path).
// Verified vs host fp32 reference dequantized from the same file bytes.
// Shapes: L3 q_proj [12288x5120], L0 gate_proj [17408x5120].
// Usage: blkexec <model.binfer> [report.json]
#include <level_zero/ze_api.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

static uint32_t crc32_update(uint32_t crc, const uint8_t *p, size_t n) {
  static uint32_t tab[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k)
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      tab[i] = c;
    }
    init = true;
  }
  crc ^= 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i)
    crc = tab[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
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

// dp4a decode GEMV, runtime K (cf. gemv_t32 pathC which templates K).
// P/S are arena pointers already advanced to the entry's offsets.
struct DKTag {};
struct DQTag {};
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

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: blkexec <model.binfer> [report.json]\n");
    return 2;
  }
  const char *path = argv[1];
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return 2;
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "BINFER\x00\x01", 8) != 0) {
    std::fprintf(stderr, "bad magic\n");
    return 2;
  }
  uint32_t ver = rd32(f), flags = rd32(f);
  uint64_t n;
  f.read((char *)&n, 8);
  uint64_t table_off = rd64(f);
  (void)ver;
  (void)flags;
  f.seekg((std::streamoff)(table_off + 4 * 32));
  f.seekg(4 + 8 + 8, std::ios::cur);
  uint32_t dir_crc;
  f.read((char *)&dir_crc, 4);
  f.seekg((std::streamoff)table_off);
  uint64_t dir_off = 0, dir_bytes = 0;
  for (int i = 0; i < 5; ++i) {
    uint32_t sid = rd32(f);
    uint64_t off = rd64(f), nb = rd64(f), c = rd32(f);
    (void)c;
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
    f.seekg(1 + 7 + 64 + 1 + 1 + 2 + 4, std::ios::cur); // ndim..group
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
  uint64_t scSize = (sc_hi > sc_lo) ? (sc_hi - sc_lo) : 0;
  if (scSize)
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
  if (scSize)
    up(sc_lo, scArena, sc_lo, scSize);

  sycl::device sdev = pick_b60();
  sycl::queue q(sdev, {sycl::property::queue::enable_profiling()});

  struct Shape {
    const char *name;
    int M, K;
  };
  Shape shapes[] = {
      {"model.language_model.layers.3.self_attn.q_proj.weight", 12288, 5120},
      {"model.language_model.layers.0.mlp.gate_proj.weight", 17408, 5120},
  };
  std::string json = "{\"device\":\"B60\",\"results\":[";
  bool first = true, allok = true;
  // deterministic fp activation shared by device + host reference
  std::vector<float> X(5120);
  {
    uint64_t s = 0xbeef;
    for (auto &v : X) {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      v = (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
    }
  }
  float xmax = 0;
  for (float v : X)
    xmax = std::max(xmax, std::fabs(v));
  float sq = xmax / 127.0f;
  std::vector<int8_t> XQ(5120);
  for (int i = 0; i < 5120; ++i) {
    int qi = (int)(X[i] / sq >= 0 ? X[i] / sq + 0.5f : X[i] / sq - 0.5f);
    XQ[i] = (int8_t)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
  }
  float *dX = sycl::malloc_device<float>(5120, q);
  int8_t *dXQ = sycl::malloc_device<int8_t>(5120, q);
  q.memcpy(dX, X.data(), 5120 * 4).wait();
  // device-side activation quantize (same kernel family as gemv_t32)
  {
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<DQTag>(sycl::range<1>(5120), [=](sycl::id<1> id) {
        float v = dX[id[0]] / sq;
        int qi = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        dXQ[id[0]] = (int8_t)qi;
      });
    });
    e.wait();
  }
  // verify device XQ matches host XQ
  {
    std::vector<int8_t> back(5120);
    q.memcpy(back.data(), dXQ, 5120).wait();
    int bad = 0;
    for (int i = 0; i < 5120; ++i)
      if (back[i] != XQ[i])
        ++bad;
    if (bad) {
      std::fprintf(stderr, "XQ mismatch %d\n", bad);
      return 3;
    }
  }

  for (auto sh : shapes) {
    const Entry *e = find(sh.name);
    if (!e) {
      std::fprintf(stderr, "missing %s\n", sh.name);
      return 2;
    }
    const uint8_t *dP =
        (const uint8_t *)payArena + (e->d_off - pay_lo);
    const uint16_t *dS =
        (const uint16_t *)((const char *)scArena + (e->sc_off - sc_lo));
    float *dY = sycl::malloc_device<float>(sh.M, q);
    double best = 1e18;
    for (int r = 0; r < 5; ++r) {
      double ns = gemv_dp4a(q, dP, dS, dXQ, sq, dY, sh.M, sh.K);
      if (ns < best)
        best = ns;
    }
    std::vector<float> got(sh.M);
    q.memcpy(got.data(), dY, (size_t)sh.M * 4).wait();
    // host fp32 reference dequantized from the same file bytes
    f.clear();
    std::vector<uint8_t> P(e->d_bytes);
    f.seekg((std::streamoff)e->d_off);
    f.read((char *)P.data(), e->d_bytes);
    std::vector<uint16_t> S(e->sc_bytes / 2);
    f.seekg((std::streamoff)e->sc_off);
    f.read((char *)S.data(), e->sc_bytes);
    double maxd = 0, meand = 0, refmax = 0;
    int G = sh.K / 128;
    // reference computed per-row in fp64, compared streaming (no full matrix)
    for (int m = 0; m < sh.M; ++m) {
      double acc = 0;
      for (int g = 0; g < G; ++g) {
        float sc = bf16_to_f32(S[(size_t)m * G + g]);
        for (int j = 0; j < 128; ++j) {
          size_t idx = (size_t)m * sh.K + g * 128 + j;
          uint8_t b = P[idx / 2];
          int nib = (idx & 1) ? (b >> 4) : (b & 0xF);
          if (nib >= 8)
            nib -= 16;
          acc += (double)nib * sc * X[g * 128 + j];
        }
      }
      // int8-activation reference: XQ*sq instead of X
      double accq = 0;
      for (int g = 0; g < G; ++g) {
        float sc = bf16_to_f32(S[(size_t)m * G + g]);
        for (int j = 0; j < 128; ++j) {
          size_t idx = (size_t)m * sh.K + g * 128 + j;
          uint8_t b = P[idx / 2];
          int nib = (idx & 1) ? (b >> 4) : (b & 0xF);
          if (nib >= 8)
            nib -= 16;
          accq += (double)nib * sc * (XQ[g * 128 + j] * sq);
        }
      }
      (void)acc;
      refmax = std::max(refmax, std::fabs(accq));
      maxd = std::max(maxd, std::fabs((double)got[m] - accq));
      meand += std::fabs((double)got[m] - accq);
    }
    meand /= sh.M;
    size_t traffic = e->d_bytes + e->sc_bytes + 5120 + (size_t)sh.M * 4;
    double gbs = (double)traffic / best * 1e9 / 1e9;
    bool ok = (maxd / refmax) < 1e-4;
    allok &= ok;
    char row[512];
    std::snprintf(row, sizeof row,
                  "%s{\"tensor\":\"%s\",\"ms\":%.3f,\"gbs\":%.1f,\"maxrel\":%."
                  "2e,\"%s\":true}",
                  first ? "" : ",", sh.name, best * 1e-6, gbs, maxd / refmax,
                  ok ? "PASS" : "FAIL");
    json += row;
    first = false;
    std::printf("%-58s %7.3f ms %6.1f GB/s maxrel %.2e %s\n", sh.name,
                best * 1e-6, gbs, maxd / refmax, ok ? "PASS" : "FAIL");
    sycl::free(dY, q);
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
  if (scArena)
    CHECK(zeMemFree(ctx, scArena));
  CHECK(zeContextDestroy(ctx));
  sycl::free(dX, q);
  sycl::free(dXQ, q);
  return allok ? 0 : 3;
}
