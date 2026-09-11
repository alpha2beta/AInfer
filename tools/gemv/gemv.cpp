// AInfer INT4 decode GEMV shootout (T3.2) at real Qwen3.8 shapes.
// Paths: A scalar-fp32 | B vec-fp16 ESIMD+subgroup | C int8-dp4a |
//        D fp16-DPAS broadcast-8rows | E int8-DPAS broadcast-8rows.
// Weights use the real .binfer layout-0 packing (nibble pairs, g128 BF16
// scales). Batch-1 GEMV: y = Wx. Effective BW counts packed+scales+act+out.
// Usage: gemv_t32 [report.json]
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace mx = sycl::ext::oneapi::experimental::matrix;

static sycl::device pick_b60() {
  for (auto p : sycl::platform::get_platforms())
    for (auto d : p.get_devices(sycl::info::device_type::gpu))
      if (d.get_info<sycl::info::device::name>().find("B60") !=
          std::string::npos)
        return d;
  std::fprintf(stderr, "FATAL: no B60\n");
  std::exit(1);
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

// ---- host: pack fp32 weights (row-major MxK) to layout-0 + BF16 scales ----
static void pack_int4(const std::vector<float> &w, int M, int K,
                      std::vector<uint8_t> &packed,
                      std::vector<uint16_t> &scales) {
  int G = K / 128;
  packed.assign((size_t)M * K / 2, 0);
  scales.assign((size_t)M * G, 0);
  for (int m = 0; m < M; ++m) {
    for (int g = 0; g < G; ++g) {
      float amax = 0;
      for (int j = 0; j < 128; ++j)
        amax = std::max(amax, std::fabs(w[(size_t)m * K + g * 128 + j]));
      float sc = amax == 0 ? 1.0f : amax / 7.0f;
      scales[(size_t)m * G + g] = f32_to_bf16(sc);
      for (int j = 0; j < 128; ++j) {
        float v = w[(size_t)m * K + g * 128 + j] / sc;
        int q = (int)std::lrint(v);
        q = q < -8 ? -8 : (q > 7 ? 7 : q);
        size_t idx = (size_t)m * K + g * 128 + j;
        if (j & 1)
          packed[idx / 2] |= (uint8_t)((q & 0xF) << 4);
        else
          packed[idx / 2] = (uint8_t)(q & 0xF);
      }
    }
  }
}

// ================= Path A: scalar fp32, 1 row per WI =================
template <int K> struct ATag {};
template <int K> double pathA(sycl::queue &q, const uint8_t *P,
                              const uint16_t *S, const float *X, float *Y,
                              int M) {
  constexpr int G = K / 128;
  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for<ATag<K>>(sycl::range<1>(M), [=](sycl::id<1> id) {
      int m = id[0];
      float acc = 0;
      for (int g = 0; g < G; ++g) {
        uint32_t ub = S[(size_t)m * G + g];
        ub <<= 16;
        float sc;
        std::memcpy(&sc, &ub, 4);
        size_t base = (size_t)m * K / 2 + (size_t)g * 64;
        for (int j = 0; j < 128; ++j) {
          uint8_t b = P[base + j / 2];
          int nib = (j & 1) ? (b >> 4) : (b & 0xF);
          if (nib >= 8)
            nib -= 16;
          acc += (float)nib * sc * X[g * 128 + j];
        }
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

// ============ Path B: one row per WI, ESIMD vector unpack, scalar fold ====
// (esimd::reduce<float> miscompiles to a silent no-store on this toolchain;
// subgroup+SLM variants also fail to store -- toolchain findings logged.
// Manual fold is slower but provably correct.)
template <int K> struct BTag {};
template <int K> double pathB(sycl::queue &q, const uint8_t *P,
                              const uint16_t *S, const float *X, float *Y,
                              int M) {
  constexpr int G = K / 128;
  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for<BTag<K>>(sycl::range<1>(M),
                            [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
                              int m = id[0];
                              float acc = 0;
                              for (int g = 0; g < G; ++g) {
                                uint32_t ub = S[(size_t)m * G + g];
                                ub <<= 16;
                                float sc;
                                __builtin_memcpy(&sc, &ub, 4);
                                size_t base =
                                    (size_t)m * K / 2 + (size_t)g * 64;
                                for (int j = 0; j < 128; j += 32) {
                                  esimd::simd<unsigned char, 16> bytes =
                                      esimd::block_load<unsigned char, 16>(
                                          P + base + j / 2);
                                  esimd::simd<unsigned short, 32> v;
                                  v.template select<16, 2>(0) =
                                      esimd::convert<unsigned short>(bytes &
                                                                     0xFu);
                                  v.template select<16, 2>(1) =
                                      esimd::convert<unsigned short>(bytes >>
                                                                     4);
                                  esimd::simd<short, 32> s =
                                      esimd::convert<short>(v);
                                  s -= (s & 8) << 1;
                                  esimd::simd<float, 32> xv =
                                      esimd::block_load<float, 32>(
                                          X + g * 128 + j);
                                  esimd::simd<float, 32> prod =
                                      esimd::convert<float>(s) * (sc * xv);
                                  float tmp[32];
                                  prod.copy_to(tmp);
                                  for (int u = 0; u < 32; ++u)
                                    acc += tmp[u];
                                }
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

// ============ Path C: one row per WI, dp4a int8 dots, per-group rescale ====
// (Root-caused: esimd::reduce<int/float> miscompiles (zeros/no-store); dp4a
// itself is fine per dp4a_probe. Folds use copy_to + scalar loops.)
template <int K> struct CTag {};
template <int K> double pathC(sycl::queue &q, const uint8_t *P,
                              const uint16_t *S, const int8_t *XQ, float sq,
                              float *Y, int M) {
  constexpr int G = K / 128;
  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for<CTag<K>>(sycl::range<1>(M),
                            [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
                              int m = id[0];
                              float acc = 0;
                              for (int g = 0; g < G; ++g) {
                                uint32_t ub = S[(size_t)m * G + g];
                                ub <<= 16;
                                float sc;
                                __builtin_memcpy(&sc, &ub, 4);
                                size_t base =
                                    (size_t)m * K / 2 + (size_t)g * 64;
                                esimd::simd<int, 8> acc8(0);
                                for (int j = 0; j < 128; j += 32) {
                                  esimd::simd<unsigned char, 16> bytes =
                                      esimd::block_load<unsigned char, 16>(
                                          P + base + j / 2);
                                  esimd::simd<unsigned short, 32> v;
                                  v.template select<16, 2>(0) =
                                      esimd::convert<unsigned short>(bytes &
                                                                     0xFu);
                                  v.template select<16, 2>(1) =
                                      esimd::convert<unsigned short>(bytes >>
                                                                     4);
                                  esimd::simd<short, 32> sv =
                                      esimd::convert<short>(v);
                                  sv -= (sv & 8) << 1;
                                  esimd::simd<int8_t, 32> w8 =
                                      esimd::convert<int8_t>(sv);
                                  esimd::simd<int8_t, 32> xq =
                                      esimd::block_load<int8_t, 32>(XQ + g * 128 +
                                                                    j);
                                  esimd::simd<int, 8> w32 =
                                      w8.bit_cast_view<int>();
                                  esimd::simd<int, 8> x32 =
                                      xq.bit_cast_view<int>();
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

// ============ Path C4: UR=4 rows per WI, dp4a, shared x loads ============
// Same math as C; each WI owns 4 consecutive rows to amortize x traffic and
// launch overhead on short-row shapes.
template <int K> struct C4Tag {};
template <int K> double pathC4(sycl::queue &q, const uint8_t *P,
                               const uint16_t *S, const int8_t *XQ, float sq,
                               float *Y, int M) {
  constexpr int G = K / 128;
  constexpr int UR = 4;
  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for<C4Tag<K>>(sycl::range<1>(M / UR),
                             [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
                               int m0 = id[0] * UR;
                               float acc[UR] = {0, 0, 0, 0};
                               for (int g = 0; g < G; ++g) {
                                 float sc[UR];
                                 size_t base[UR];
                                 for (int r = 0; r < UR; ++r) {
                                   int m = m0 + r;
                                   uint32_t ub = S[(size_t)m * G + g];
                                   ub <<= 16;
                                   __builtin_memcpy(&sc[r], &ub, 4);
                                   base[r] =
                                       (size_t)m * K / 2 + (size_t)g * 64;
                                 }
                                 esimd::simd<int, 8> acc8[UR] = {};
                                 for (int j = 0; j < 128; j += 32) {
                                   esimd::simd<int8_t, 32> xq =
                                       esimd::block_load<int8_t, 32>(XQ + g * 128 +
                                                                     j);
                                   esimd::simd<int, 8> x32 =
                                       xq.bit_cast_view<int>();
                                   for (int r = 0; r < UR; ++r) {
                                     esimd::simd<unsigned char, 16> bytes =
                                         esimd::block_load<unsigned char, 16>(
                                             P + base[r] + j / 2);
                                     esimd::simd<unsigned short, 32> v;
                                     v.template select<16, 2>(0) =
                                         esimd::convert<unsigned short>(bytes &
                                                                        0xFu);
                                     v.template select<16, 2>(1) =
                                         esimd::convert<unsigned short>(bytes >>
                                                                        4);
                                     esimd::simd<short, 32> sv =
                                         esimd::convert<short>(v);
                                     sv -= (sv & 8) << 1;
                                     esimd::simd<int8_t, 32> w8 =
                                         esimd::convert<int8_t>(sv);
                                     esimd::simd<int, 8> w32 =
                                         w8.bit_cast_view<int>();
                                     acc8[r] = esimd::dp4a<int>(acc8[r], w32,
                                                                x32);
                                   }
                                 }
                                 int tmp[8];
                                 for (int r = 0; r < UR; ++r) {
                                   acc8[r].copy_to(tmp);
                                   int gsum = 0;
                                   for (int u = 0; u < 8; ++u)
                                     gsum += tmp[u];
                                   acc[r] += (float)gsum * sc[r] * sq;
                                 }
                               }
                               for (int r = 0; r < UR; ++r)
                                 Y[m0 + r] = acc[r];
                             });
  });
  e.wait();
  return (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_end>() -
         (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_start>();
}

// ============ Path D: f16 DPAS, 8 rows per subgroup, broadcast B ============
// Per 128-group (8 K-tiles of 16): dequant-fill SLM A(8x16 f16) + B(16x16 f16
// broadcast x), DPAS mad into joint acc, then fold col 0 with row scales.
template <int K> struct DTag {};
template <int K> double pathD(sycl::queue &q, const uint8_t *P,
                              const uint16_t *S, const float *X, float *Y,
                              int M) {
  constexpr int TR = 8, TC = 16, TN = 16, G = K / 128, TPB = 128 / TC;
  int nBlocks = M / TR;
  auto e = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint8_t, 1> scratch(sycl::range<1>(2048), h);
    h.parallel_for<DTag<K>>(
        sycl::nd_range<1>({(size_t)nBlocks * 16}, {16}),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          auto sg = it.get_sub_group();
          int blk = it.get_group(0);
          int lid = (int)sg.get_local_id()[0];
          uint8_t *base = scratch.get_pointer();
          sycl::half *sA = (sycl::half *)base;
          sycl::half *sB = (sycl::half *)(base + 256);
          float *sC = (float *)(base + 256 + 512);
          float ym = 0; // lane lid<8 owns row blk*8+lid
          mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TR,
                           TN, mx::layout::dynamic>
              acc;
          for (int g = 0; g < G; ++g) {
            mx::joint_matrix_fill(sg, acc, 0.0f);
            for (int t = 0; t < TPB; ++t) {
              int kt = g * 128 + t * TC;
              for (int u = 0; u < 8; ++u) { // A: 128 elems, 8 per lane
                int idx = lid * 8 + u, r = idx / TC, c = idx % TC;
                int row = blk * TR + r;
                size_t li = (size_t)row * K + kt + c;
                uint8_t b = P[li / 2];
                int nib = (li & 1) ? (b >> 4) : (b & 0xF);
                if (nib >= 8)
                  nib -= 16;
                uint32_t ub = S[(size_t)row * G + g];
                ub <<= 16;
                float sc;
                __builtin_memcpy(&sc, &ub, 4);
                sA[r * TC + c] = sycl::half((float)nib * sc);
              }
              for (int u = 0; u < 16; ++u) { // B: 256 elems, 16 per lane
                int idx = lid * 16 + u, i = idx / TN, n2 = idx % TN;
                sB[i * TN + n2] = sycl::half(X[kt + i]);
              }
              sg.barrier();
              mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TR,
                               TC, mx::layout::row_major>
                  ta;
              mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TC,
                               TN, mx::layout::row_major>
                  tb;
              sycl::multi_ptr<sycl::half,
                              sycl::access::address_space::local_space>
                  mpA(sA), mpB(sB);
              mx::joint_matrix_load(sg, ta, mpA, TC);
              mx::joint_matrix_load(sg, tb, mpB, TN);
              mx::joint_matrix_mad(sg, acc, ta, tb, acc);
              sg.barrier();
            }
            sycl::multi_ptr<float, sycl::access::address_space::local_space> mpC(
                sC);
            mx::joint_matrix_store(sg, acc, mpC, TN, mx::layout::row_major);
            sg.barrier();
            if (lid < TR)
              ym += sC[lid * TN + 0];
          }
          if (lid < TR)
            Y[blk * TR + lid] = ym;
        });
  });
  e.wait();
  return (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_end>() -
         (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_start>();
}

// ============ Path E: int8 DPAS broadcast (x pre-quantized, sq host) ============
template <int K> struct ETag {};
template <int K> double pathE(sycl::queue &q, const uint8_t *P,
                              const uint16_t *S, const int8_t *XQ, float sq,
                              float *Y, int M) {
  constexpr int TR = 8, TC = 32, TN = 16, G = K / 128, TPB = 128 / TC;
  int nBlocks = M / TR;
  auto e = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint8_t, 1> scratch(sycl::range<1>(2048), h);
    h.parallel_for<ETag<K>>(
        sycl::nd_range<1>({(size_t)nBlocks * 16}, {16}),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          auto sg = it.get_sub_group();
          int blk = it.get_group(0);
          int lid = (int)sg.get_local_id()[0];
          uint8_t *base = scratch.get_pointer();
          int8_t *sA = (int8_t *)base;
          int8_t *sB = (int8_t *)(base + 256);
          int32_t *sC = (int32_t *)(base + 256 + 512);
          float ym = 0;
          mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, TR,
                           TN, mx::layout::dynamic>
              acc;
          for (int g = 0; g < G; ++g) {
            mx::joint_matrix_fill(sg, acc, (int32_t)0);
            for (int t = 0; t < TPB; ++t) {
              int kt = g * 128 + t * TC;
              for (int u = 0; u < 16; ++u) { // A: 256 elems, 16 per lane
                int idx = lid * 16 + u, r = idx / TC, c = idx % TC;
                int row = blk * TR + r;
                size_t li = (size_t)row * K + kt + c;
                uint8_t b = P[li / 2];
                int nib = (li & 1) ? (b >> 4) : (b & 0xF);
                if (nib >= 8)
                  nib -= 16;
                sA[r * TC + c] = (int8_t)nib;
              }
              for (int u = 0; u < 32; ++u) { // B: 512 elems, 32 per lane
                int idx = lid * 32 + u, i = idx / TN, n2 = idx % TN;
                sB[i * TN + n2] = XQ[kt + i];
              }
              sg.barrier();
              mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, TR, TC,
                               mx::layout::row_major>
                  ta;
              mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, TC, TN,
                               mx::layout::row_major>
                  tb;
              sycl::multi_ptr<int8_t, sycl::access::address_space::local_space>
                  mpA(sA), mpB(sB);
              mx::joint_matrix_load(sg, ta, mpA, TC);
              mx::joint_matrix_load(sg, tb, mpB, TN);
              mx::joint_matrix_mad(sg, acc, ta, tb, acc);
              sg.barrier();
            }
            sycl::multi_ptr<int32_t,
                            sycl::access::address_space::local_space>
                mpC(sC);
            mx::joint_matrix_store(sg, acc, mpC, TN, mx::layout::row_major);
            sg.barrier();
            if (lid < TR) {
              int row = blk * TR + lid;
              uint32_t ub = S[(size_t)row * G + g];
              ub <<= 16;
              float sc;
              __builtin_memcpy(&sc, &ub, 4);
              ym += (float)sC[lid * TN + 0] * sc * sq;
            }
          }
          if (lid < TR)
            Y[blk * TR + lid] = ym;
        });
  });
  e.wait();
  return (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_end>() -
         (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_start>();
}

// ---- activation quantize (int8, per-tensor; sq from host, staged for T4) ----
struct QTag {};
static double actquant(sycl::queue &q, const float *X, int8_t *XQ, float sq,
                       int K) {
  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for<QTag>(sycl::range<1>(K), [=](sycl::id<1> id) {
      float v = X[id[0]] / sq;
      int qi = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
      qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
      XQ[id[0]] = (int8_t)qi;
    });
  });
  e.wait();
  return (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_end>() -
         (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_start>();
}

template <int K>
static void host_ref(const std::vector<uint8_t> &packed,
                     const std::vector<uint16_t> &scales,
                     const std::vector<float> &X, std::vector<float> &Y,
                     int M) {
  // Reference depacks the ACTUAL packed bytes (not a re-quantization), so
  // relerr isolates kernel error from quantization noise.
  for (int m = 0; m < M; ++m) {
    double acc = 0;
    for (int g = 0; g < K / 128; ++g) {
      float sc = bf16_to_f32(scales[(size_t)m * (K / 128) + g]);
      for (int j = 0; j < 128; ++j) {
        size_t idx = (size_t)m * K + g * 128 + j;
        uint8_t b = packed[idx / 2];
        int nib = (idx & 1) ? (b >> 4) : (b & 0xF);
        if (nib >= 8)
          nib -= 16;
        acc += (double)((float)nib * sc * X[g * 128 + j]);
      }
    }
    Y[m] = (float)acc;
  }
}

template <int M, int K>
static void run_shape(sycl::queue &q, std::string &json, bool &first) {
  // PRNG weights (init-scale) + activation
  std::vector<float> W((size_t)M * K), X(K), ref(M);
  uint64_t s = 0x9e3779b97f4a7c15ull + M * 1315423911u + K;
  for (size_t i = 0; i < W.size(); ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    W[i] = (float)((int)((s >> 33) & 0xFFFF) - 32768) * (0.02f / 32768.0f);
  }
  for (int i = 0; i < K; ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    X[i] = (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  }
  std::vector<uint8_t> packed;
  std::vector<uint16_t> scales;
  pack_int4(W, M, K, packed, scales);
  size_t packedB = packed.size(), scaleB = scales.size() * 2;
  host_ref<K>(packed, scales, X, ref, M);
  double refmax = 0;
  for (float v : ref)
    refmax = std::max(refmax, (double)std::fabs(v));

  uint8_t *dP = sycl::malloc_device<uint8_t>(packedB, q);
  uint16_t *dS = sycl::malloc_device<uint16_t>(scaleB / 2, q);
  float *dX = sycl::malloc_device<float>(K, q);
  float *dY = sycl::malloc_device<float>(M, q);
  int8_t *dXQ = sycl::malloc_device<int8_t>(K, q);
  q.memcpy(dP, packed.data(), packedB).wait();
  q.memcpy(dS, scales.data(), scaleB).wait();
  q.memcpy(dX, X.data(), (size_t)K * 4).wait();
  float xmax = 0;
  for (float v : X)
    xmax = std::max(xmax, std::fabs(v));
  float sq = xmax / 127.0f;
  double qms = actquant(q, dX, dXQ, sq, K) * 1e-6;

  auto bench = [&](const char *name, auto fn, size_t actB, double tol) {
    for (int w = 0; w < 2; ++w)
      fn();
    double best = 1e18;
    for (int r = 0; r < 10; ++r) {
      double ns = fn();
      if (ns < best)
        best = ns;
    }
    // poison output first: NaN survivors prove the kernel never wrote
    std::vector<float> poison(M, std::nanf(""));
    q.memcpy(dY, poison.data(), (size_t)M * 4).wait();
    fn();
    std::vector<float> got(M);
    q.memcpy(got.data(), dY, (size_t)M * 4).wait();
    double maxd = 0, meand = 0, meanr = 0;
    bool hasnan = false;
    for (int m = 0; m < M; ++m) {
      if (std::isnan(got[m])) {
        hasnan = true;
        break;
      }
      maxd = std::max(maxd, std::fabs((double)got[m] - (double)ref[m]) / refmax);
      meand += std::fabs((double)got[m] - (double)ref[m]);
      meanr += std::fabs((double)ref[m]);
    }
    double meanrel = meand / meanr;
    size_t traffic = packedB + scaleB + actB + (size_t)M * 4;
    double gbs = (double)traffic / best * 1e9 / 1e9;
    char row[512];
    std::snprintf(row, sizeof row,
                  "%s{\"shape\":[%d,%d],\"path\":\"%s\",\"ms\":%.3f,\"gbs\":%."
                  "1f,\"pct_roof\":%.1f,\"rel_err\":%.4f,\"mean_rel\":%.5f}",
                  first ? "" : ",", M, K, name, best * 1e-6, gbs,
                  gbs / 437.0 * 100.0, maxd, meanrel);
    json += row;
    first = false;
    std::printf("[%dx%d] %-8s %7.3f ms %6.1f GB/s %5.1f%% roof max %.4f mean %.5f%s%s\n",
                M, K, name, best * 1e-6, gbs, gbs / 437.0 * 100.0, maxd,
                meanrel, meanrel > tol ? "  FAIL" : "",
                hasnan ? "  NOTWRITTEN" : "");
  };

  bench("scalar", [&]() { return pathA<K>(q, dP, dS, dX, dY, M); },
        (size_t)K * 4, 0.002);
  bench("vecfp", [&]() { return pathB<K>(q, dP, dS, dX, dY, M); },
        (size_t)K * 4, 0.002);
  bench("dp4a-i8", [&]() { return pathC<K>(q, dP, dS, dXQ, sq, dY, M); },
        (size_t)K, 0.02);
  bench("dp4a-ur4", [&]() { return pathC4<K>(q, dP, dS, dXQ, sq, dY, M); },
        (size_t)K, 0.02);
  bench("dpas-f16", [&]() { return pathD<K>(q, dP, dS, dX, dY, M); },
        (size_t)K * 4, 0.002);
  bench("dpas-i8", [&]() { return pathE<K>(q, dP, dS, dXQ, sq, dY, M); },
        (size_t)K, 0.02);
  std::printf("[%dx%d] actquant-i8 %.3f ms (staged, excluded from GB/s)\n", M,
              K, qms);
  sycl::free(dP, q);
  sycl::free(dS, q);
  sycl::free(dX, q);
  sycl::free(dY, q);
  sycl::free(dXQ, q);
}

int main(int argc, char **argv) {
  sycl::device dev = pick_b60();
  auto async_h = [](sycl::exception_list l) {
    for (auto &e : l)
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception &x) {
        std::fprintf(stderr, "ASYNC SYCL: %s\n", x.what());
      }
  };
  sycl::queue q(dev, async_h, {sycl::property::queue::enable_profiling()});
  std::string json = "{\"device\":\"B60\",\"roof_gbs\":437,\"results\":[";
  bool first = true;
  run_shape<10240, 5120>(q, json, first);
  run_shape<17408, 5120>(q, json, first);
  run_shape<5120, 17408>(q, json, first);
  run_shape<248320, 5120>(q, json, first);
  json += "]}";
  FILE *o = stdout;
  if (argc > 1) {
    o = std::fopen(argv[1], "w");
    if (!o)
      return 1;
  }
  std::fprintf(o, "%s\n", json.c_str());
  if (o != stdout)
    std::fclose(o);
  return 0;
}
