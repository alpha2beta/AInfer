// AInfer prefill tiled GEMM (T3.4): C[M][N] = A[M][K] (fp16 act) x B[K][N]
// (INT4 layout-0, dequantized on the fly to f16), DPAS 8x16x16, fp32 acc.
// One SG16 per 8x16 output tile; K tiled by 16; scales reloaded per group.
// Verified vs fp32 host (depacked weights). Usage: prefill_t34 [report.json]
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

template <int K, int TN> struct GemmTag {};
// TN = N (multiple of 16). A: MxK half row-major. P: KxN packed nibbles
// (B row k = weight row for output col? No: B is K rows x N cols stored as
// N/K-major? We store B transposed: Bt[N][K] row-major packed, i.e. output
// channel n holds K nibbles + scales; dequant Bt[n][k] on the fly.)
template <int K, int TN> struct GemmMT4Tag {};
// M-tiled variant: each subgroup computes 4 stacked 8x16 tiles (32 rows).
// B is loaded once per K-tile and shared across the 4 row-blocks -> B traffic /4.
template <int K, int TN>
double gemmMT4(sycl::queue &q, const sycl::half *A, const uint8_t *P,
               const uint16_t *S, float *C, int M) {
  constexpr int TR = 8, TC = 16, G = K / 128, UR = 4;
  int nBr = M / (TR * UR), nBc = TN / 16;
  auto e = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<sycl::half, 1> sA(sycl::range<1>(128 * UR), h);
    sycl::local_accessor<sycl::half, 1> sB(sycl::range<1>(256), h);
    sycl::local_accessor<float, 1> sC(sycl::range<1>(128 * UR), h);
    h.parallel_for<GemmMT4Tag<K, TN>>(
        sycl::nd_range<1>({(size_t)nBr * nBc * 16}, {16}),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          auto sg = it.get_sub_group();
          int gid = (int)it.get_group(0);
          int br = gid / nBc, bc = gid % nBc;
          int lid = (int)sg.get_local_id()[0];
          mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TR,
                           16, mx::layout::dynamic>
              acc[UR];
          for (int r = 0; r < UR; ++r)
            mx::joint_matrix_fill(sg, acc[r], 0.0f);
          for (int kt = 0; kt < K; kt += TC) {
            for (int r = 0; r < UR; ++r)
              for (int u = 0; u < 8; ++u) { // A: 4x128 elems
                int idx = lid * 8 + u, rr = idx / TC, c = idx % TC;
                sA[r * 128 + rr * TC + c] =
                    A[(br * TR * UR + r * TR + rr) * K + kt + c];
              }
            int g = kt / 128;
            for (int u = 0; u < 16; ++u) { // B: 256 elems (shared)
              int idx = lid * 16 + u, i = idx / 16, n2 = idx % 16;
              int n = bc * 16 + n2;
              size_t li = (size_t)n * K + kt + i;
              uint8_t b = P[li / 2];
              int nib = (li & 1) ? (b >> 4) : (b & 0xF);
              if (nib >= 8)
                nib -= 16;
              sB[i * 16 + n2] =
                  sycl::half((float)nib * bf16_to_f32(S[(size_t)n * G + g]));
            }
            sg.barrier();
            for (int r = 0; r < UR; ++r) {
              mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TR,
                               TC, mx::layout::row_major>
                  ta;
              mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TC,
                               16, mx::layout::row_major>
                  tb;
              auto mpA =
                  sycl::multi_ptr<sycl::half,
                                  sycl::access::address_space::local_space>(
                      &sA[r * 128]);
              mx::joint_matrix_load(sg, ta, mpA, TC);
              mx::joint_matrix_load(
                  sg, tb,
                  sB.template get_multi_ptr<sycl::access::decorated::legacy>(),
                  16);
              mx::joint_matrix_mad(sg, acc[r], ta, tb, acc[r]);
            }
            sg.barrier();
          }
          for (int r = 0; r < UR; ++r) {
            mx::joint_matrix_store(
                sg, acc[r],
                sycl::multi_ptr<float,
                                sycl::access::address_space::local_space>(
                    &sC[r * 128]),
                16, mx::layout::row_major);
          }
          sg.barrier();
          for (int u = 0; u < 32; ++u) { // 512 elems, 32 per lane
            int idx = lid * 32 + u, r = idx / 128, rest = idx % 128;
            int rr = rest / 16, c = rest % 16;
            C[(br * TR * UR + r * TR + rr) * TN + bc * 16 + c] =
                sC[r * 128 + rr * 16 + c];
          }
        });
  });
  e.wait();
  return (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_end>() -
         (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_start>();
}

template <int K, int TN>
double gemm(sycl::queue &q, const sycl::half *A, const uint8_t *P,
            const uint16_t *S, float *C, int M) {
  constexpr int TR = 8, TC = 16, G = K / 128;
  int nBr = M / TR, nBc = TN / 16;
  auto e = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<sycl::half, 1> sA(sycl::range<1>(128), h);
    sycl::local_accessor<sycl::half, 1> sB(sycl::range<1>(256), h);
    sycl::local_accessor<float, 1> sC(sycl::range<1>(128), h);
    h.parallel_for<GemmTag<K, TN>>(
        sycl::nd_range<1>({(size_t)nBr * nBc * 16}, {16}),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          auto sg = it.get_sub_group();
          int gid = (int)it.get_group(0);
          int br = gid / nBc, bc = gid % nBc;
          int lid = (int)sg.get_local_id()[0];
          mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TR,
                           16, mx::layout::dynamic>
              acc;
          mx::joint_matrix_fill(sg, acc, 0.0f);
          for (int kt = 0; kt < K; kt += TC) {
            for (int u = 0; u < 8; ++u) { // A: 128 elems, 8 per lane
              int idx = lid * 8 + u, r = idx / TC, c = idx % TC;
              sA[r * TC + c] = A[(br * TR + r) * K + kt + c];
            }
            int g = kt / 128;
            for (int u = 0; u < 16; ++u) { // B: 256 elems, 16 per lane
              int idx = lid * 16 + u, i = idx / 16, n2 = idx % 16;
              int n = bc * 16 + n2;
              size_t li = (size_t)n * K + kt + i;
              uint8_t b = P[li / 2];
              int nib = (li & 1) ? (b >> 4) : (b & 0xF);
              if (nib >= 8)
                nib -= 16;
              sB[i * 16 + n2] =
                  sycl::half((float)nib * bf16_to_f32(S[(size_t)n * G + g]));
            }
            sg.barrier();
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TR, TC,
                             mx::layout::row_major>
                ta;
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TC, 16,
                             mx::layout::row_major>
                tb;
            mx::joint_matrix_load(sg, ta, sA.template get_multi_ptr<sycl::access::decorated::legacy>(), TC);
            mx::joint_matrix_load(sg, tb, sB.template get_multi_ptr<sycl::access::decorated::legacy>(), 16);
            mx::joint_matrix_mad(sg, acc, ta, tb, acc);
            sg.barrier();
          }
          mx::joint_matrix_store(sg, acc, sC.template get_multi_ptr<sycl::access::decorated::legacy>(), 16,
                                 mx::layout::row_major);
          sg.barrier();
          for (int u = 0; u < 8; ++u) { // 128 elems, 8 per lane
            int idx = lid * 8 + u, r = idx / 16, c = idx % 16;
            C[(br * TR + r) * TN + bc * 16 + c] = sC[r * 16 + c];
          }
        });
  });
  e.wait();
  return (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_end>() -
         (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_start>();
}

template <int K, int TN>
static void run_shape(sycl::queue &q, std::string &json, bool &first, int M) {
  // PRNG fp16-range activations; INT4 weights packed like the exporter.
  std::vector<sycl::half> A((size_t)M * K);
  std::vector<float> Bf((size_t)TN * K), X(K, 0);
  uint64_t s = 0x51ab + M * 97 + K * 13 + TN;
  for (size_t i = 0; i < A.size(); ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    A[i] = sycl::half((float)((int)((s >> 33) & 0xFFFF) - 32768) *
                      (0.5f / 32768.0f));
  }
  for (size_t i = 0; i < Bf.size(); ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    Bf[i] = (float)((int)((s >> 33) & 0xFFFF) - 32768) * (0.02f / 32768.0f);
  }
  // pack B rows (output channels) g128
  constexpr int G = K / 128;
  std::vector<uint8_t> P((size_t)TN * K / 2, 0);
  std::vector<uint16_t> S((size_t)TN * G, 0);
  for (int n = 0; n < TN; ++n)
    for (int g = 0; g < G; ++g) {
      float amax = 0;
      for (int j = 0; j < 128; ++j)
        amax = std::max(amax, std::fabs(Bf[(size_t)n * K + g * 128 + j]));
      float sc = amax == 0 ? 1.0f : amax / 7.0f;
      S[(size_t)n * G + g] = f32_to_bf16(sc);
      for (int j = 0; j < 128; ++j) {
        float v = Bf[(size_t)n * K + g * 128 + j] / sc;
        int qi = (int)std::lrint(v);
        qi = qi < -8 ? -8 : (qi > 7 ? 7 : qi);
        size_t idx = (size_t)n * K + g * 128 + j;
        if (j & 1)
          P[idx / 2] |= (uint8_t)((qi & 0xF) << 4);
        else
          P[idx / 2] = (uint8_t)(qi & 0xF);
      }
    }
  // host fp32 reference (depacked)
  std::vector<float> ref((size_t)M * TN, 0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < TN; ++n) {
      double acc = 0;
      for (int g = 0; g < G; ++g) {
        float sc = bf16_to_f32(S[(size_t)n * G + g]);
        for (int j = 0; j < 128; ++j) {
          size_t idx = (size_t)n * K + g * 128 + j;
          uint8_t b = P[idx / 2];
          int nib = (idx & 1) ? (b >> 4) : (b & 0xF);
          if (nib >= 8)
            nib -= 16;
          acc += (double)((float)A[(size_t)m * K + g * 128 + j] * nib * sc);
        }
      }
      ref[(size_t)m * TN + n] = (float)acc;
    }
  double refmax = 0;
  for (float v : ref)
    refmax = std::max(refmax, (double)std::fabs(v));

  sycl::half *dA = sycl::malloc_device<sycl::half>((size_t)M * K, q);
  uint8_t *dP = sycl::malloc_device<uint8_t>(P.size(), q);
  uint16_t *dS = sycl::malloc_device<uint16_t>(S.size(), q);
  float *dC = sycl::malloc_device<float>((size_t)M * TN, q);
  q.memcpy(dA, A.data(), A.size() * 2).wait();
  q.memcpy(dP, P.data(), P.size()).wait();
  q.memcpy(dS, S.data(), S.size() * 2).wait();
  (void)X;
  auto verify = [&](const char *tag, double best) {
    std::vector<float> got((size_t)M * TN);
    q.memcpy(got.data(), dC, got.size() * 4).wait();
    double maxd = 0, meand = 0, meanr = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      maxd = std::max(maxd, std::fabs((double)got[i] - ref[i]) / refmax);
      meand += std::fabs(got[i] - ref[i]);
      meanr += std::fabs(ref[i]);
    }
    double flops = 2.0 * M * TN * K;
    double traffic = (double)M * K * 2 + P.size() + S.size() * 2 +
                     (double)M * TN * 4;
    char row[512];
    std::snprintf(row, sizeof row,
                  "%s{\"M\":%d,\"K\":%d,\"N\":%d,\"path\":\"%s\",\"ms\":%.3f,"
                  "\"gflops\":%.0f,\"gbs\":%.1f,\"max\":%.2e,\"mean_rel\":%.2e}",
                  first ? "" : ",", M, K, TN, tag, best * 1e-6,
                  flops / best, traffic / best, maxd, meand / meanr);
    json += row;
    first = false;
    std::printf("[MxKxN=%dx%dx%d %-4s] %7.3f ms %6.0f GFLOPS %6.1f GB/s max %.2e mean %.2e\n",
                M, K, TN, tag, best * 1e-6, flops / best, traffic / best, maxd,
                meand / meanr);
  };
  for (int w = 0; w < 2; ++w)
    gemm<K, TN>(q, dA, dP, dS, dC, M);
  double best = 1e18;
  for (int r = 0; r < 10; ++r) {
    double ns = gemm<K, TN>(q, dA, dP, dS, dC, M);
    if (ns < best)
      best = ns;
  }
  verify("base", best);
  if (M % 32 == 0) {
    for (int w = 0; w < 2; ++w)
      gemmMT4<K, TN>(q, dA, dP, dS, dC, M);
    best = 1e18;
    for (int r = 0; r < 10; ++r) {
      double ns = gemmMT4<K, TN>(q, dA, dP, dS, dC, M);
      if (ns < best)
        best = ns;
    }
    verify("mt4", best);
  }
  sycl::free(dA, q);
  sycl::free(dP, q);
  sycl::free(dS, q);
  sycl::free(dC, q);
}

int main(int argc, char **argv) {
  sycl::device dev = pick_b60();
  sycl::queue q(dev, {sycl::property::queue::enable_profiling()});
  std::string json = "{\"device\":\"B60\",\"results\":[";
  bool first = true;
  run_shape<5120, 17408>(q, json, first, 128);
  run_shape<5120, 17408>(q, json, first, 512);
  run_shape<5120, 17408>(q, json, first, 1024);
  run_shape<17408, 5120>(q, json, first, 512);
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
