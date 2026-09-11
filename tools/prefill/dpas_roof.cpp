// DPAS roof microbench: pure joint_matrix mad, no dequant, no SLM fill games.
// A/B/C all global row-major f16/f32. Reports achieved DPAS TFLOPS to tell
// whether 8x16x16 engages XMX on the B60 or falls back to FMA emulation.
// Usage: dpas_roof [M]
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace mx = sycl::ext::oneapi::experimental::matrix;

template <int K, int TN> struct RoofTag {};
template <int K, int TN> double roof(sycl::queue &q, const sycl::half *A,
                                     const sycl::half *B, float *C, int M) {
  constexpr int TR = 8;
  int nBr = M / TR, nBc = TN / 16;
  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for<RoofTag<K, TN>>(
        sycl::nd_range<1>({(size_t)nBr * nBc * 16}, {16}),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          auto sg = it.get_sub_group();
          int gid = (int)it.get_group(0);
          int br = gid / nBc, bc = gid % nBc;
          mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TR,
                           16, mx::layout::dynamic>
              acc;
          mx::joint_matrix_fill(sg, acc, 0.0f);
          for (int kt = 0; kt < K; kt += 16) {
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TR, 16,
                             mx::layout::row_major>
                ta;
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16,
                             mx::layout::row_major>
                tb;
            sycl::multi_ptr<const sycl::half,
                            sycl::access::address_space::global_space>
                mpA(A + (br * TR) * K + kt), mpB(B + kt * TN + bc * 16);
            mx::joint_matrix_load(sg, ta, mpA, K);
            mx::joint_matrix_load(sg, tb, mpB, TN);
            mx::joint_matrix_mad(sg, acc, ta, tb, acc);
          }
          sycl::multi_ptr<float, sycl::access::address_space::global_space> mpC(
              C + (br * TR) * TN + bc * 16);
          mx::joint_matrix_store(sg, acc, mpC, TN, mx::layout::row_major);
        });
  });
  e.wait();
  return (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_end>() -
         (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_start>();
}

int main(int argc, char **argv) {
  sycl::device dev = sycl::device(sycl::gpu_selector_v);
  sycl::queue q(dev, {sycl::property::queue::enable_profiling()});
  int M = argc > 1 ? atoi(argv[1]) : 2048;
  constexpr int K = 4096, TN = 512;
  std::vector<sycl::half> A((size_t)M * K, sycl::half(0.5f));
  std::vector<sycl::half> B((size_t)K * TN, sycl::half(0.5f));
  sycl::half *dA = sycl::malloc_device<sycl::half>(A.size(), q);
  sycl::half *dB = sycl::malloc_device<sycl::half>(B.size(), q);
  float *dC = sycl::malloc_device<float>((size_t)M * TN, q);
  q.memcpy(dA, A.data(), A.size() * 2).wait();
  q.memcpy(dB, B.data(), B.size() * 2).wait();
  for (int w = 0; w < 3; ++w)
    roof<K, TN>(q, dA, dB, dC, M);
  double best = 1e18;
  for (int r = 0; r < 10; ++r) {
    double ns = roof<K, TN>(q, dA, dB, dC, M);
    if (ns < best)
      best = ns;
  }
  double flops = 2.0 * M * TN * K;
  std::printf("dpas roof MxKxN=%dx%dx%d: %.3f ms %.1f TFLOPS\n", M, K, TN,
              best * 1e-6, flops / best / 1e3);
  return 0;
}
