// dp4a isolated correctness probe: known inputs, device-computed, host-checked.
// Usage: dp4a_probe
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>

namespace esimd = sycl::ext::intel::esimd;

int main() {
  sycl::device dev = sycl::device(sycl::gpu_selector_v);
  sycl::queue q(dev);
  // 8 int32 lanes; each lane holds 4 int8s: w = lane+1 pattern, x = all 1s.
  // expected lane l: 4*(l+1)*1 = 4(l+1).
  int *out = sycl::malloc_shared<int>(8, q);
  for (int i = 0; i < 8; ++i)
    out[i] = -999;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class Dp4aProbe>(
        sycl::range<1>(1), [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
          esimd::simd<int, 8> acc(0);
          esimd::simd<int8_t, 32> w, x;
#pragma unroll
          for (int i = 0; i < 32; ++i) {
            w[i] = (int8_t)((i / 4) + 1);
            x[i] = (int8_t)1;
          }
          esimd::simd<int, 8> w32 = w.bit_cast_view<int>();
          esimd::simd<int, 8> x32 = x.bit_cast_view<int>();
          acc = esimd::dp4a<int>(acc, w32, x32);
          int tmp[8];
          acc.copy_to(tmp);
          for (int i = 0; i < 8; ++i)
            out[i] = tmp[i];
        });
  }).wait();
  bool ok = true;
  for (int i = 0; i < 8; ++i) {
    int expect = 4 * (i + 1);
    std::printf("lane %d: got %d expect %d %s\n", i, out[i], expect,
                out[i] == expect ? "OK" : "WRONG");
    ok &= (out[i] == expect);
  }
  // Case 2: same computation but int8 vectors built via bit_cast_view,
  // exactly like the GEMV kernel does.
  for (int i = 0; i < 8; ++i)
    out[i] = -999;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class Dp4aProbeBC>(
        sycl::range<1>(1), [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
          esimd::simd<int8_t, 32> w, x;
#pragma unroll
          for (int i = 0; i < 32; ++i) {
            w[i] = (int8_t)((i / 4) + 1);
            x[i] = (int8_t)1;
          }
          esimd::simd<int, 8> w32 = w.bit_cast_view<int>();
          esimd::simd<int, 8> x32 = x.bit_cast_view<int>();
          esimd::simd<int, 8> acc(0);
          acc = esimd::dp4a<int>(acc, w32, x32);
          int tmp[8];
          acc.copy_to(tmp);
          for (int i = 0; i < 8; ++i)
            out[i] = tmp[i];
        });
  }).wait();
  for (int i = 0; i < 8; ++i) {
    int expect = 4 * (i + 1);
    std::printf("bc lane %d: got %d expect %d %s\n", i, out[i], expect,
                out[i] == expect ? "OK" : "WRONG");
    ok &= (out[i] == expect);
  }
  std::printf("dp4a_probe: %s\n", ok ? "PASS" : "FAIL");
  sycl::free(out, q);
  return ok ? 0 : 1;
}
