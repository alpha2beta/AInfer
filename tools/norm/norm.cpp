// AInfer RMSNorm kernels (T3.5): plain + fused residual-add.
// Hidden 5120, eps 1e-6. Rows striped 1/WI (decode) or batched (prefill).
// Verified vs float64 host reference (same math as reference/operators_t14).
// Usage: norm_t35 [report.json]
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
constexpr int H = 5120;
constexpr float EPS = 1e-6f;

static sycl::device pick_b60() {
  for (auto p : sycl::platform::get_platforms())
    for (auto d : p.get_devices(sycl::info::device_type::gpu))
      if (d.get_info<sycl::info::device::name>().find("B60") !=
          std::string::npos)
        return d;
  std::fprintf(stderr, "FATAL: no B60\n");
  std::exit(1);
}

template <bool FUSED> struct NormTag {};
// FUSED=false: y = rmsnorm(x) * w ; true: y = rmsnorm(x + r) * w
template <bool FUSED>
double norm_kernel(sycl::queue &q, const float *X, const float *R,
                   const float *W, float *Y, int rows) {
  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for<NormTag<FUSED>>(
        sycl::range<1>(rows), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
          int r = id[0];
          const float *x = X + (size_t)r * H;
          esimd::simd<float, 128> psum(0);
          for (int j = 0; j < H; j += 128) {
            esimd::simd<float, 128> v =
                esimd::block_load<float, 128>(x + j);
            if constexpr (FUSED) {
              esimd::simd<float, 128> rv =
                  esimd::block_load<float, 128>(R + (size_t)r * H + j);
              v += rv;
            }
            psum += v * v;
          }
          // NOTE: esimd::reduce<float> miscompiles on this toolchain; fold manually.
          float ptmp[128];
          psum.copy_to(ptmp);
          double ss = 0;
          for (int u = 0; u < 128; ++u)
            ss += (double)ptmp[u];
          float inv = 1.0f / std::sqrt(ss / H + EPS);
          for (int j = 0; j < H; j += 128) {
            esimd::simd<float, 128> v =
                esimd::block_load<float, 128>(x + j);
            if constexpr (FUSED)
              v += esimd::block_load<float, 128>(R + (size_t)r * H + j);
            esimd::simd<float, 128> w =
                esimd::block_load<float, 128>(W + j);
            esimd::block_store<float, 128>(Y + (size_t)r * H + j, v * inv * w);
          }
        });
  });
  e.wait();
  return (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_end>() -
         (double)e.template get_profiling_info<
             sycl::info::event_profiling::command_start>();
}

int main(int argc, char **argv) {
  sycl::device dev = pick_b60();
  sycl::queue q(dev, {sycl::property::queue::enable_profiling()});
  std::string json = "{\"device\":\"B60\",\"hidden\":5120,\"results\":[";
  bool first = true;
  // realistic magnitudes: x ~ N(0,1), w = 1 + 0.1*N(0,1)
  for (int rows : {1, 512, 4096}) {
    std::vector<float> X((size_t)rows * H), R((size_t)rows * H), W(H);
    uint64_t s = 0xabcdef + rows;
    auto rnd = [&]() {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
    };
    for (auto &v : X)
      v = rnd();
    for (auto &v : R)
      v = rnd() * 2.0f;
    for (auto &v : W)
      v = 1.0f + rnd() * 0.1f;
    // host float64 reference
    std::vector<float> ref((size_t)rows * H);
    for (int r = 0; r < rows; ++r) {
      double ss = 0;
      for (int j = 0; j < H; ++j) {
        double v = X[(size_t)r * H + j] + R[(size_t)r * H + j];
        ss += v * v;
      }
      double inv = 1.0 / std::sqrt(ss / H + 1e-6);
      for (int j = 0; j < H; ++j)
        ref[(size_t)r * H + j] =
            (float)((X[(size_t)r * H + j] + R[(size_t)r * H + j]) * inv *
                    W[j]);
    }
    float *dX = sycl::malloc_device<float>((size_t)rows * H, q);
    float *dR = sycl::malloc_device<float>((size_t)rows * H, q);
    float *dW = sycl::malloc_device<float>(H, q);
    float *dY = sycl::malloc_device<float>((size_t)rows * H, q);
    q.memcpy(dX, X.data(), (size_t)rows * H * 4).wait();
    q.memcpy(dR, R.data(), (size_t)rows * H * 4).wait();
    q.memcpy(dW, W.data(), (size_t)H * 4).wait();
    // plain path: zero residual so ref formula still holds
    std::vector<float> Z((size_t)rows * H, 0.0f);
    float *dZ = sycl::malloc_device<float>((size_t)rows * H, q);
    q.memcpy(dZ, Z.data(), (size_t)rows * H * 4).wait();
    // reference for plain: recompute without residual
    std::vector<float> refP((size_t)rows * H);
    for (int r = 0; r < rows; ++r) {
      double ss = 0;
      for (int j = 0; j < H; ++j) {
        double v = X[(size_t)r * H + j];
        ss += v * v;
      }
      double inv = 1.0 / std::sqrt(ss / H + 1e-6);
      for (int j = 0; j < H; ++j)
        refP[(size_t)r * H + j] = (float)(X[(size_t)r * H + j] * inv * W[j]);
    }
    auto bench = [&](const char *nm, auto fn, const std::vector<float> &rf,
                      size_t traffic) {
      for (int w = 0; w < 2; ++w)
        fn();
      double best = 1e18;
      for (int r = 0; r < 10; ++r) {
        double ns = fn();
        if (ns < best)
          best = ns;
      }
      std::vector<float> got((size_t)rows * H);
      q.memcpy(got.data(), dY, (size_t)rows * H * 4).wait();
      double maxd = 0, meand = 0, meanr = 0;
      bool nan = false;
      for (size_t i = 0; i < got.size(); ++i) {
        if (std::isnan(got[i])) {
          nan = true;
          break;
        }
        maxd = std::max(maxd, (double)std::fabs(got[i] - rf[i]));
        meand += std::fabs(got[i] - rf[i]);
        meanr += std::fabs(rf[i]);
      }
      double gbs = (double)traffic / best * 1e9 / 1e9;
      char row[512];
      std::snprintf(
          row, sizeof row,
          "%s{\"rows\":%d,\"path\":\"%s\",\"ms\":%.4f,\"gbs\":%.1f,\"max\":%.2e,"
          "\"mean_rel\":%.2e%s}",
          first ? "" : ",", rows, nm, best * 1e-6, gbs, maxd, meand / meanr,
          nan ? ",\"NOTWRITTEN\":true" : "");
      json += row;
      first = false;
      std::printf("[rows=%d] %-12s %8.4f ms %6.1f GB/s max %.2e meanrel %.2e%s\n",
                  rows, nm, best * 1e-6, gbs, maxd, meand / meanr,
                  nan ? " NOTWRITTEN" : "");
    };
    bench("rmsnorm", [&]() { return norm_kernel<false>(q, dX, dZ, dW, dY, rows); },
          refP, (size_t)rows * H * 4 * 2 + H * 4);
    bench("resid+rms", [&]() { return norm_kernel<true>(q, dX, dR, dW, dY, rows); },
          ref, (size_t)rows * H * 4 * 3 + H * 4);
    sycl::free(dX, q);
    sycl::free(dR, q);
    sycl::free(dW, q);
    sycl::free(dY, q);
    sycl::free(dZ, q);
  }
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
