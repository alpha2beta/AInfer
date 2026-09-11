// AInfer ESIMD smoke + XMX capability probe (T0.3) with timestamp capture (T0.4).
// Compiles under icpx (DPC++, -fsycl), runs on the B60, writes JSON report.
// Usage: esimd_smoke [report.json]
#include <level_zero/ze_api.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>

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
  std::fprintf(stderr, "FATAL: no B60 GPU visible to SYCL\n");
  std::exit(1);
}

template <int VL> struct AddTag {};
// Kernel names must be forward-declarable at namespace scope (C++17, no
// lambdas in unevaluated operands): tags live here, fill functions are
// plain function pointers.
struct TagS8 {};
struct TagF16 {};
struct TagBF16 {};
template <int VL> bool esimd_vadd(sycl::queue &q) {
  constexpr int N = 1024;
  float *a = sycl::malloc_shared<float>(N, q);
  float *b = sycl::malloc_shared<float>(N, q);
  float *c = sycl::malloc_shared<float>(N, q);
  for (int i = 0; i < N; ++i) {
    a[i] = (float)i * 0.5f;
    b[i] = (float)i * 0.25f + 1.0f;
  }
  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for<AddTag<VL>>(
        sycl::range<1>(N / VL), [=](sycl::id<1> idx) SYCL_ESIMD_KERNEL {
          unsigned off = idx[0] * VL;
          esimd::simd<float, VL> va = esimd::block_load<float, VL>(a + off);
          esimd::simd<float, VL> vb = esimd::block_load<float, VL>(b + off);
          esimd::block_store<float, VL>(c + off, va + vb);
        });
  });
  e.wait();
  bool ok = true;
  for (int i = 0; i < N; ++i)
    if (std::fabs(c[i] - (a[i] + b[i])) > 1e-5f) {
      ok = false;
      break;
    }
  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  return ok;
}

// Generic joint_matrix tile test: A[M,K] * B[K,N] + C -> D[M,N], one work-group.
template <typename Ta, typename Tb, typename Tc, int M, int N, int K,
          int SG, typename HostA, typename HostB, typename Tag>
bool jm_tile(sycl::queue &q, HostA fillA, HostB fillB,
             double *out_flops = nullptr) {
  using Ta8 = Ta;
  std::vector<Ta8> A(M * K), B(K * N);
  std::vector<Tc> C(M * N, (Tc)1), D(M * N, (Tc)0);
  for (int i = 0; i < M * K; ++i)
    A[i] = fillA(i);
  for (int i = 0; i < K * N; ++i)
    B[i] = fillB(i);
  Ta8 *dA = sycl::malloc_shared<Ta8>(M * K, q);
  Tb *dB = sycl::malloc_shared<Tb>(K * N, q);
  Tc *dC = sycl::malloc_shared<Tc>(M * N, q);
  Tc *dD = sycl::malloc_shared<Tc>(M * N, q);
  std::memcpy(dA, A.data(), sizeof(Ta8) * M * K);
  std::memcpy(dB, B.data(), sizeof(Tb) * K * N);
  std::memcpy(dC, C.data(), sizeof(Tc) * M * N);
  auto e = q.submit([&](sycl::handler &h) {
    h.parallel_for<Tag>(
        sycl::nd_range<1>({(size_t)SG}, {(size_t)SG}),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
          auto sg = it.get_sub_group();
          mx::joint_matrix<sycl::sub_group, Ta8, mx::use::a, M, K,
                           mx::layout::row_major>
              ta;
          mx::joint_matrix<sycl::sub_group, Tb, mx::use::b, K, N,
                           mx::layout::row_major>
              tb;
          mx::joint_matrix<sycl::sub_group, Tc, mx::use::accumulator, M, N,
                           mx::layout::dynamic>
              tc, td;
          mx::joint_matrix_load(
              sg, ta,
              sycl::multi_ptr<Ta8, sycl::access::address_space::global_space>(
                  dA),
              K);
          mx::joint_matrix_load(
              sg, tb,
              sycl::multi_ptr<Tb, sycl::access::address_space::global_space>(
                  dB),
              N);
          mx::joint_matrix_load(
              sg, tc,
              sycl::multi_ptr<Tc, sycl::access::address_space::global_space>(
                  dC),
              N, mx::layout::row_major);
          mx::joint_matrix_mad(sg, td, ta, tb, tc);
          mx::joint_matrix_store(
              sg, td,
              sycl::multi_ptr<Tc, sycl::access::address_space::global_space>(
                  dD),
              N, mx::layout::row_major);
        });
  });
  e.wait();
  // host reference in double
  bool ok = true;
  double maxdiff = 0;
  for (int m = 0; m < M && ok; ++m)
    for (int n = 0; n < N && ok; ++n) {
      double acc = (double)C[m * N + n];
      for (int k = 0; k < K; ++k)
        acc += (double)A[m * K + k] * (double)B[k * N + n];
      double got = (double)dD[m * N + n];
      double diff = std::fabs(got - acc);
      maxdiff = diff > maxdiff ? diff : maxdiff;
      double tol = std::is_same<Tc, float>::value ? 1e-3 * (1.0 + std::fabs(acc))
                                                  : 0.5;
      if (diff > tol)
        ok = false;
    }
  if (out_flops)
    *out_flops = maxdiff;
  sycl::free(dA, q);
  sycl::free(dB, q);
  sycl::free(dC, q);
  sycl::free(dD, q);
  return ok;
}

int main(int argc, char **argv) {
  sycl::device dev = pick_b60();
  sycl::queue q(dev, {sycl::property::queue::enable_profiling()});
  std::printf("device: %s\n",
              dev.get_info<sycl::info::device::name>().c_str());

  bool sg16 = esimd_vadd<16>(q);
  bool sg32 = esimd_vadd<32>(q);
  std::printf("esimd vadd SG16=%d SG32=%d\n", sg16, sg32);

  double md = 0;
  bool s8 =
      jm_tile<int8_t, int8_t, int32_t, 8, 16, 32, 16, int8_t (*)(int),
              int8_t (*)(int), TagS8>(
          q, +[](int i) { return (int8_t)((i * 7) % 13 - 6); },
          +[](int i) { return (int8_t)((i * 3) % 11 - 5); }, &md);
  std::printf("xmx int8 8x16x32: %d maxdiff=%g\n", s8, md);
  bool f16 = jm_tile<sycl::half, sycl::half, float, 8, 16, 16, 16,
                     sycl::half (*)(int), sycl::half (*)(int), TagF16>(
      q,
      +[](int i) { return sycl::half((float)((i * 7) % 17 - 8) * 0.25f); },
      +[](int i) { return sycl::half((float)((i * 3) % 13 - 6) * 0.25f); },
      &md);
  std::printf("xmx fp16 8x16x16: %d maxdiff=%g\n", f16, md);
  using bf16_t = sycl::ext::oneapi::bfloat16;
  bool bf16 = jm_tile<bf16_t, bf16_t, float, 8, 16, 16, 16, bf16_t (*)(int),
                      bf16_t (*)(int), TagBF16>(
      q,
      +[](int i) { return bf16_t((float)((i * 7) % 17 - 8) * 0.25f); },
      +[](int i) { return bf16_t((float)((i * 3) % 13 - 6) * 0.25f); }, &md);
  std::printf("xmx bf16 8x16x16: %d maxdiff=%g\n", bf16, md);

  // SYCL profiling timestamps from the last kernel event
  auto t0 = q.submit([&](sycl::handler &h) { h.single_task([=]() {}); });
  t0.wait();
  uint64_t ps = t0.get_profiling_info<
      sycl::info::event_profiling::command_submit>();
  uint64_t pe =
      t0.get_profiling_info<sycl::info::event_profiling::command_end>();

  // Raw Level Zero timestamps (proves the L0 timing path for T0.4/T3.1)
  uint64_t hostTs = 0, devTs = 0;
  bool l0ok = false;
  if (zeInit(ZE_INIT_FLAG_GPU_ONLY) == ZE_RESULT_SUCCESS) {
    uint32_t nd = 0;
    if (zeDriverGet(&nd, nullptr) == ZE_RESULT_SUCCESS && nd > 0) {
      std::vector<ze_driver_handle_t> dr(nd);
      zeDriverGet(&nd, dr.data());
      for (auto d : dr) {
        uint32_t nv = 0;
        if (zeDeviceGet(d, &nv, nullptr) != ZE_RESULT_SUCCESS)
          continue;
        std::vector<ze_device_handle_t> vs(nv);
        zeDeviceGet(d, &nv, vs.data());
        for (auto v : vs) {
          ze_device_properties_t pr = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
          if (zeDeviceGetProperties(v, &pr) == ZE_RESULT_SUCCESS &&
              pr.vendorId == 0x8086 && pr.deviceId == 0xe211 &&
              zeDeviceGetGlobalTimestamps(v, &hostTs, &devTs) ==
                  ZE_RESULT_SUCCESS)
            l0ok = true;
        }
      }
    }
  }

  FILE *f = stdout;
  if (argc > 1) {
    f = std::fopen(argv[1], "w");
    if (!f) {
      std::fprintf(stderr, "cannot open %s\n", argv[1]);
      return 1;
    }
  }
  std::fprintf(f,
               "{\"device\":\"B60\",\"esimd\":{\"sg16\":%s,\"sg32\":%s},\"xmx\":{"
               "\"int8_8x16x32\":%s,\"fp16_8x16x16\":%s,\"bf16_8x16x16\":%s,"
               "\"int4_native\":false},\"timestamps\":{\"sycl_submit_ns\":%llu,"
               "\"sycl_end_ns\":%llu,\"l0_ok\":%s,\"l0_host\":%llu,\"l0_dev\":"
               "%llu}}\n",
               sg16 ? "true" : "false", sg32 ? "true" : "false",
               s8 ? "true" : "false", f16 ? "true" : "false",
               bf16 ? "true" : "false", (unsigned long long)ps,
               (unsigned long long)pe, l0ok ? "true" : "false",
               (unsigned long long)hostTs, (unsigned long long)devTs);
  if (f != stdout)
    std::fclose(f);
  std::printf("l0 timestamps ok=%d\n", l0ok);
  return (sg16 && sg32 && s8 && f16 && bf16 && l0ok) ? 0 : 3;
}
