// AInfer bandwidth + dispatch baselines (T3.1).
// L0 C API for copies/allocs/queues + one ESIMD read kernel + empty-launch
// overhead + command-list replay timing. Writes JSON report.
// Usage: bench_t31 [report.json]
#include <level_zero/ze_api.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
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

template <int VL> struct CopyTag {};

// Block copy: every byte read is forced by a matching verified write.
// Returns best device seconds via SYCL profiling.
template <int VL>
double esimd_copy_bw(sycl::queue &q, const float *src, float *dst, size_t n,
                     int iters) {
  size_t nWI = n / VL;
  double best = 1e9;
  for (int it = 0; it < iters; ++it) {
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<CopyTag<VL>>(
          sycl::range<1>(nWI), [=](sycl::id<1> idx) SYCL_ESIMD_KERNEL {
            esimd::simd<float, VL> v =
                esimd::block_load<float, VL>(src + idx[0] * VL);
            esimd::block_store<float, VL>(dst + idx[0] * VL, v);
          });
    });
    e.wait();
    double s = (double)e.template get_profiling_info<
                   sycl::info::event_profiling::command_start>(),
           t = (double)e.template get_profiling_info<
                   sycl::info::event_profiling::command_end>();
    double dt = (t - s) * 1e-9;
    if (dt < best)
      best = dt;
  }
  return best;
}

int main(int argc, char **argv) {
  sycl::device sdev = pick_b60();
  sycl::queue q(sdev, {sycl::property::queue::enable_profiling()});

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
  ze_command_list_handle_t list = nullptr;
  ze_command_queue_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &ldesc, &list));

  std::string json = "{\"device\":\"B60\",\"copies\":[";
  const size_t sizes[] = {1u << 22, 1u << 26, 1u << 28, 1u << 30};
  const int reps = 5;
  // pinned host staging (true PCIe roof, not USB-bound)
  void *hbuf = nullptr;
  ze_host_mem_alloc_desc_t hdesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                    nullptr, 0};
  CHECK(zeMemAllocHost(ctx, &hdesc, 1u << 30, 0, &hbuf));
  ze_device_mem_alloc_desc_t mdesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  void *dA = nullptr, *dB = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, 1ull << 32, 0, dev, &dA)); // 4 GiB
  CHECK(zeMemAllocDevice(ctx, &mdesc, 1u << 30, 0, dev, &dB)); // 1 GiB
  std::memset(hbuf, 0x5a, 1u << 30);

  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  bool first = true;
  const char *dnames[3] = {"h2d_pinned", "d2h_pinned", "d2d"};
  for (int dir = 0; dir < 3; ++dir) {
    for (size_t sz : sizes) {
      if (dir < 2 && sz > (1u << 30))
        continue;
      std::vector<double> ts;
      for (int r = 0; r < reps + 1; ++r) {
        double t0 = now_s();
        if (dir == 0)
          CHECK(zeCommandListAppendMemoryCopy(list, dB, hbuf, sz, nullptr, 0,
                                              nullptr));
        else if (dir == 1)
          CHECK(zeCommandListAppendMemoryCopy(list, hbuf, dB, sz, nullptr, 0,
                                              nullptr));
        else
          CHECK(zeCommandListAppendMemoryCopy(list, dB, dA, sz, nullptr, 0,
                                              nullptr));
        double t1 = now_s();
        if (r > 0)
          ts.push_back((t1 - t0));
      }
      double bw = (double)sz / med(ts) / 1e9;
      char row[256];
      std::snprintf(row, sizeof row, "%s{\"dir\":\"%s\",\"bytes\":%zu,\"gbs\":%.2f}",
                    first ? "" : ",", dnames[dir], sz, bw);
      json += row;
      first = false;
      std::printf("%s %zu MiB: %.2f GB/s\n", dnames[dir], sz >> 20, bw);
    }
  }
  // 4 GiB D2D single shot
  {
    std::vector<double> ts;
    for (int r = 0; r < 3 + 1; ++r) {
      double t0 = now_s();
      CHECK(zeCommandListAppendMemoryCopy(list, dB, dA, 1u << 30, nullptr, 0,
                                          nullptr));
      // 4x1GiB to approximate 4GiB without a second huge alloc
      CHECK(zeCommandListAppendMemoryCopy(
          list, (char *)dB, (char *)dA + (1u << 30), 1u << 30, nullptr, 0,
          nullptr));
      CHECK(zeCommandListAppendMemoryCopy(
          list, (char *)dB, (char *)dA + (2u << 30), 1u << 30, nullptr, 0,
          nullptr));
      CHECK(zeCommandListAppendMemoryCopy(
          list, (char *)dB, (char *)dA + (3u << 30), 1u << 30, nullptr, 0,
          nullptr));
      double t1 = now_s();
      if (r > 0)
        ts.push_back(t1 - t0);
    }
    char row[256];
    std::snprintf(row, sizeof row, ",{\"dir\":\"d2d\",\"bytes\":%llu,\"gbs\":%.2f}",
                  4ull << 30, (double)(4ull << 30) / med(ts) / 1e9);
    json += row;
    std::printf("d2d 4096 MiB: %.2f GB/s\n",
                (double)(4ull << 30) / med(ts) / 1e9);
  }
  json += "],\"esimd_copy\":[";

  // ESIMD device-side copy bandwidth: 1 GiB array, VL=1024 (4 KiB/WI,
  // 2 GiB traffic). Source is incompressible PRNG data (uniform fills can hit
  // memory-side compression and fake the roofline); dst sample verified.
  const size_t N = 1u << 28; // floats = 1 GiB
  float *src = sycl::malloc_device<float>(N, q);
  float *dst = sycl::malloc_device<float>(N, q);
  std::vector<float> pattern(1u << 20);
  {
    uint64_t s = 0x12345678u;
    for (size_t i = 0; i < N; i += pattern.size()) {
      for (size_t j = 0; j < pattern.size(); ++j) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        pattern[j] =
            (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
      }
      q.memcpy(src + i, pattern.data(), pattern.size() * 4).wait();
    }
  }
  first = true;
  for (int it = 0; it < 1; ++it) {
    (void)it;
    double dt = esimd_copy_bw<1024>(q, src, dst, N, 6);
    std::vector<float> host(1u << 20), expect(1u << 20);
    q.memcpy(host.data(), dst, host.size() * 4).wait();
    q.memcpy(expect.data(), src, expect.size() * 4).wait();
    double maxd = 0;
    for (size_t i = 0; i < host.size(); ++i) {
      double d = std::fabs((double)host[i] - (double)expect[i]);
      if (d > maxd)
        maxd = d;
    }
    double bw = (double)N * 4 * 2 / dt / 1e9;
    char row[256];
    std::snprintf(row, sizeof row,
                  "%s{\"vl\":1024,\"bytes\":%zu,\"gbs\":%.1f,\"maxdiff\":%.3f}",
                  first ? "" : ",", N * 4 * 2, bw, maxd);
    json += row;
    first = false;
    std::printf("esimd copy 2048 MiB traffic: %.1f GB/s maxdiff=%.3f\n", bw,
                maxd);
  }
  json += "],";

  // Dispatch overhead: 2000 empty single_task launches
  {
    for (int r = 0; r < 50; ++r)
      q.submit([&](sycl::handler &h) { h.single_task([=]() {}); }).wait();
    double t0 = now_s();
    const int K = 2000;
    for (int r = 0; r < K; ++r)
      q.submit([&](sycl::handler &h) { h.single_task([=]() {}); }).wait();
    double per_us = (now_s() - t0) / K * 1e6;
    char row[256];
    std::snprintf(row, sizeof row, "\"dispatch_us\":%.2f,", per_us);
    json += row;
    std::printf("empty launch: %.2f us/launch\n", per_us);
  }

  // Command-list replay: regular list with timestamp write + barrier, x1000
  {
    void *tsdev = nullptr;
    CHECK(zeMemAllocDevice(ctx, &mdesc, 4096, 0, dev, &tsdev));
    ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                     nullptr, 0, 0, 0,
                                     ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                     ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_queue_handle_t queue = nullptr;
    CHECK(zeCommandQueueCreate(ctx, dev, &qdesc, &queue));
    ze_command_list_desc_t cdesc2 = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                                     nullptr, 0, 0};
    ze_command_list_handle_t cl = nullptr;
    CHECK(zeCommandListCreate(ctx, dev, &cdesc2, &cl));
    CHECK(zeCommandListAppendWriteGlobalTimestamp(cl, (uint64_t *)tsdev,
                                                  nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(cl, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(cl));
    ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
    ze_fence_handle_t fence = nullptr;
    CHECK(zeFenceCreate(queue, &fdesc, &fence));
    for (int r = 0; r < 50; ++r) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cl, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
    }
    double t0 = now_s();
    const int K = 1000;
    for (int r = 0; r < K; ++r) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cl, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
    }
    double per_us = (now_s() - t0) / K * 1e6;
    char row[256];
    std::snprintf(row, sizeof row, "\"replay_us\":%.2f}", per_us);
    json += row;
    std::printf("cmdlist replay: %.2f us/execute\n", per_us);
    CHECK(zeFenceDestroy(fence));
    CHECK(zeCommandListDestroy(cl));
    CHECK(zeCommandQueueDestroy(queue));
    CHECK(zeMemFree(ctx, tsdev));
  }

  CHECK(zeCommandListDestroy(list));
  CHECK(zeMemFree(ctx, dA));
  CHECK(zeMemFree(ctx, dB));
  CHECK(zeMemFree(ctx, hbuf));
  CHECK(zeContextDestroy(ctx));

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
