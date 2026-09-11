// AInfer device-side argmax (T3.9): greedy token selection on device.
// Two-stage parallel reduction over vocab (248320); host copies back 4 bytes
// (one int32) instead of the full logits vector. First-max wins ties.
// Fixtures: planted spike (random pos), spike at 0, spike at last, all-equal,
// double-max tie. Validated vs host reference. Usage: sample_t39 [report.json]
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

struct S1Tag {};
struct S2Tag {};

static sycl::device pick_b60() {
  for (auto p : sycl::platform::get_platforms())
    for (auto d : p.get_devices(sycl::info::device_type::gpu))
      if (d.get_info<sycl::info::device::name>().find("B60") !=
          std::string::npos)
        return d;
  std::fprintf(stderr, "FATAL: no B60\n");
  std::exit(1);
}
static double ev_ms(sycl::event e) {
  e.wait();
  return (double)(e.get_profiling_info<
                      sycl::info::event_profiling::command_end>() -
                  e.get_profiling_info<
                      sycl::info::event_profiling::command_start>()) *
         1e-6;
}

// Stage 1: NWG work-groups; each reduces its strided slice to one (val, idx).
// Stage 2: single WG reduces NWG partials to the global argmax.
static int device_argmax(sycl::queue &q, const float *dLogits, int V,
                         float *dPVal, int *dPIdx, int *dOut, double &ms1,
                         double &ms2) {
  constexpr int WG = 256;
  constexpr int NWG = 64;
  auto e1 = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> bv(WG, h);
    sycl::local_accessor<int, 1> bi(WG, h);
    h.parallel_for<S1Tag>(sycl::nd_range<1>(NWG * WG, WG), [=](sycl::nd_item<1> it) {
      int lid = (int)it.get_local_id(0);
      int gid = (int)it.get_group(0);
      float best = -std::numeric_limits<float>::infinity();
      int bi_ = std::numeric_limits<int>::max();
      for (int i = gid * WG + lid; i < V; i += NWG * WG) {
        float v = dLogits[i];
        if (v > best) {
          best = v;
          bi_ = i;
        }
      }
      bv[lid] = best;
      bi[lid] = bi_;
      it.barrier(sycl::access::fence_space::local_space);
      for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) {
          // strictly greater wins => earliest index survives ties; on equal
          // values keep the smaller index explicitly
          if (bv[lid + s] > bv[lid] ||
              (bv[lid + s] == bv[lid] && bi[lid + s] < bi[lid])) {
            bv[lid] = bv[lid + s];
            bi[lid] = bi[lid + s];
          }
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      if (lid == 0) {
        dPVal[gid] = bv[0];
        dPIdx[gid] = bi[0];
      }
    });
  });
  ms1 = ev_ms(e1);
  // Stage 2 via nd_item for barrier support:
  auto e3 = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> bv2(WG, h);
    sycl::local_accessor<int, 1> bi2(WG, h);
    h.parallel_for<S2Tag>(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
      int lid = (int)it.get_local_id(0);
      if (lid < NWG) {
        bv2[lid] = dPVal[lid];
        bi2[lid] = dPIdx[lid];
      } else {
        bv2[lid] = -std::numeric_limits<float>::infinity();
        bi2[lid] = std::numeric_limits<int>::max();
      }
      it.barrier(sycl::access::fence_space::local_space);
      for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) {
          if (bv2[lid + s] > bv2[lid] ||
              (bv2[lid + s] == bv2[lid] && bi2[lid + s] < bi2[lid])) {
            bv2[lid] = bv2[lid + s];
            bi2[lid] = bi2[lid + s];
          }
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      if (lid == 0)
        dOut[0] = bi2[0];
    });
  });
  ms2 = ev_ms(e3);
  int out = -1;
  q.memcpy(&out, dOut, 4).wait();
  return out;
}

int main(int argc, char **argv) {
  constexpr int V = 248320;
  sycl::device dev = pick_b60();
  sycl::queue q(dev, {sycl::property::queue::enable_profiling()});
  float *dL = sycl::malloc_device<float>(V, q);
  float *dPV = sycl::malloc_device<float>(64, q);
  int *dPI = sycl::malloc_device<int>(64, q);
  int *dO = sycl::malloc_device<int>(1, q);

  struct Case {
    const char *name;
    int expect;
  };
  // build fixtures on host
  std::vector<std::vector<float>> logits;
  std::vector<Case> cases;
  {
    uint64_t s = 0x390;
    auto rnd = [&]() {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
    };
    // 1. planted spike at pseudo-random pos
    std::vector<float> v1(V);
    for (auto &x : v1)
      x = rnd() - 0.5f; // (-1.5, 0.5]
    int p1 = 123456;
    v1[p1] = 10.0f;
    logits.push_back(std::move(v1));
    cases.push_back({"spike-mid", p1});
    // 2. spike at 0
    std::vector<float> v2(V, -1.0f);
    v2[0] = 5.0f;
    logits.push_back(std::move(v2));
    cases.push_back({"spike-zero", 0});
    // 3. spike at last
    std::vector<float> v3(V, -2.0f);
    v3[V - 1] = 7.0f;
    logits.push_back(std::move(v3));
    cases.push_back({"spike-last", V - 1});
    // 4. all equal -> first index wins
    std::vector<float> v4(V, 0.25f);
    logits.push_back(std::move(v4));
    cases.push_back({"all-equal", 0});
    // 5. double-max tie -> smaller index wins
    std::vector<float> v5(V, -3.0f);
    v5[99999] = 4.0f;
    v5[199999] = 4.0f;
    logits.push_back(std::move(v5));
    cases.push_back({"tie-pair", 99999});
  }

  std::string json = "{\"device\":\"B60\",\"vocab\":248320,\"results\":[";
  bool first = true, allok = true;
  double t1sum = 0, t2sum = 0;
  for (size_t c = 0; c < cases.size(); ++c) {
    // host reference (first-max)
    int ref = 0;
    for (int i = 1; i < V; ++i)
      if (logits[c][i] > logits[c][ref])
        ref = i;
    bool refok = (ref == cases[c].expect);
    q.memcpy(dL, logits[c].data(), (size_t)V * 4).wait();
    double ms1 = 0, ms2 = 0;
    int got = device_argmax(q, dL, V, dPV, dPI, dO, ms1, ms2);
    t1sum += ms1;
    t2sum += ms2;
    bool ok = (got == ref) && refok;
    allok &= ok;
    char b[320];
    std::snprintf(b, sizeof b,
                  "%s{\"case\":\"%s\",\"expect\":%d,\"got\":%d,\"stage1_ms\":"
                  "%.4f,\"stage2_ms\":%.4f,\"%s\":true}",
                  first ? "" : ",", cases[c].name, cases[c].expect, got, ms1,
                  ms2, ok ? "PASS" : "FAIL");
    json += b;
    first = false;
    std::printf("%-12s expect=%7d got=%7d s1=%.4fms s2=%.4fms %s\n",
                cases[c].name, cases[c].expect, got, ms1, ms2,
                ok ? "PASS" : "FAIL");
  }
  char tail[256];
  std::snprintf(tail, sizeof tail,
                "],\"host_xfer_bytes\":4,\"all_pass\":%s}",
                allok ? "true" : "false");
  json += tail;
  FILE *o = stdout;
  if (argc > 1) {
    o = std::fopen(argv[1], "w");
    if (!o)
      return 1;
  }
  std::fprintf(o, "%s\n", json.c_str());
  if (o != stdout)
    std::fclose(o);
  sycl::free(dL, q);
  sycl::free(dPV, q);
  sycl::free(dPI, q);
  sycl::free(dO, q);
  return allok ? 0 : 3;
}
