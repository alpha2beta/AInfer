// AInfer SSM linear-attention kernels (T3.7 remainder): decode path.
// 1. causal depthwise conv1d update (k=4) + SiLU, validated vs host loop.
// 2. recurrent gated-delta step, 1 WI/head, fp32 S[128][128]:
//    l2norm q/k, scale, S*=exp(g), delta update, out=S^T q.
//    Validated vs HF reference/lin_block_L0.json head-0 vectors.
// Usage: ssm_t37 [report.json]
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

struct CvTag {};
struct RecTag {};
struct Rec48Tag {};

// --- minimal JSON nested-float-array parser (same pattern as attn_t36) ---
static const char *g_p;
static void skip() {
  while (*g_p == ' ' || *g_p == '\n' || *g_p == '\r' || *g_p == '\t' ||
         *g_p == ',' || *g_p == ':')
    ++g_p;
}
static double pnum() {
  skip();
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
    skip();
    if (*g_p != '[')
      return;
    ++g_p;
    while (true) {
      skip();
      if (*g_p == ']') {
        ++g_p;
        break;
      }
      if (*g_p == '[')
        rec();
      else
        out.push_back(pnum());
      skip();
    }
  };
  rec();
  return out;
}
static std::string slurp(const char *path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
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
static double ev_ms(sycl::event e) {
  e.wait();
  return (double)(e.get_profiling_info<
                      sycl::info::event_profiling::command_end>() -
                  e.get_profiling_info<
                      sycl::info::event_profiling::command_start>()) *
         1e-6;
}

int main(int argc, char **argv) {
  sycl::device dev = pick_b60();
  sycl::queue q(dev, {sycl::property::queue::enable_profiling()});
  std::string json = "{\"device\":\"B60\",\"results\":[";
  bool first = true;
  auto row = [&](const char *nm, double ms, double maxd, double meand,
                 const char *extra = "") {
    char b[640];
    std::snprintf(b, sizeof b,
                  "%s{\"name\":\"%s\",\"ms\":%.4f,\"maxdiff\":%.2e,\"meandiff\":"
                  "%.2e%s}",
                  first ? "" : ",", nm, ms, maxd, meand, extra);
    json += b;
    first = false;
    std::printf("%-24s %8.4f ms max %.2e mean %.2e %s\n", nm, ms, maxd, meand,
                extra);
  };

  // ---- 1. conv update: C=2048, k=4, silu; host-loop reference ----
  {
    const int C = 2048, K = 4;
    std::vector<float> W(C * K), S(C * (K - 1)), X(C), ref(C);
    uint64_t s = 0x50a;
    auto rnd = [&]() {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (0.02f / 32768.0f);
    };
    for (auto &v : W)
      v = rnd();
    for (auto &v : S)
      v = rnd();
    for (auto &v : X)
      v = rnd() * 4;
    for (int c = 0; c < C; ++c) {
      float acc = 0;
      acc += S[c * 3 + 0] * W[c * 4 + 0];
      acc += S[c * 3 + 1] * W[c * 4 + 1];
      acc += S[c * 3 + 2] * W[c * 4 + 2];
      acc += X[c] * W[c * 4 + 3];
      ref[c] = acc / (1.0f + std::exp(-acc));
    }
    float *dW = sycl::malloc_device<float>(W.size(), q);
    float *dS = sycl::malloc_device<float>(S.size(), q);
    float *dX = sycl::malloc_device<float>(C, q);
    float *dY = sycl::malloc_device<float>(C, q);
    q.memcpy(dW, W.data(), W.size() * 4).wait();
    q.memcpy(dS, S.data(), S.size() * 4).wait();
    q.memcpy(dX, X.data(), C * 4).wait();
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<CvTag>(sycl::range<1>(C), [=](sycl::id<1> id) {
        int c = id[0];
        float acc = dS[c * 3 + 0] * dW[c * 4 + 0] +
                    dS[c * 3 + 1] * dW[c * 4 + 1] +
                    dS[c * 3 + 2] * dW[c * 4 + 2] + dX[c] * dW[c * 4 + 3];
        dY[c] = acc / (1.0f + sycl::exp(-acc));
        dS[c * 3 + 0] = dS[c * 3 + 1];
        dS[c * 3 + 1] = dS[c * 3 + 2];
        dS[c * 3 + 2] = dX[c];
      });
    });
    double ms = ev_ms(e);
    std::vector<float> got(C);
    q.memcpy(got.data(), dY, C * 4).wait();
    double maxd = 0, meand = 0;
    for (int i = 0; i < C; ++i) {
      maxd = std::max(maxd, (double)std::fabs(got[i] - ref[i]));
      meand += std::fabs(got[i] - ref[i]);
    }
    row("conv-update-k4", ms, maxd, meand / C);
    sycl::free(dW, q);
    sycl::free(dS, q);
    sycl::free(dX, q);
    sycl::free(dY, q);
  }

  // ---- 2. recurrent step, HF head-0 vectors ----
  {
    std::string js = slurp("/mnt/usb/AInfer/reference/lin_block_L0.json");
    auto J = [&](const char *k) {
      auto v = jget(js, k);
      return std::vector<float>(v.begin(), v.end());
    };
    std::vector<float> qh = J("decode_q_h0"), kh = J("decode_k_h0"),
                       vh = J("decode_v_h0"), Sh = J("S_head0"),
                       core = J("decode_core_h0_full");
    std::vector<float> gg = J("decode_g48"), bb = J("decode_beta48");
    float g0 = gg[0], b0 = bb[0];
    const int D = 128;
    float *dQ = sycl::malloc_device<float>(D, q);
    float *dK = sycl::malloc_device<float>(D, q);
    float *dV = sycl::malloc_device<float>(D, q);
    float *dS = sycl::malloc_device<float>(D * D, q);
    float *dO = sycl::malloc_device<float>(D, q);
    q.memcpy(dQ, qh.data(), D * 4).wait();
    q.memcpy(dK, kh.data(), D * 4).wait();
    q.memcpy(dV, vh.data(), D * 4).wait();
    q.memcpy(dS, Sh.data(), D * D * 4).wait();
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<RecTag>(sycl::range<1>(1), [=](sycl::id<1>) {
        float qq[128], kk[128];
        for (int i = 0; i < D; ++i) {
          qq[i] = dQ[i];
          kk[i] = dK[i];
        }
        // l2norm q,k (eps 1e-6) + q scale 1/sqrt(128)
        float sq = 0, sk = 0;
        for (int i = 0; i < D; ++i) {
          sq += qq[i] * qq[i];
          sk += kk[i] * kk[i];
        }
        float iq = 1.0f / sycl::sqrt(sq + 1e-6f) * 0.0883883476f;
        float ik = 1.0f / sycl::sqrt(sk + 1e-6f);
        for (int i = 0; i < D; ++i) {
          qq[i] *= iq;
          kk[i] *= ik;
        }
        float gt = sycl::exp(g0);
        for (int i = 0; i < D * D; ++i)
          dS[i] *= gt;
        float kv[128] = {0}, dl[128];
        // S is [k][v] row-major: S[k*D+v]. kv[v] = sum_k S[k][v]*k[k]
        for (int v = 0; v < D; ++v) {
          float s = 0;
          for (int k = 0; k < D; ++k)
            s += dS[k * D + v] * kk[k];
          kv[v] = s;
        }
        for (int v = 0; v < D; ++v)
          dl[v] = (dV[v] - kv[v]) * b0;
        for (int k = 0; k < D; ++k)
          for (int v = 0; v < D; ++v)
            dS[k * D + v] += kk[k] * dl[v];
        for (int v = 0; v < D; ++v) {
          float s = 0;
          for (int k = 0; k < D; ++k)
            s += dS[k * D + v] * qq[k];
          dO[v] = s;
        }
        (void)kv;
      });
    });
    double ms = ev_ms(e);
    std::vector<float> got(D);
    q.memcpy(got.data(), dO, D * 4).wait();
    double maxd = 0, meand = 0, cmax = 0;
    for (int i = 0; i < D; ++i) {
      cmax = std::max(cmax, (double)std::fabs(core[i]));
      maxd = std::max(maxd, (double)std::fabs(got[i] - core[i]));
      meand += std::fabs(got[i] - core[i]);
    }
    char extra[128];
    std::snprintf(extra, sizeof extra, ",\"rel\":%.2e", maxd / cmax);
    row("recurrent-h0", ms, maxd, meand / D, extra);
    sycl::free(dQ, q);
    sycl::free(dK, q);
    sycl::free(dV, q);
    sycl::free(dS, q);
    sycl::free(dO, q);
  }

  // ---- 3. 48-head throughput (synthetic, self-consistent) ----
  {
    const int H = 48, D = 128;
    std::vector<float> S(H * D * D, 0.01f), Q(H * D, 0.1f), K(H * D, 0.1f),
        V(H * D, 0.1f), G(H, -2.0f), B(H, 0.5f);
    float *dS = sycl::malloc_device<float>(S.size(), q);
    float *dQ = sycl::malloc_device<float>(Q.size(), q);
    float *dK = sycl::malloc_device<float>(K.size(), q);
    float *dV = sycl::malloc_device<float>(V.size(), q);
    float *dG = sycl::malloc_device<float>(H, q);
    float *dB = sycl::malloc_device<float>(H, q);
    float *dO = sycl::malloc_device<float>(H * D, q);
    q.memcpy(dS, S.data(), S.size() * 4).wait();
    q.memcpy(dQ, Q.data(), Q.size() * 4).wait();
    q.memcpy(dK, K.data(), K.size() * 4).wait();
    q.memcpy(dV, V.data(), V.size() * 4).wait();
    q.memcpy(dG, G.data(), H * 4).wait();
    q.memcpy(dB, B.data(), H * 4).wait();
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<Rec48Tag>(sycl::range<1>(H), [=](sycl::id<1> id) {
        int hh = id[0];
        float *Sh = dS + (size_t)hh * D * D;
        float *qh = dQ + (size_t)hh * D, *kh = dK + (size_t)hh * D,
              *vh = dV + (size_t)hh * D;
        float gt = sycl::exp(dG[hh]), bt = dB[hh];
        for (int i = 0; i < D * D; ++i)
          Sh[i] *= gt;
        float kv[128];
        for (int v = 0; v < D; ++v) {
          float s = 0;
          for (int k = 0; k < D; ++k)
            s += Sh[k * D + v] * kh[k];
          kv[v] = s;
        }
        for (int v = 0; v < D; ++v)
          kv[v] = (vh[v] - kv[v]) * bt; // reuse as delta
        for (int k = 0; k < D; ++k)
          for (int v = 0; v < D; ++v)
            Sh[k * D + v] += kh[k] * kv[v];
        float *oh = dO + (size_t)hh * D;
        for (int v = 0; v < D; ++v) {
          float s = 0;
          for (int k = 0; k < D; ++k)
            s += Sh[k * D + v] * qh[k];
          oh[v] = s;
        }
      });
    });
    double ms = ev_ms(e);
    // traffic: S 3MB/head-group read+write => 48*3MB*2 = 288MB
    double gbs = 2.0 * H * D * D * 4 / (ms * 1e-3) / 1e9;
    char extra[128];
    std::snprintf(extra, sizeof extra, ",\"gbs\":%.1f", gbs);
    row("recurrent-48h", ms, -1, -1, extra);
    sycl::free(dS, q);
    sycl::free(dQ, q);
    sycl::free(dK, q);
    sycl::free(dV, q);
    sycl::free(dG, q);
    sycl::free(dB, q);
    sycl::free(dO, q);
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
