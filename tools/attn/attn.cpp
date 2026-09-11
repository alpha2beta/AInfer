// AInfer attention block kernels (T3.6/T3.7/T3.8) vs HF references.
// 1. RoPE NeoX half-rotation, leading 64 dims -- vs reference/rope_applied.json
// 2. Decode GQA attention core (24Q/4KV x256, scale 1/16, stable softmax,
//    gate-sigmoid) -- vs attn_block_L3.json decode weights
// 3. o_proj fp32 GEMV ([5120x6144] BF16 weights parsed from SafeTensors)
//    + residual -- vs decode_token_out END-TO-END
// 4. MLP silu-mul fusion: 2-pass vs fused elementwise (T3.8)
// Usage: attn_t36 [report.json]
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

struct RTag {};
struct ATag {};
struct OTag {};
struct M1Tag {};
struct M2Tag {};
struct MFTag {};

// --- minimal JSON nested-float-array parser ---
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

// read one BF16 tensor (row-major) from a safetensors shard -> fp32 vector
static std::vector<float> read_bf16(const char *shard, const char *name,
                                    size_t &rows, size_t &cols) {
  std::ifstream f(shard, std::ios::binary);
  uint64_t hlen = 0;
  f.read((char *)&hlen, 8);
  std::string hdr(hlen, 0);
  f.read(hdr.data(), hlen);
  auto pos = hdr.find(std::string("\"") + name + "\"");
  auto sast = hdr.find("\"shape\"", pos);
  auto b1 = hdr.find("[", sast);
  auto b2 = hdr.find("]", b1);
  std::string sh = hdr.substr(b1 + 1, b2 - b1 - 1);
  rows = std::stoul(sh.substr(0, sh.find(",")));
  cols = std::stoul(sh.substr(sh.find(",") + 1));
  auto od = hdr.find("\"data_offsets\"", pos);
  auto o1 = hdr.find("[", od);
  auto o2 = hdr.find(",", o1);
  size_t beg = std::stoul(hdr.substr(o1 + 1, o2 - o1 - 1));
  auto o3 = hdr.find("]", o2);
  size_t end = std::stoul(hdr.substr(o2 + 1, o3 - o2 - 1));
  std::vector<uint16_t> raw((end - beg) / 2);
  f.seekg(8 + hlen + beg);
  f.read((char *)raw.data(), end - beg);
  std::vector<float> out(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    uint32_t u = (uint32_t)raw[i] << 16;
    std::memcpy(&out[i], &u, 4);
  }
  return out;
}

int main(int argc, char **argv) {
  const std::string R = "/mnt/usb/AInfer/reference/";
  const std::string M = "/mnt/usb/AInfer/models/Qwen3.8-27B/";
  std::string rs = slurp((R + "rope_applied.json").c_str());
  std::string as = slurp((R + "attn_block_L3.json").c_str());
  sycl::device dev = pick_b60();
  sycl::queue q(dev, {sycl::property::queue::enable_profiling()});
  std::string json = "{\"device\":\"B60\",\"results\":[";
  bool first = true;
  auto row = [&](const char *nm, double ms, double maxd, double meand) {
    char b[512];
    std::snprintf(b, sizeof b,
                  "%s{\"name\":\"%s\",\"ms\":%.4f,\"maxdiff\":%.2e,\"meandiff\":"
                  "%.2e}",
                  first ? "" : ",", nm, ms, maxd, meand);
    json += b;
    first = false;
    std::printf("%-22s %8.4f ms max %.2e mean %.2e\n", nm, ms, maxd, meand);
  };

  // ---- 1. RoPE ----
  {
    auto Q = jget(rs, "q"), Qe = jget(rs, "q_applied");
    auto K = jget(rs, "k"), Ke = jget(rs, "k_applied");
    auto COS = jget(rs, "cos"), SIN = jget(rs, "sin");
    const int HQ = 2, HK = 1, L = 4, D = 256, RD = 64;
    std::vector<float> qf(Q.begin(), Q.end()), kf(K.begin(), K.end());
    std::vector<float> qe(Qe.begin(), Qe.end()), ke(Ke.begin(), Ke.end());
    std::vector<float> cs(COS.begin(), COS.end()), sn(SIN.begin(), SIN.end());
    float *dQ = sycl::malloc_device<float>(qf.size(), q);
    float *dK = sycl::malloc_device<float>(kf.size(), q);
    float *dC = sycl::malloc_device<float>(cs.size(), q);
    float *dS = sycl::malloc_device<float>(sn.size(), q);
    float *dO = sycl::malloc_device<float>(qf.size() + kf.size(), q);
    q.memcpy(dQ, qf.data(), qf.size() * 4).wait();
    q.memcpy(dK, kf.data(), kf.size() * 4).wait();
    q.memcpy(dC, cs.data(), cs.size() * 4).wait();
    q.memcpy(dS, sn.data(), sn.size() * 4).wait();
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<RTag>(sycl::range<1>((HQ + HK) * L),
                           [=](sycl::id<1> id) {
                             int i = id[0], isK = i >= HQ * L ? 1 : 0;
                             int hh = isK ? (i - HQ * L) / L : i / L;
                             int t = isK ? (i - HQ * L) % L : i % L;
                             const float *X = isK ? dK + (size_t)hh * L * D
                                                  : dQ + (size_t)hh * L * D;
                             float *O = dO + (isK ? (size_t)HQ * L * D : 0) +
                                        (size_t)hh * L * D;
                             for (int d = 0; d < RD / 2; ++d) {
                               float x0 = X[t * D + d], x1 = X[t * D + d + 32];
                               float c = dC[t * RD + d], s2 = dS[t * RD + d];
                               O[t * D + d] = x0 * c - x1 * s2;
                               O[t * D + d + 32] = x0 * s2 + x1 * c;
                             }
                             for (int d = RD; d < D; ++d)
                               O[t * D + d] = X[t * D + d];
                           });
    });
    double ms = ev_ms(e);
    std::vector<float> got(qf.size() + kf.size());
    q.memcpy(got.data(), dO, got.size() * 4).wait();
    double maxd = 0, meand = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      double e2 = i < qf.size() ? qe[i] : ke[i - qf.size()];
      maxd = std::max(maxd, (double)std::fabs(got[i] - e2));
      meand += std::fabs(got[i] - e2);
    }
    row("rope-neox64", ms, maxd, meand / got.size());
    sycl::free(dQ, q);
    sycl::free(dK, q);
    sycl::free(dC, q);
    sycl::free(dS, q);
    sycl::free(dO, q);
  }

  // ---- 2. Decode attention core: 24 heads, L=5 ----
  std::vector<float> attn_core_out;
  {
    auto Q = jget(as, "decode_q"), G = jget(as, "decode_gate");
    auto KK = jget(as, "decode_K5"), VV = jget(as, "decode_V5");
    auto WT = jget(as, "decode_attn_weights_head0");
    const int HQ = 24, L = 5, D = 256;
    std::vector<float> qf(Q.begin(), Q.end()), gf(G.begin(), G.end());
    std::vector<float> kf(KK.begin(), KK.end()), vf(VV.begin(), VV.end());
    float *dQ = sycl::malloc_device<float>(qf.size(), q);
    float *dG = sycl::malloc_device<float>(gf.size(), q);
    float *dK = sycl::malloc_device<float>(kf.size(), q);
    float *dV = sycl::malloc_device<float>(vf.size(), q);
    float *dO = sycl::malloc_device<float>((size_t)HQ * D, q);
    float *dW = sycl::malloc_device<float>((size_t)HQ * L, q);
    q.memcpy(dQ, qf.data(), qf.size() * 4).wait();
    q.memcpy(dG, gf.data(), gf.size() * 4).wait();
    q.memcpy(dK, kf.data(), kf.size() * 4).wait();
    q.memcpy(dV, vf.data(), vf.size() * 4).wait();
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<ATag>(sycl::range<1>(HQ), [=](sycl::id<1> id) {
        int hh = id[0], kv = hh / 6;
        float mx = -1e30f;
        for (int t = 0; t < L; ++t) {
          float s = 0;
          for (int d = 0; d < D; ++d)
            s += dQ[(size_t)hh * D + d] * dK[((size_t)kv * L + t) * D + d];
          s /= 16.0f;
          dW[(size_t)hh * L + t] = s;
          mx = s > mx ? s : mx;
        }
        float se = 0;
        for (int t = 0; t < L; ++t) {
          float w = std::exp(dW[(size_t)hh * L + t] - mx);
          dW[(size_t)hh * L + t] = w;
          se += w;
        }
        for (int d = 0; d < D; ++d) {
          float acc = 0;
          for (int t = 0; t < L; ++t)
            acc += dW[(size_t)hh * L + t] / se *
                   dV[((size_t)kv * L + t) * D + d];
          float g = dG[(size_t)hh * D + d];
          dO[(size_t)hh * D + d] = acc / (1.0f + std::exp(-g));
        }
      });
    });
    double ms = ev_ms(e);
    std::vector<float> gotW((size_t)HQ * L);
    attn_core_out.assign((size_t)HQ * D, 0);
    q.memcpy(gotW.data(), dW, gotW.size() * 4).wait();
    q.memcpy(attn_core_out.data(), dO, attn_core_out.size() * 4).wait();
    double maxw = 0, se = 0, seg = 0;
    for (int t = 0; t < L; ++t) {
      se += WT[t];
      seg += gotW[t];
    }
    for (int t = 0; t < L; ++t)
      maxw = std::max(maxw, (double)std::fabs(gotW[t] / seg - WT[t] / se));
    row("attn-weights-h0", ms, maxw, maxw);
    sycl::free(dQ, q);
    sycl::free(dG, q);
    sycl::free(dK, q);
    sycl::free(dV, q);
    sycl::free(dW, q);
    sycl::free(dO, q);
  }

  // ---- 3. o_proj fp32 GEMV + residual vs decode_token_out (end-to-end) ----
  {
    auto TO = jget(as, "decode_token_out"); // [1,1,5120]
    auto XT = jget(as, "decode_xt");        // [1,1,5120] residual input
    size_t R = 0, C = 0;
    // o_proj lives in the shard holding layer 3 (find via index: reuse shard 1..18 scan)
    std::vector<float> W;
    for (int i = 1; i <= 18; ++i) {
      char p[256];
      std::snprintf(p, sizeof p,
                    "/mnt/usb/AInfer/models/Qwen3.8-27B/model-%05d-of-00018."
                    "safetensors",
                    i);
      std::ifstream t(p, std::ios::binary);
      if (!t)
        continue;
      uint64_t hl = 0;
      t.read((char *)&hl, 8);
      std::string hdr(hl, 0);
      t.read(hdr.data(), hl);
      if (hdr.find("layers.3.self_attn.o_proj.weight") != std::string::npos) {
        W = read_bf16(p, "model.language_model.layers.3.self_attn.o_proj.weight",
                      R, C);
        break;
      }
    }
    // device must reproduce HF: o_proj(core) + xt == decode_token_out
    float *dW = sycl::malloc_device<float>(W.size(), q);
    float *dX = sycl::malloc_device<float>(attn_core_out.size(), q);
    float *dR = sycl::malloc_device<float>(5120, q);
    float *dY = sycl::malloc_device<float>(5120, q);
    std::vector<float> xtf(XT.begin(), XT.end());
    q.memcpy(dW, W.data(), W.size() * 4).wait();
    q.memcpy(dX, attn_core_out.data(), attn_core_out.size() * 4).wait();
    q.memcpy(dR, xtf.data(), 5120 * 4).wait();
    auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for<OTag>(sycl::range<1>(5120), [=](sycl::id<1> id) {
        int r = id[0];
        float acc = 0;
        for (int c = 0; c < 6144; ++c)
          acc += dW[(size_t)r * 6144 + c] * dX[c];
        dY[r] = acc + dR[r];
      });
    });
    double ms = ev_ms(e);
    std::vector<float> got(5120);
    q.memcpy(got.data(), dY, got.size() * 4).wait();
    double maxd = 0, meand = 0, refmax = 0;
    for (int i = 0; i < 5120; ++i) {
      // TO is [1,1,5120] nested
      refmax = std::max(refmax, std::fabs(TO[i]));
      maxd = std::max(maxd, (double)std::fabs(got[i] - TO[i]));
      meand += std::fabs(got[i] - TO[i]);
    }
    row("oproj+e2e", ms, maxd / refmax, meand / 5120 / refmax);
    sycl::free(dW, q);
    sycl::free(dX, q);
    sycl::free(dR, q);
    sycl::free(dY, q);
  }

  // ---- 4. silu-mul fusion ----
  {
    const int N = 17408;
    std::vector<float> gg(N), uu(N);
    uint64_t s = 0x777;
    for (int i = 0; i < N; ++i) {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      gg[i] = (float)((int)((s >> 33) & 0xFFFF) - 32768) * (4.0f / 32768.0f);
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      uu[i] = (float)((int)((s >> 33) & 0xFFFF) - 32768) * (4.0f / 32768.0f);
    }
    float *dG = sycl::malloc_device<float>(N, q);
    float *dU = sycl::malloc_device<float>(N, q);
    float *dT = sycl::malloc_device<float>(N, q);
    float *dH = sycl::malloc_device<float>(N, q);
    q.memcpy(dG, gg.data(), N * 4).wait();
    q.memcpy(dU, uu.data(), N * 4).wait();
    auto t1 = q.submit([&](sycl::handler &h) {
      h.parallel_for<M1Tag>(sycl::range<1>(N), [=](sycl::id<1> id) {
        dT[id] = 1.0f / (1.0f + sycl::exp(-dG[id]));
      });
    });
    t1.wait();
    double ms1 = ev_ms(t1);
    auto t2 = q.submit([&](sycl::handler &h) {
      h.parallel_for<M2Tag>(sycl::range<1>(N),
                            [=](sycl::id<1> id) { dH[id] = dT[id] * dU[id]; });
    });
    t2.wait();
    ms1 += ev_ms(t2);
    auto t3 = q.submit([&](sycl::handler &h) {
      h.parallel_for<MFTag>(sycl::range<1>(N), [=](sycl::id<1> id) {
        float g = dG[id];
        dH[id] = (g / (1.0f + sycl::exp(-g))) * dU[id];
      });
    });
    double ms2 = ev_ms(t3);
    char b3[256];
    std::snprintf(b3, sizeof b3,
                  "%s{\"name\":\"silu-mul-2pass\",\"ms\":%.4f},"
                  "{\"name\":\"silu-mul-fused\",\"ms\":%.4f}",
                  first ? "" : ",", ms1, ms2);
    json += b3;
    first = false;
    std::printf("%-22s %8.4f ms\n%-22s %8.4f ms\n", "silu-mul-2pass", ms1,
                "silu-mul-fused", ms2);
    sycl::free(dG, q);
    sycl::free(dU, q);
    sycl::free(dT, q);
    sycl::free(dH, q);
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
