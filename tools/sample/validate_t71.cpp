// AInfer T7.1 sampler validation: statistical contract tests, not eyeballing.
//  1. greedy-consistency vs T3.9 fixtures (temp<=0 -> first-max-wins argmax)
//  2. determinism: identical seed -> identical draw sequence
//  3. top-k support: no draw outside the top-k set over 20k draws
//  4. top-p support: no draw outside the minimal nucleus over 20k draws
//  5. chi-square distribution match on handcrafted V=8 logits (temp 1.0, 0.5)
//  6. temperature limits: tiny temp -> argmax; huge temp -> uniform
// Usage: validate_t71 [report.json]
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "sampler.h"

static int failures = 0;
static void check(bool ok, const char *name, const std::string &detail = "") {
  std::printf("%s %-22s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
  if (!ok)
    ++failures;
}

int main(int argc, char **argv) {
  const int V = 248320;
  // ---- 1. greedy fixtures (mirror T3.9) ----
  {
    std::vector<float> lg(V, -1.0f);
    uint64_t rng = 1;
    SampleParams gp{0.0f, 0, 1.0f, 1.0f, 0};
    lg[12345] = 10.0f;
    check(sample_token(lg.data(), V, gp, rng) == 12345, "greedy-spike");
    std::fill(lg.begin(), lg.end(), -1.0f);
    lg[0] = 5.0f;
    check(sample_token(lg.data(), V, gp, rng) == 0, "greedy-first");
    std::fill(lg.begin(), lg.end(), -1.0f);
    lg[V - 1] = 5.0f;
    check(sample_token(lg.data(), V, gp, rng) == V - 1, "greedy-last");
    std::fill(lg.begin(), lg.end(), 0.0f);
    check(sample_token(lg.data(), V, gp, rng) == 0, "greedy-allequal");
    std::fill(lg.begin(), lg.end(), -1.0f);
    lg[100] = lg[200] = 7.0f;
    check(sample_token(lg.data(), V, gp, rng) == 100, "greedy-tie-firstmax");
  }
  // ---- 2. determinism ----
  {
    const int W = 64;
    std::vector<float> lg(W);
    for (int i = 0; i < W; ++i)
      lg[i] = (float)(i % 7) - 3.0f;
    SampleParams p{0.8f, 0, 1.0f, 1.0f, 42};
    uint64_t r1 = 42, r2 = 42;
    bool same = true;
    for (int i = 0; i < 1000; ++i)
      if (sample_token(lg.data(), W, p, r1) !=
          sample_token(lg.data(), W, p, r2))
        same = false;
    check(same, "determinism-seed42");
  }
  // ---- 3. top-k support ----
  {
    const int W = 256;
    std::vector<float> lg(W);
    for (int i = 0; i < W; ++i)
      lg[i] = (float)((i * 7919) % 100) / 25.0f - 2.0f;
    SampleParams p{1.0f, 10, 1.0f, 1.0f, 7};
    uint64_t rng = 7;
    std::vector<int> order(W);
    for (int i = 0; i < W; ++i)
      order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return lg[a] > lg[b]; });
    std::vector<char> in_top(W, 0);
    for (int i = 0; i < 10; ++i)
      in_top[order[i]] = 1;
    bool okk = true;
    for (int i = 0; i < 20000; ++i)
      if (!in_top[sample_token(lg.data(), W, p, rng)])
        okk = false;
    check(okk, "topk-support-k10");
  }
  // ---- 4. top-p support ----
  {
    const int W = 64;
    std::vector<float> lg(W, -10.0f);
    lg[3] = 5.0f;
    lg[9] = 3.0f; // nucleus {3,9}: cum 0.88 then 1.0 >= 0.95 at T=1
    SampleParams p{1.0f, 0, 0.95f, 1.0f, 11};
    uint64_t rng = 11;
    std::vector<float> pr;
    sample_token(lg.data(), W, p, rng, &pr);
    std::vector<char> allowed(W, 0);
    for (int i = 0; i < W; ++i)
      if (pr[i] > 0)
        allowed[i] = 1;
    bool okp = true;
    rng = 11;
    for (int i = 0; i < 20000; ++i)
      if (!allowed[sample_token(lg.data(), W, p, rng)])
        okp = false;
    double mass = pr[3] + pr[9];
    char detail[128];
    std::snprintf(detail, sizeof detail, "nucleus-mass=%.4f", mass);
    check(okp && mass > 0.99, "topp-nucleus-p95", detail);
  }
  // ---- 5. chi-square distribution match (V=8) ----
  auto chi2 = [&](std::vector<float> lg, float temp, int N, double &stat,
                  double &crit) {
    SampleParams p{temp, 0, 1.0f, 1.0f, 1234};
    uint64_t rng = 1234;
    // expected distribution via the sampler's own probs path (single draw)
    std::vector<float> pr;
    sample_token(lg.data(), (int)lg.size(), p, rng, &pr);
    rng = 1234; // reset: first probs call consumed no RNG (softmax only)
    // NOTE: probs path consumes no RNG; draws below restart the stream.
    std::vector<int> cnt(lg.size(), 0);
    for (int i = 0; i < N; ++i)
      cnt[sample_token(lg.data(), (int)lg.size(), p, rng)]++;
    stat = 0;
    for (size_t i = 0; i < lg.size(); ++i) {
      double e = (double)pr[i] * N;
      if (e > 0)
        stat += (cnt[i] - e) * (cnt[i] - e) / e;
    }
    crit = 24.32; // df=7, p=0.001
    return stat < crit;
  };
  {
    std::vector<float> lg = {2.0f, 1.0f, 0.5f, 0.0f,
                             -0.5f, -1.0f, -2.0f, -3.0f};
    double stat, crit;
    bool okc = chi2(lg, 1.0f, 50000, stat, crit);
    char detail[128];
    std::snprintf(detail, sizeof detail, "chi2=%.2f crit=%.2f", stat, crit);
    check(okc, "chi2-temp1.0", detail);
    bool okc2 = chi2(lg, 0.5f, 50000, stat, crit);
    std::snprintf(detail, sizeof detail, "chi2=%.2f crit=%.2f", stat, crit);
    check(okc2, "chi2-temp0.5", detail);
  }
  // ---- 6. temperature limits ----
  {
    const int W = 32;
    std::vector<float> lg(W);
    for (int i = 0; i < W; ++i)
      lg[i] = (float)i / 4.0f;
    SampleParams cold{1e-3f, 0, 1.0f, 1.0f, 5};
    uint64_t rng = 5;
    bool okc = true;
    for (int i = 0; i < 500; ++i)
      if (sample_token(lg.data(), W, cold, rng) != W - 1)
        okc = false;
    check(okc, "temp-tiny-argmax");
    SampleParams hot{1e3f, 0, 1.0f, 1.0f, 9};
    rng = 9;
    std::vector<int> cnt(W, 0);
    for (int i = 0; i < 20000; ++i)
      cnt[sample_token(lg.data(), W, hot, rng)]++;
    double stat = 0, e = 20000.0 / W;
    for (int c : cnt)
      stat += (c - e) * (c - e) / e;
    char detail[128];
    std::snprintf(detail, sizeof detail, "chi2=%.2f crit=61.1(df=31,p=.001)",
                  stat);
    check(stat < 61.1, "temp-huge-uniform", detail);
  }

  // ---- 7. repetition penalty ----
  {
    const int W = 64;
    std::vector<float> lg(W, 0.0f);
    lg[5] = 3.0f;
    lg[9] = 3.0f;
    std::vector<float> base = lg, pen = lg;
    int seen[] = {5};
    apply_rep_penalty(pen, seen, 1, 1.5f);
    SampleParams p{1.0f, 0, 1.0f, 1.0f, 0};
    uint64_t r1 = 0, r2 = 0;
    std::vector<float> pb, pp;
    sample_token(base.data(), W, p, r1, &pb);
    sample_token(pen.data(), W, p, r2, &pp);
    char detail[128];
    std::snprintf(detail, sizeof detail, "mass5 %.4f->%.4f", pb[5], pp[5]);
    check(pp[5] < pb[5] && pp[9] > pb[9], "rep-penalty-shifts-mass", detail);
    std::vector<float> ident = lg;
    apply_rep_penalty(ident, seen, 1, 1.0f);
    check(ident == lg, "rep-penalty-unity-identity");
  }

  std::printf(failures == 0 ? "ALL-OK\n" : "FAILURES=%d\n", failures);
  if (argc > 1) {
    FILE *o = std::fopen(argv[1], "w");
    if (o) {
      std::fprintf(o,
                   "{\"sampler\":\"temp/top-k/top-p/xorshift64star\","
                   "\"tests\":14,\"failures\":%d,\"all_ok\":%s}\n",
                   failures, failures == 0 ? "true" : "false");
      std::fclose(o);
    }
  }
  return failures == 0 ? 0 : 1;
}
