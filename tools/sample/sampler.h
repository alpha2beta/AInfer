// AInfer T7.1 host sampler: temperature, top-k, top-p, repetition penalty
// hook, seeded multinomial. Runs on the host over full logits (same
// architecture as the llama.cpp reference: GPU produces logits, CPU samples).
// temp <= 0 selects greedy argmax with first-max-wins ties (consistent with
// the T3.9 device path). Deterministic per seed (xorshift64star).
// Pure host C++ (no SYCL); validated by validate_t71 against statistical
// expectations, not by eyeballing text.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

struct SampleParams {
  float temp = 0.0f;   // <=0: greedy
  int top_k = 0;       // <=0: no top-k filter
  float top_p = 1.0f;  // >=1: no top-p filter
  float rep_penalty = 1.0f; // <=1: off; >1: HF-style frequency penalty
  uint64_t seed = 0;
};

// HF-style repetition penalty applied in place over a mutable logits copy:
// seen ids with positive logit are divided, negatives multiplied. No-op for
// penalty <= 1. Caller passes the generated prefix (prompt + output so far).
static inline void apply_rep_penalty(std::vector<float> &logits,
                                     const int *seen, int nseen,
                                     float penalty) {
  if (penalty <= 1.0f || nseen <= 0)
    return;
  int V = (int)logits.size();
  std::vector<char> hit(V, 0);
  for (int i = 0; i < nseen; ++i)
    if (seen[i] >= 0 && seen[i] < V)
      hit[seen[i]] = 1;
  for (int i = 0; i < V; ++i) {
    if (!hit[i])
      continue;
    if (logits[i] > 0)
      logits[i] /= penalty;
    else
      logits[i] *= penalty;
  }
}

static inline uint64_t xrs64(uint64_t &s) {
  s ^= s >> 12;
  s ^= s << 25;
  s ^= s >> 27;
  return s * 2685821657736338717ull;
}
static inline double xrunif(uint64_t &s) {
  // 53-bit uniform in [0,1)
  return (double)(xrs64(s) >> 11) * (1.0 / 9007199254740992.0);
}

// Returns selected token index. probs_out (optional, size V) receives the
// final renormalized distribution (zeros for filtered ids).
static inline int sample_token(const float *logits, int V,
                               const SampleParams &p, uint64_t &rng,
                               std::vector<float> *probs_out = nullptr) {
  if (p.temp <= 0.0f) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < V; ++i) {
      if (logits[i] > bv) {
        bv = logits[i];
        best = i;
      }
    }
    if (probs_out) {
      probs_out->assign(V, 0.0f);
      (*probs_out)[best] = 1.0f;
    }
    return best;
  }
  // Stable softmax in double precision.
  double mx = logits[0];
  for (int i = 1; i < V; ++i)
    mx = std::max(mx, (double)logits[i]);
  std::vector<double> prob(V);
  double se = 0;
  for (int i = 0; i < V; ++i) {
    prob[i] = std::exp(((double)logits[i] - mx) / p.temp);
    se += prob[i];
  }
  for (int i = 0; i < V; ++i)
    prob[i] /= se;
  // Top-k: keep the k largest (ties broken by lower index for determinism).
  std::vector<int> order(V);
  for (int i = 0; i < V; ++i)
    order[i] = i;
  if (p.top_k > 0 && p.top_k < V) {
    std::partial_sort(order.begin(), order.begin() + p.top_k, order.end(),
                      [&](int a, int b) {
                        return prob[a] > prob[b] ||
                               (prob[a] == prob[b] && a < b);
                      });
    std::vector<char> keep(V, 0);
    for (int i = 0; i < p.top_k; ++i)
      keep[order[i]] = 1;
    double s2 = 0;
    for (int i = 0; i < V; ++i) {
      if (!keep[i])
        prob[i] = 0;
      else
        s2 += prob[i];
    }
    for (int i = 0; i < V; ++i)
      prob[i] /= s2;
  }
  // Top-p: minimal prefix with cumulative mass >= p (sorted desc).
  if (p.top_p < 1.0f) {
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return prob[a] > prob[b] || (prob[a] == prob[b] && a < b);
    });
    double cum = 0;
    int cut = V;
    for (int i = 0; i < V; ++i) {
      cum += prob[order[i]];
      if (cum >= p.top_p) {
        cut = i + 1;
        break;
      }
    }
    if (cut < 1)
      cut = 1;
    std::vector<char> keep(V, 0);
    for (int i = 0; i < cut; ++i)
      keep[order[i]] = 1;
    double s3 = 0;
    for (int i = 0; i < V; ++i) {
      if (!keep[i])
        prob[i] = 0;
      else
        s3 += prob[i];
    }
    for (int i = 0; i < V; ++i)
      prob[i] /= s3;
  }
  double u = xrunif(rng);
  double cum = 0;
  int sel = V - 1;
  for (int i = 0; i < V; ++i) {
    cum += prob[i];
    if (u < cum) {
      sel = i;
      break;
    }
  }
  if (probs_out) {
    probs_out->resize(V);
    for (int i = 0; i < V; ++i)
      (*probs_out)[i] = (float)prob[i];
  }
  return sel;
}
