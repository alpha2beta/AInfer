// T5.3 L0 kernel module: decode-exact kernels as named functors for stable
// SPIR-V entry points (launched from raw L0 lists by the *_replay harnesses).
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace esimd = sycl::ext::intel::esimd;
namespace mx = sycl::ext::oneapi::experimental::matrix;

// BF16 KV helpers (shared by KvAppend/AttnCore; RNE both directions).
static inline uint16_t kv_f32_to_bf16(float x) {
  uint32_t u;
  __builtin_memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static inline float kv_bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float x;
  __builtin_memcpy(&x, &u, 4);
  return x;
}

struct SiluMul {
  float *G;
  float *U;
  float *Z;
  void operator()(sycl::id<1> id) const {
    float g = G[id[0]];
    Z[id[0]] = (g / (1.0f + sycl::exp(-g))) * U[id[0]];
  }
};

// Forces device-image codegen for the SPIR-V bundle (never called at runtime;
// the L0 harness launches the extracted entry point directly).
void launch_silumul(sycl::queue &q, float *G, float *U, float *Z, int N) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(N), SiluMul{G, U, Z});
  }).wait();
}

// T5.4: position-dependent kernel driven ONLY by a device-visible control
// block. Layout matches DecodeControl{token_id,position,active_length,
// selected_token}; the kernel reads Ctrl[1] (position). Kernel arguments
// (Out/In/Ctrl addresses) stay fixed across replays — per-step mutation is
// control contents only.
struct ControlAdd {
  float *Out;
  const float *In;
  const int *Ctrl;
  void operator()(sycl::id<1> id) const {
    Out[id[0]] = In[id[0]] + (float)Ctrl[1];
  }
};

void launch_controladd(sycl::queue &q, float *Out, const float *In,
                       const int *Ctrl, int N) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(N), ControlAdd{Out, In, Ctrl});
  }).wait();
}

// T5.3 layer-class prototype: decode-exact RMSNorm (1+w, eps 1e-6, single row)
// + residual-add with control-block position bias. Together with SiluMul they
// form a 3-kernel recorded list: norm -> silu-mul -> residual, the same
// subgraph shape as a decode MLP tail (modulo the GEMVs, which stay SYCL
// until the raw-L0 GEMV port lands).
struct RMSNormW {
  float *Y;
  const float *X;
  const float *W;
  int N;
  sycl::local_accessor<double, 1> PS; // 256 partial sums (double tree)
  // T6.1: 256 WIs share one row (N=5120: 20 strided elements each); pairwise
  // double-precision SLM reduction (argmax pattern), then parallel normalize.
  // Tree-vs-sequential summation differs ~1 ulp — same class as decode's own
  // double accumulation. Scalar fallback for any other N (uniform branch).
  // Launch: 1 group x 256 WIs; set the local arg by size (256 doubles).
  void operator()(sycl::nd_item<1> it) const {
    int lid = (int)it.get_local_id(0);
    if (N != 5120) {
      if (lid == 0) {
        double ss = 0;
        for (int j = 0; j < N; ++j) {
          float v = X[j];
          ss += (double)v * v;
        }
        float inv = 1.0f / sycl::sqrt((float)(ss / N) + 1e-6f);
        for (int j = 0; j < N; ++j)
          Y[j] = X[j] * inv * (1.0f + W[j]);
      }
      return;
    }
    double ss = 0;
    for (int j = lid; j < 5120; j += 256) {
      float v = X[j];
      ss += (double)v * v;
    }
    PS[lid] = ss;
    it.barrier(sycl::access::fence_space::local_space);
    for (int st = 128; st > 0; st >>= 1) {
      if (lid < st)
        PS[lid] += PS[lid + st];
      it.barrier(sycl::access::fence_space::local_space);
    }
    double tot = PS[0];
    float inv = 1.0f / sycl::sqrt((float)(tot / 5120) + 1e-6f);
    for (int j = lid; j < 5120; j += 256)
      Y[j] = X[j] * inv * (1.0f + W[j]);
  }
};

struct ResAdd {
  float *R;
  const float *X;
  const float *Z;
  const int *Ctrl;
  int H;
  void operator()(sycl::id<1> id) const {
    R[id[0]] = X[id[0]] + Z[id[0] % H] + (float)Ctrl[1];
  }
};

void launch_layer_tail(sycl::queue &q, float *Y, const float *X,
                       const float *W, int N, float *Z, float *G, float *U,
                       int I, float *R, const int *Ctrl) {
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<double, 1> ps(256, h);
    h.parallel_for(sycl::nd_range<1>(256, 256), RMSNormW{Y, X, W, N, ps});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(I), SiluMul{G, U, Z});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(N), ResAdd{R, X, Z, Ctrl, N});
  }).wait();
}

// T5.3 GEMV port: decode-exact INT4 dp4a GEMV (one row per WI, per-group-128
// weight scales + per-group activation scales). Body mirrors decode.cpp gemvA
// / gemv.cpp pathC instruction-for-instruction; any divergence is a bug.
// Launch with 1 WI per L0 group (no cross-WI communication).
struct Int4Gemv {
  float *Y;
  const uint8_t *P;
  const uint16_t *S;
  const int8_t *XQ;
  const float *SQ;
  int K;
  void operator()(sycl::id<1> id) const SYCL_ESIMD_KERNEL {
    int m = id[0];
    float acc = 0;
    int Gg = K / 128;
    for (int g = 0; g < Gg; ++g) {
      uint32_t ub = S[(size_t)m * Gg + g];
      ub <<= 16;
      float sc;
      __builtin_memcpy(&sc, &ub, 4);
      float sqg = SQ[g];
      size_t base = (size_t)m * K / 2 + (size_t)g * 64;
      esimd::simd<int, 8> acc8(0);
      for (int j = 0; j < 128; j += 32) {
        esimd::simd<unsigned char, 16> bytes =
            esimd::block_load<unsigned char, 16>(P + base + j / 2);
        esimd::simd<unsigned short, 32> v;
        v.template select<16, 2>(0) =
            esimd::convert<unsigned short>(bytes & 0xFu);
        v.template select<16, 2>(1) =
            esimd::convert<unsigned short>(bytes >> 4);
        esimd::simd<short, 32> sv = esimd::convert<short>(v);
        sv -= (sv & 8) << 1;
        esimd::simd<int8_t, 32> w8 = esimd::convert<int8_t>(sv);
        esimd::simd<int8_t, 32> xq =
            esimd::block_load<int8_t, 32>(XQ + g * 128 + j);
        esimd::simd<int, 8> w32 = w8.bit_cast_view<int>();
        esimd::simd<int, 8> x32 = xq.bit_cast_view<int>();
        acc8 = esimd::dp4a<int>(acc8, w32, x32);
      }
      int tmp[8];
      acc8.copy_to(tmp);
      int gs = 0;
      for (int u = 0; u < 8; ++u)
        gs += tmp[u];
      acc += (float)gs * sc * sqg;
    }
    Y[m] = acc;
  }
};

void launch_int4gemv(sycl::queue &q, float *Y, const uint8_t *P,
                     const uint16_t *S, const int8_t *XQ, const float *SQ,
                     int M, int K) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(M), Int4Gemv{Y, P, S, XQ, SQ, K});
  }).wait();
}

// T5.3 attention port: decode-exact GQA core + sigmoid gate (one full-attention
// half minus the QKV/o GEMVs, which are covered by Int4Gemv). Body mirrors
// decode.cpp's KGqa kernel: 24 Q heads over 4 KV heads (head_dim 256),
// causal scores over t < active_length, stable softmax, weighted V, gate.
// The loop bound comes from Ctrl[2] (active_length) — variable context in a
// recorded list with zero arg mutation, the T5.4/T5.3 integration point.
// Launch with 1 WI per L0 group.
struct AttnCore {
  float *Att;         // 24*256 out
  const float *Q;     // 24*256
  const uint16_t *Kc; // 4*TMAX*256 BF16 KV cache
  const uint16_t *Vc; // 4*TMAX*256
  const float *Gate; // 24*256
  const int *Ctrl;  // DecodeControl; [2] = active_length
  int TMAX;
  float *Wts; // 24*TMAX scratch (private wts[] cannot scale to 4K contexts)
  // Vectorized over d (256 = 8 simd<float,32> lanes): Q hoisted to registers.
  // BF16 rows decode by zero-extend + <<16 + bit_cast (EXACT reinterpret;
  // esimd::convert<uint16->float> would be a numeric conversion and is WRONG
  // here — first vector attempt shipped exactly that bug, outputs ~1e4 scale).
  // Horizontal sums fold by scalar extract (esimd::reduce miscompiles, T3.2).
  // Same members/layout/order; FMA reassociation only (harness tol absorbs).
  void operator()(sycl::id<1> id) const SYCL_ESIMD_KERNEL {
    int hh = id[0], kv = hh / 6, T = Ctrl[2];
    if (T > TMAX)
      T = TMAX; // clamp: active_length can never overrun caches
    float *wts = Wts + (size_t)hh * TMAX;
    const float *qh = Q + (size_t)hh * 256;
    esimd::simd<float, 32> q0 = esimd::block_load<float, 32>(qh + 0);
    esimd::simd<float, 32> q1 = esimd::block_load<float, 32>(qh + 32);
    esimd::simd<float, 32> q2 = esimd::block_load<float, 32>(qh + 64);
    esimd::simd<float, 32> q3 = esimd::block_load<float, 32>(qh + 96);
    esimd::simd<float, 32> q4 = esimd::block_load<float, 32>(qh + 128);
    esimd::simd<float, 32> q5 = esimd::block_load<float, 32>(qh + 160);
    esimd::simd<float, 32> q6 = esimd::block_load<float, 32>(qh + 192);
    esimd::simd<float, 32> q7 = esimd::block_load<float, 32>(qh + 224);
    auto bf16row = [](const uint16_t *p) SYCL_ESIMD_FUNCTION {
      esimd::simd<uint32_t, 32> u =
          esimd::convert<uint32_t>(esimd::block_load<uint16_t, 32>(p));
      u <<= 16;
      return u.template bit_cast_view<float>();
    };
    float mx = -1e30f;
    for (int t = 0; t < T; ++t) {
      const uint16_t *kr = Kc + ((size_t)t * 4 + kv) * 256;
      esimd::simd<float, 32> k0 = bf16row(kr + 0);
      esimd::simd<float, 32> k1 = bf16row(kr + 32);
      esimd::simd<float, 32> k2 = bf16row(kr + 64);
      esimd::simd<float, 32> k3 = bf16row(kr + 96);
      esimd::simd<float, 32> k4 = bf16row(kr + 128);
      esimd::simd<float, 32> k5 = bf16row(kr + 160);
      esimd::simd<float, 32> k6 = bf16row(kr + 192);
      esimd::simd<float, 32> k7 = bf16row(kr + 224);
      esimd::simd<float, 32> p0 = q0 * k0, p1 = q1 * k1, p2 = q2 * k2,
                             p3 = q3 * k3, p4 = q4 * k4, p5 = q5 * k5,
                             p6 = q6 * k6, p7 = q7 * k7;
      float s = 0;
#pragma unroll
      for (int l = 0; l < 32; ++l)
        s += (float)p0[l] + (float)p1[l] + (float)p2[l] + (float)p3[l] +
             (float)p4[l] + (float)p5[l] + (float)p6[l] + (float)p7[l];
      s /= 16.0f;
      wts[t] = s;
      mx = s > mx ? s : mx;
    }
    float se = 0;
    for (int t = 0; t < T; ++t) {
      float w = sycl::exp(wts[t] - mx);
      wts[t] = w;
      se += w;
    }
    float inv_se = 1.0f / se;
    esimd::simd<float, 32> a0(0), a1(0), a2(0), a3(0), a4(0), a5(0), a6(0),
        a7(0);
    for (int t = 0; t < T; ++t) {
      float w = wts[t] * inv_se;
      const uint16_t *vr = Vc + ((size_t)t * 4 + kv) * 256;
      a0 += w * bf16row(vr + 0);
      a1 += w * bf16row(vr + 32);
      a2 += w * bf16row(vr + 64);
      a3 += w * bf16row(vr + 96);
      a4 += w * bf16row(vr + 128);
      a5 += w * bf16row(vr + 160);
      a6 += w * bf16row(vr + 192);
      a7 += w * bf16row(vr + 224);
    }
    float *ah = Att + (size_t)hh * 256;
    const float *gh = Gate + (size_t)hh * 256;
    esimd::simd<float, 32> acc[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    for (int j = 0; j < 8; ++j) {
      esimd::simd<float, 32> g =
          esimd::block_load<float, 32>(gh + (size_t)j * 32);
      esimd::simd<float, 32> o(0);
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        float gl = (float)g[l];
        o[l] = (float)acc[j][l] / (1.0f + sycl::exp(-gl));
      }
      esimd::block_store<float, 32>(ah + (size_t)j * 32, o);
    }
  }
};

void launch_attncore(sycl::queue &q, float *Att, const float *Q,
                     const uint16_t *Kc, const uint16_t *Vc,
                     const float *Gate, const int *Ctrl, int TMAX,
                     float *Wts) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(24),
                   AttnCore{Att, Q, Kc, Vc, Gate, Ctrl, TMAX, Wts});
  }).wait();
}

// T5.3 SSM port (stateful core): 1D conv-k4 + SiLU with persistent 3-slot
// history, then the 48-head FP32 delta-rule recurrence with persistent
// 128x128 state per head. Bodies mirror decode.cpp KConv/KRec exactly; the
// stateless split/repeat, l2norm, and beta/g stages are bypassed (harness
// uploads their equivalents directly) — the recording risk lives in the
// persistent states, which is what this port proves. Launch 1 WI per L0 group.
struct SsmConv {
  float *Mx;       // CONVD conv output (silu)
  const float *QKV; // CONVD feed-forward input
  float *CS;       // CONVD*3 persistent history (zero-init)
  const float *CW; // CONVD*4 weights
  void operator()(sycl::id<1> id) const {
    int c = id[0];
    float acc = CS[c * 3 + 0] * CW[(size_t)c * 4 + 0] +
                CS[c * 3 + 1] * CW[(size_t)c * 4 + 1] +
                CS[c * 3 + 2] * CW[(size_t)c * 4 + 2] +
                QKV[c] * CW[(size_t)c * 4 + 3];
    Mx[c] = acc / (1.0f + sycl::exp(-acc));
    CS[c * 3 + 0] = CS[c * 3 + 1];
    CS[c * 3 + 1] = CS[c * 3 + 2];
    CS[c * 3 + 2] = QKV[c];
  }
};

struct SsmRecur {
  float *Out;      // 48*128 head outputs
  const float *Q;  // 48*128 (l2-normed, q-scaled)
  const float *K;  // 48*128 (l2-normed)
  const float *V;  // 48*128
  float *S;        // 48*128*128 persistent FP32 state (zero-init)
  const float *Bt; // 48 beta
  const float *Gt; // 48 log-decay (device stores pre-exp gating)
  int D;
  // T6.1: ESIMD-vectorized over the v (column) dimension in simd<float,32>
  // lanes. S is row-major, so fixed-k rows are contiguous; kv lives in 4
  // simd registers across steps 2-4 (no SLM, no array). Same members, same
  // arg layout, same operation order as the scalar version — only the
  // lane width changed. 1 WI per head preserved (L0 groups unchanged).
  void operator()(sycl::id<1> id) const SYCL_ESIMD_KERNEL {
    int hh = id[0];
    float *S0 = S + (size_t)hh * D * D;
    const float *qh = Q + (size_t)hh * 128, *kh = K + (size_t)hh * 128,
                *vh = V + (size_t)hh * 128;
    float gt = sycl::exp(Gt[hh]), bt = Bt[hh];
    for (int i = 0; i < D * D; i += 32) {
      esimd::simd<float, 32> r = esimd::block_load<float, 32>(S0 + i);
      r *= gt;
      esimd::block_store<float, 32>(S0 + i, r);
    }
    esimd::simd<float, 32> a0(0), a1(0), a2(0), a3(0);
    for (int k = 0; k < D; ++k) {
      float kk = kh[k];
      a0 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 0) * kk;
      a1 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 32) * kk;
      a2 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 64) * kk;
      a3 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 96) * kk;
    }
    esimd::simd<float, 32> v0 = esimd::block_load<float, 32>(vh + 0);
    esimd::simd<float, 32> v1 = esimd::block_load<float, 32>(vh + 32);
    esimd::simd<float, 32> v2 = esimd::block_load<float, 32>(vh + 64);
    esimd::simd<float, 32> v3 = esimd::block_load<float, 32>(vh + 96);
    a0 = (v0 - a0) * bt;
    a1 = (v1 - a1) * bt;
    a2 = (v2 - a2) * bt;
    a3 = (v3 - a3) * bt;
    for (int k = 0; k < D; ++k) {
      float kk = kh[k];
      esimd::simd<float, 32> r0 = esimd::block_load<float, 32>(S0 + (size_t)k * D + 0);
      esimd::simd<float, 32> r1 = esimd::block_load<float, 32>(S0 + (size_t)k * D + 32);
      esimd::simd<float, 32> r2 = esimd::block_load<float, 32>(S0 + (size_t)k * D + 64);
      esimd::simd<float, 32> r3 = esimd::block_load<float, 32>(S0 + (size_t)k * D + 96);
      r0 += a0 * kk;
      r1 += a1 * kk;
      r2 += a2 * kk;
      r3 += a3 * kk;
      esimd::block_store<float, 32>(S0 + (size_t)k * D + 0, r0);
      esimd::block_store<float, 32>(S0 + (size_t)k * D + 32, r1);
      esimd::block_store<float, 32>(S0 + (size_t)k * D + 64, r2);
      esimd::block_store<float, 32>(S0 + (size_t)k * D + 96, r3);
    }
    esimd::simd<float, 32> o0(0), o1(0), o2(0), o3(0);
    for (int k = 0; k < D; ++k) {
      float qq = qh[k];
      o0 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 0) * qq;
      o1 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 32) * qq;
      o2 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 64) * qq;
      o3 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 96) * qq;
    }
    float *oh = Out + (size_t)hh * 128;
    esimd::block_store<float, 32>(oh + 0, o0);
    esimd::block_store<float, 32>(oh + 32, o1);
    esimd::block_store<float, 32>(oh + 64, o2);
    esimd::block_store<float, 32>(oh + 96, o3);
  }
};

void launch_ssm(sycl::queue &q, float *Mx, const float *QKV, float *CS,
                const float *CW, int C, float *Out, const float *Q,
                const float *K, const float *V, float *S, const float *Bt,
                const float *Gt, int D) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(C), SsmConv{Mx, QKV, CS, CW});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(48),
                   SsmRecur{Out, Q, K, V, S, Bt, Gt, D});
  }).wait();
}

// T5.3 RoPE/KV port: decode-exact NeoX half-rotation (leading 64 dims) over
// 24 Q + 4 K heads, then append into single-slot KV caches. Position comes
// from Ctrl[1]; KvAppend clamps it to keep the TMAX member live (AttnCore
// lesson) and the cache access in range. Launch 1 WI per L0 group.
struct RopeApply {
  float *Q; // 24*256, rotated in place
  float *K; // 4*256, rotated in place
  const float *Cos; // 64*TMAX
  const float *Sin;
  const int *Ctrl; // pos = Ctrl[1]
  void operator()(sycl::id<1> id) const {
    int i = id[0], pos = Ctrl[1];
    float *X = i < 24 ? Q + (size_t)i * 256 : K + (size_t)(i - 24) * 256;
    for (int d = 0; d < 32; ++d) {
      float x0 = X[d], x1 = X[d + 32];
      float c = Cos[pos * 64 + d], s = Sin[pos * 64 + d];
      X[d] = x0 * c - x1 * s;
      X[d + 32] = x0 * s + x1 * c;
    }
  }
};

struct KvAppend {
  uint16_t *Kc; // 4*TMAX*256 BF16 cache (T7.4 64K: halves footprint)
  uint16_t *Vc;
  const float *Kn; // 4*256 rotated keys
  const float *V;  // 4*256 values
  const int *Ctrl;
  int TMAX;
  void operator()(sycl::id<1> id) const {
    int i = id[0], hh = i / 256, d = i % 256, pos = Ctrl[1];
    int p = pos < TMAX ? pos : TMAX - 1;
    Kc[((size_t)p * 4 + hh) * 256 + d] =
        kv_f32_to_bf16(Kn[(size_t)hh * 256 + d]);
    Vc[((size_t)p * 4 + hh) * 256 + d] =
        kv_f32_to_bf16(V[(size_t)hh * 256 + d]);
  }
};

void launch_ropekv(sycl::queue &q, float *Q, float *K, const float *Cos,
                   const float *Sin, const int *Ctrl, uint16_t *Kc,
                   uint16_t *Vc, const float *Kn, const float *V, int TMAX) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(28), RopeApply{Q, K, Cos, Sin, Ctrl});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(4 * 256),
                   KvAppend{Kc, Vc, Kn, V, Ctrl, TMAX});
  }).wait();
}

// T5.3 argmax port: decode-exact two-stage SLM reduction (T3.9 design) over
// vocab 248320, first-max-wins ties. Local accessors are functor members;
// under raw L0 they are set as local-memory args (size + NULL). 64x256 then
// 1x256; launch with matching nd_range group counts.
struct ArgmaxS1 {
  const float *Logits;
  float *PV;
  int *PI;
  sycl::local_accessor<float, 1> BV;
  sycl::local_accessor<int, 1> BI;
  void operator()(sycl::nd_item<1> it) const {
    int lid = (int)it.get_local_id(0), gid = (int)it.get_group(0);
    float best = -INFINITY;
    int bidx = INT_MAX;
    for (int i = gid * 256 + lid; i < 248320; i += 64 * 256) {
      float v = Logits[i];
      if (v > best || (v == best && i < bidx)) {
        best = v;
        bidx = i;
      }
    }
    BV[lid] = best;
    BI[lid] = bidx;
    it.barrier(sycl::access::fence_space::local_space);
    for (int st = 128; st > 0; st >>= 1) {
      if (lid < st) {
        if (BV[lid + st] > BV[lid] ||
            (BV[lid + st] == BV[lid] && BI[lid + st] < BI[lid])) {
          BV[lid] = BV[lid + st];
          BI[lid] = BI[lid + st];
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
    }
    if (lid == 0) {
      PV[gid] = BV[0];
      PI[gid] = BI[0];
    }
  }
};

struct ArgmaxS2 {
  const float *PV;
  const int *PI;
  int *OutT;
  sycl::local_accessor<float, 1> BV;
  sycl::local_accessor<int, 1> BI;
  void operator()(sycl::nd_item<1> it) const {
    int lid = (int)it.get_local_id(0);
    if (lid < 64) {
      BV[lid] = PV[lid];
      BI[lid] = PI[lid];
    } else {
      BV[lid] = -INFINITY;
      BI[lid] = INT_MAX;
    }
    it.barrier(sycl::access::fence_space::local_space);
    for (int st = 128; st > 0; st >>= 1) {
      if (lid < st) {
        if (BV[lid + st] > BV[lid] ||
            (BV[lid + st] == BV[lid] && BI[lid + st] < BI[lid])) {
          BV[lid] = BV[lid + st];
          BI[lid] = BI[lid + st];
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
    }
    if (lid == 0)
      OutT[0] = BI[0];
  }
};

// T5.3 layer adoption: stateless linear-layer stages, mirroring the decode
// loop bodies (KMax/KQuant/KSpl2/KNorm2/KBG/KNG/KNG2/KRes/KRes2) so a full
// linear layer composes into one recorded list. Launch 1 WI per L0 group.
struct ScalesMax {
  const float *X;
  float *SQ;
  int G; // group count (guard keeps it live for the L0 arg layout)
  void operator()(sycl::id<1> id) const {
    int g = id[0];
    if (g >= G)
      return;
    float m = 0;
    for (int j = 0; j < 128; ++j) {
      float a = sycl::fabs(X[g * 128 + j]);
      m = a > m ? a : m;
    }
    float s = m / 127.0f;
    SQ[g] = s == 0 ? 1.0f : s;
  }
};

struct Quantize {
  const float *X;
  const float *SQ;
  int8_t *Q8;
  void operator()(sycl::id<1> id) const {
    float s = SQ[id[0] / 128];
    float v = X[id[0]] / s;
    int qi = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
    Q8[id[0]] = (int8_t)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
  }
};

struct SplitRepeat {
  const float *Mx; // CONVD conv output
  float *Q48, *K48, *V48; // 48*128 each
  void operator()(sycl::id<1> id) const {
    int i = id[0], hh = i / 128, d = i % 128, kh = hh / 3;
    Q48[i] = Mx[(size_t)kh * 128 + d];
    K48[i] = Mx[(size_t)(2048 + kh * 128) + d];
    V48[i] = Mx[(size_t)(4096 + hh * 128) + d];
  }
};

struct L2NormQK {
  float *Q48, *K48; // 48*128 each, normalized in place
  void operator()(sycl::id<1> id) const {
    int i = id[0];
    float *X = i < 48 ? Q48 + (size_t)i * 128 : K48 + (size_t)(i - 48) * 128;
    float ss = 0;
    for (int d = 0; d < 128; ++d)
      ss += X[d] * X[d];
    float inv = 1.0f / sycl::sqrt(ss + 1e-6f);
    for (int d = 0; d < 128; ++d)
      X[d] *= inv * (i < 48 ? 0.0883883476f : 1.0f);
  }
};

struct BetaG {
  float *Bt, *G48; // 48 each, out
  const float *B48, *A48; // 48 each, in
  const float *AL, *DT; // 48 each, A_log / dt_bias
  void operator()(sycl::id<1> id) const {
    int hh = id[0];
    Bt[hh] = 1.0f / (1.0f + sycl::exp(-B48[hh]));
    float sa = A48[hh] + DT[hh];
    float soft = sa > 20 ? sa : sycl::log(1.0f + sycl::exp(sa));
    G48[hh] = -sycl::exp(AL[hh]) * soft;
  }
};

struct NormGated {
  float *Att; // 6144 out
  const float *Mx; // 6144 SSM outputs
  const float *Z; // 6144 gate inputs
  const float *NG; // 128 norm weights
  const float *Bt; // 48 per-head inv-rms (reused buffer, decode parity)
  void operator()(sycl::id<1> id) const {
    int i = id[0], hh = i / 128, d = i % 128;
    float zv = Z[(size_t)hh * 128 + d];
    Att[i] = NG[d] * Mx[i] * Bt[hh] * (zv / (1.0f + sycl::exp(-zv)));
  }
};

struct RmsInv {
  float *Inv; // 48 out (per-head inv-rms into shared Bt buffer)
  const float *Mx; // 6144 SSM outputs
  void operator()(sycl::id<1> id) const {
    int hh = id[0];
    float ss = 0;
    for (int d = 0; d < 128; ++d) {
      float v = Mx[(size_t)hh * 128 + d];
      ss += v * v;
    }
    Inv[hh] = 1.0f / sycl::sqrt(ss / 128 + 1e-6f);
  }
};

struct ResAddF {
  float *Y;
  const float *A, *B;
  void operator()(sycl::id<1> id) const {
    Y[id[0]] = A[id[0]] + B[id[0]];
  }
};

// T5.3 full-attention layer adoption: QKV split + batched (1+w) norms for
// the 24-row Q and 4-row K norm stages. Launch 1 WI per L0 group (one row).
struct SplitQK {
  const float *Q16; // 12288 = 24 heads x (content 256 | gate 256)
  float *Qn;        // 6144 content
  float *Gate;      // 6144 gate
  void operator()(sycl::id<1> id) const {
    int i = id[0], hh = i / 256, d = i % 256;
    Qn[i] = Q16[(size_t)hh * 512 + d];
    Gate[i] = Q16[(size_t)hh * 512 + 256 + d];
  }
};

struct BatchNorm {
  float *Y;
  const float *X;
  const float *W; // 256 (shared across rows)
  int N;          // cols (256 here; member kept live by the guard)
  void operator()(sycl::id<1> id) const {
    int r = id[0];
    double ss = 0;
    for (int j = 0; j < N; ++j) {
      float v = X[(size_t)r * N + j];
      ss += (double)v * v;
    }
    float inv = 1.0f / sycl::sqrt((float)(ss / N) + 1e-6f);
    for (int j = 0; j < N; ++j)
      Y[(size_t)r * N + j] = X[(size_t)r * N + j] * inv * (1.0f + W[j]);
  }
};

// T7.4 chunked prefill: tiled INT4 DPAS GEMM (decode-exact MT4 design
// from tools/prefill/prefill.cpp gemmMT4). C[M][N] = A[M][K] (fp16) x B
// (INT4 layout-0 Bt[N][K], dequantized on the fly to f16), 8x16x16 DPAS,
// fp32 accumulate. K/TN/M are runtime members (DPAS tile shapes are fixed);
// one SG16 per 8x16 tile, M tiled x4 (B traffic /4). Launch: groups =
// (M/32)*(TN/16), size 16, reqd SG16, 3 SLM args (512h + 256h + 512f).
struct ChunkGemm {
  const sycl::half *A;
  const uint8_t *P;
  const uint16_t *S;
  float *C;
  int M, K, TN;
  sycl::local_accessor<sycl::half, 1> sA;
  sycl::local_accessor<sycl::half, 1> sB;
  sycl::local_accessor<float, 1> sC;
  // NOTE: no reqd_sub_group_size attribute (icpx rejects that spelling on
  // functor operator()); SG16 is forced structurally instead — L0 launches
  // groups of exactly 16 WIs, i.e. one hardware subgroup per group, and the
  // harness verifies numerics (a wrong SG size fails loudly, not silently).
  void operator()(sycl::nd_item<1> it) const {
    constexpr int TR = 8, TC = 16, UR = 4;
    auto sg = it.get_sub_group();
    int nBc = TN / 16;
    int gid = (int)it.get_group(0);
    int br = gid / nBc, bc = gid % nBc;
    if (br * 32 >= M)
      return; // partial final block guard (also keeps M live for L0 args)
    int lid = (int)sg.get_local_id()[0];
    int G = K / 128;
    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TR, 16,
                     mx::layout::dynamic>
        acc[UR];
    for (int r = 0; r < UR; ++r)
      mx::joint_matrix_fill(sg, acc[r], 0.0f);
    for (int kt = 0; kt < K; kt += TC) {
      for (int r = 0; r < UR; ++r)
        for (int u = 0; u < 8; ++u) {
          int idx = lid * 8 + u, rr = idx / TC, c = idx % TC;
          sA[r * 128 + rr * TC + c] =
              A[(br * TR * UR + r * TR + rr) * K + kt + c];
        }
      int g = kt / 128;
      for (int u = 0; u < 16; ++u) {
        int idx = lid * 16 + u, i = idx / 16, n2 = idx % 16;
        int n = bc * 16 + n2;
        size_t li = (size_t)n * K + kt + i;
        uint8_t b = P[li / 2];
        int nib = (li & 1) ? (b >> 4) : (b & 0xF);
        if (nib >= 8)
          nib -= 16;
        uint32_t ub = S[(size_t)n * G + g];
        ub <<= 16;
        float sc;
        __builtin_memcpy(&sc, &ub, 4);
        sB[i * 16 + n2] = sycl::half((float)nib * sc);
      }
      sg.barrier();
      for (int r = 0; r < UR; ++r) {
        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TR, TC,
                         mx::layout::row_major>
            ta;
        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TC, 16,
                         mx::layout::row_major>
            tb;
        sycl::multi_ptr<sycl::half, sycl::access::address_space::local_space>
            mpA(&sA[r * 128]);
        mx::joint_matrix_load(sg, ta, mpA, TC);
        mx::joint_matrix_load(
            sg, tb,
            sB.template get_multi_ptr<sycl::access::decorated::legacy>(), 16);
        mx::joint_matrix_mad(sg, acc[r], ta, tb, acc[r]);
      }
      sg.barrier();
    }
    for (int r = 0; r < UR; ++r) {
      mx::joint_matrix_store(
          sg, acc[r],
          sycl::multi_ptr<float, sycl::access::address_space::local_space>(
              &sC[r * 128]),
          16, mx::layout::row_major);
    }
    sg.barrier();
    for (int u = 0; u < 32; ++u) {
      int idx = lid * 32 + u, r = idx / 128, rest = idx % 128;
      int rr = rest / 16, c = rest % 16;
      C[(br * TR * UR + r * TR + rr) * TN + bc * 16 + c] =
          sC[r * 128 + rr * 16 + c];
    }
  }
};

void launch_chunkgemm(sycl::queue &q, const sycl::half *A, const uint8_t *P,
                      const uint16_t *S, float *C, int M, int K, int TN) {
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<sycl::half, 1> sA(128 * 4, h);
    sycl::local_accessor<sycl::half, 1> sB(256, h);
    sycl::local_accessor<float, 1> sC(128 * 4, h);
    h.parallel_for(sycl::nd_range<1>({(size_t)(M / 32) * (TN / 16) * 16}, {16}),
                   ChunkGemm{A, P, S, C, M, K, TN, sA, sB, sC});
  }).wait();
}

// T7.4 chunked prefill: causal attention over a token chunk against a cache
// prefix. Query m of the chunk (global pos = base+m) attends t = 0..base+m
// over the shared KV cache; gate applied like decode. Mirrors AttnCore's
// math with the per-query offset; all members live (Wts scratch sized
// M*24*TMAX). Launch 1 WI per (chunk-pos, head), group size 1.
struct ChunkAttn {
  float *Att;        // M*6144 out
  const float *Q;    // M*6144 queries (split, normed, roped)
  const float *Kc;   // 4*TC*256 cache
  const float *Vc;
  const float *Gate; // M*6144
  const int *Ctrl;   // [1] = chunk_start (global pos of chunk row 0)
  int TMAX;          // cache depth (clamp)
  float *Wts;        // M*24*TMAX scratch
  // Vectorized like AttnCore (simd<float,32> over d, Q row hoisted, exact
  // BF16 reinterpret via extend+shift+bitcast). 1 WI per (chunk-pos, head).
  void operator()(sycl::id<1> id) const SYCL_ESIMD_KERNEL {
    int i = id[0], m = i / 24, hh = i % 24;
    int kv = hh / 6, base = Ctrl[1], T = base + m + 1;
    if (T > TMAX)
      T = TMAX;
    float *wts = Wts + ((size_t)m * 24 + hh) * TMAX;
    const float *qh = Q + (size_t)m * 6144 + hh * 256;
    esimd::simd<float, 32> q0 = esimd::block_load<float, 32>(qh + 0);
    esimd::simd<float, 32> q1 = esimd::block_load<float, 32>(qh + 32);
    esimd::simd<float, 32> q2 = esimd::block_load<float, 32>(qh + 64);
    esimd::simd<float, 32> q3 = esimd::block_load<float, 32>(qh + 96);
    esimd::simd<float, 32> q4 = esimd::block_load<float, 32>(qh + 128);
    esimd::simd<float, 32> q5 = esimd::block_load<float, 32>(qh + 160);
    esimd::simd<float, 32> q6 = esimd::block_load<float, 32>(qh + 192);
    esimd::simd<float, 32> q7 = esimd::block_load<float, 32>(qh + 224);
    // NOTE: Kc/Vc are float caches in the chunk path (kept fp32, unlike
    // decode BF16) — plain float loads, no BF16 reinterpret needed.
    float mx = -1e30f;
    for (int t = 0; t < T; ++t) {
      const float *kr = Kc + ((size_t)t * 4 + kv) * 256;
      esimd::simd<float, 32> k0 = esimd::block_load<float, 32>(kr + 0);
      esimd::simd<float, 32> k1 = esimd::block_load<float, 32>(kr + 32);
      esimd::simd<float, 32> k2 = esimd::block_load<float, 32>(kr + 64);
      esimd::simd<float, 32> k3 = esimd::block_load<float, 32>(kr + 96);
      esimd::simd<float, 32> k4 = esimd::block_load<float, 32>(kr + 128);
      esimd::simd<float, 32> k5 = esimd::block_load<float, 32>(kr + 160);
      esimd::simd<float, 32> k6 = esimd::block_load<float, 32>(kr + 192);
      esimd::simd<float, 32> k7 = esimd::block_load<float, 32>(kr + 224);
      esimd::simd<float, 32> p0 = q0 * k0, p1 = q1 * k1, p2 = q2 * k2,
                             p3 = q3 * k3, p4 = q4 * k4, p5 = q5 * k5,
                             p6 = q6 * k6, p7 = q7 * k7;
      float s = 0;
#pragma unroll
      for (int l = 0; l < 32; ++l)
        s += (float)p0[l] + (float)p1[l] + (float)p2[l] + (float)p3[l] +
             (float)p4[l] + (float)p5[l] + (float)p6[l] + (float)p7[l];
      s /= 16.0f;
      wts[t] = s;
      mx = s > mx ? s : mx;
    }
    float se = 0;
    for (int t = 0; t < T; ++t) {
      float w = sycl::exp(wts[t] - mx);
      wts[t] = w;
      se += w;
    }
    float inv_se = 1.0f / se;
    esimd::simd<float, 32> a0(0), a1(0), a2(0), a3(0), a4(0), a5(0), a6(0),
        a7(0);
    for (int t = 0; t < T; ++t) {
      float w = wts[t] * inv_se;
      const float *vr = Vc + ((size_t)t * 4 + kv) * 256;
      a0 += w * esimd::block_load<float, 32>(vr + 0);
      a1 += w * esimd::block_load<float, 32>(vr + 32);
      a2 += w * esimd::block_load<float, 32>(vr + 64);
      a3 += w * esimd::block_load<float, 32>(vr + 96);
      a4 += w * esimd::block_load<float, 32>(vr + 128);
      a5 += w * esimd::block_load<float, 32>(vr + 160);
      a6 += w * esimd::block_load<float, 32>(vr + 192);
      a7 += w * esimd::block_load<float, 32>(vr + 224);
    }
    float *ah = Att + (size_t)m * 6144 + hh * 256;
    const float *gh = Gate + (size_t)m * 6144 + hh * 256;
    esimd::simd<float, 32> acc[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    for (int j = 0; j < 8; ++j) {
      esimd::simd<float, 32> g =
          esimd::block_load<float, 32>(gh + (size_t)j * 32);
      esimd::simd<float, 32> o(0);
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        float gl = (float)g[l];
        o[l] = (float)acc[j][l] / (1.0f + sycl::exp(-gl));
      }
      esimd::block_store<float, 32>(ah + (size_t)j * 32, o);
    }
  }
};

void launch_chunkattn(sycl::queue &q, float *Att, const float *Q,
                      const float *Kc, const float *Vc, const float *Gate,
                      const int *Ctrl, int TMAX, float *Wts, int M) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>((size_t)M * 24),
                   ChunkAttn{Att, Q, Kc, Vc, Gate, Ctrl, TMAX, Wts});
  }).wait();
}

// T7.4 chunked prefill: SSM chunk orchestration. Conv history (Cx3) and SSM
// state (48x128x128) PERSIST across chunk replays; each replay advances them
// through M steps. ChunkSsmConv: 1 WI/channel, sequential over chunk rows,
// history chains within the chunk. ChunkSsmRecur: 1 WI/head, ESIMD-vectorized
// over v in simd<float,32> lanes (OPT1 pattern), sequential over rows with
// Bt/Gt varying per row. Launch 1 WI/group, counts C and 48.
struct ChunkSsmConv {
  float *Mx;        // M*C conv output (silu)
  const float *QKV; // M*C feed-forward input
  float *CS;        // C*3 persistent history (zero-init)
  const float *CW;  // C*4 weights
  int C, M;
  void operator()(sycl::id<1> id) const {
    int c = id[0];
    if (c >= C)
      return;
    for (int m = 0; m < M; ++m) {
      float acc = CS[c * 3 + 0] * CW[(size_t)c * 4 + 0] +
                  CS[c * 3 + 1] * CW[(size_t)c * 4 + 1] +
                  CS[c * 3 + 2] * CW[(size_t)c * 4 + 2] +
                  QKV[(size_t)m * C + c] * CW[(size_t)c * 4 + 3];
      Mx[(size_t)m * C + c] = acc / (1.0f + sycl::exp(-acc));
      CS[c * 3 + 0] = CS[c * 3 + 1];
      CS[c * 3 + 1] = CS[c * 3 + 2];
      CS[c * 3 + 2] = QKV[(size_t)m * C + c];
    }
  }
};

struct ChunkSsmRecur {
  float *Out;      // M*48*128 head outputs
  const float *Q;  // M*48*128
  const float *K;
  const float *V;
  float *S;        // 48*128*128 persistent FP32 state (zero-init)
  const float *Bt; // M*48 beta
  const float *Gt; // M*48 log-decay
  int M;
  // Vectorized over the v (column) dimension in simd<float,32> lanes — the
  // OPT1 SsmRecur pattern, with per-row qh/kh/vh/gt/bt inside the m loop.
  // Out/Q/K/V are M*6144 tightly packed; do NOT alias Out onto a C-stride
  // buffer (stride trap, see chunklayer notes). 1 WI/head preserved.
  void operator()(sycl::id<1> id) const SYCL_ESIMD_KERNEL {
    constexpr int D = 128, NH = 48;
    int hh = id[0];
    if (hh >= NH)
      return;
    float *S0 = S + (size_t)hh * D * D;
    for (int m = 0; m < M; ++m) {
      const float *qh = Q + ((size_t)m * NH + hh) * D;
      const float *kh = K + ((size_t)m * NH + hh) * D;
      const float *vh = V + ((size_t)m * NH + hh) * D;
      float gt = sycl::exp(Gt[(size_t)m * NH + hh]);
      float bt = Bt[(size_t)m * NH + hh];
      for (int i = 0; i < D * D; i += 32) {
        esimd::simd<float, 32> r = esimd::block_load<float, 32>(S0 + i);
        r *= gt;
        esimd::block_store<float, 32>(S0 + i, r);
      }
      esimd::simd<float, 32> a0(0), a1(0), a2(0), a3(0);
      for (int k = 0; k < D; ++k) {
        float kk = kh[k];
        a0 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 0) * kk;
        a1 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 32) * kk;
        a2 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 64) * kk;
        a3 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 96) * kk;
      }
      esimd::simd<float, 32> v0 = esimd::block_load<float, 32>(vh + 0);
      esimd::simd<float, 32> v1 = esimd::block_load<float, 32>(vh + 32);
      esimd::simd<float, 32> v2 = esimd::block_load<float, 32>(vh + 64);
      esimd::simd<float, 32> v3 = esimd::block_load<float, 32>(vh + 96);
      a0 = (v0 - a0) * bt;
      a1 = (v1 - a1) * bt;
      a2 = (v2 - a2) * bt;
      a3 = (v3 - a3) * bt;
      for (int k = 0; k < D; ++k) {
        float kk = kh[k];
        esimd::simd<float, 32> r0 =
            esimd::block_load<float, 32>(S0 + (size_t)k * D + 0);
        esimd::simd<float, 32> r1 =
            esimd::block_load<float, 32>(S0 + (size_t)k * D + 32);
        esimd::simd<float, 32> r2 =
            esimd::block_load<float, 32>(S0 + (size_t)k * D + 64);
        esimd::simd<float, 32> r3 =
            esimd::block_load<float, 32>(S0 + (size_t)k * D + 96);
        r0 += a0 * kk;
        r1 += a1 * kk;
        r2 += a2 * kk;
        r3 += a3 * kk;
        esimd::block_store<float, 32>(S0 + (size_t)k * D + 0, r0);
        esimd::block_store<float, 32>(S0 + (size_t)k * D + 32, r1);
        esimd::block_store<float, 32>(S0 + (size_t)k * D + 64, r2);
        esimd::block_store<float, 32>(S0 + (size_t)k * D + 96, r3);
      }
      esimd::simd<float, 32> o0(0), o1(0), o2(0), o3(0);
      for (int k = 0; k < D; ++k) {
        float qq = qh[k];
        o0 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 0) * qq;
        o1 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 32) * qq;
        o2 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 64) * qq;
        o3 += esimd::block_load<float, 32>(S0 + (size_t)k * D + 96) * qq;
      }
      float *oh = Out + ((size_t)m * NH + hh) * D;
      esimd::block_store<float, 32>(oh + 0, o0);
      esimd::block_store<float, 32>(oh + 32, o1);
      esimd::block_store<float, 32>(oh + 64, o2);
      esimd::block_store<float, 32>(oh + 96, o3);
    }
  }
};

void launch_chunkssm(sycl::queue &q, float *Mx, const float *QKV, float *CS,
                     const float *CW, int C, int M, float *Out,
                     const float *Q, const float *K, const float *V, float *S,
                     const float *Bt, const float *Gt) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(C), ChunkSsmConv{Mx, QKV, CS, CW, C, M});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(48),
                   ChunkSsmRecur{Out, Q, K, V, S, Bt, Gt, M});
  }).wait();
}

// T7.4 chunked prefill: fp32->fp16 elementwise convert feeding chunk GEMMs
// (the layer stream stays fp32; only GEMM inputs are fp16). Launch count =
// element count, 1 WI/group.
struct CvtF32F16 {
  sycl::half *Out;
  const float *In;
  void operator()(sycl::id<1> id) const {
    Out[id[0]] = sycl::half(In[id[0]]);
  }
};

void launch_cvtf32f16(sycl::queue &q, sycl::half *Out, const float *In,
                      int N) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(N), CvtF32F16{Out, In});
  }).wait();
}

// T5.3 loop adoption: embedding gather driven by the control block
// (token_id = Ctrl[0]). BF16 rows decode inline; the recorded embed list is
// replayed per token with only the control contents changing.
struct Embed {
  float *Out; // 5120
  const uint16_t *Emb; // vocab*5120 BF16 table
  const int *Ctrl;
  void operator()(sycl::id<1> id) const {
    uint16_t b = Emb[(size_t)Ctrl[0] * 5120 + id[0]];
    uint32_t u = (uint32_t)b << 16;
    float x;
    __builtin_memcpy(&x, &u, 4);
    Out[id[0]] = x;
  }
};

void launch_linstages(sycl::queue &q, const float *X, float *SQ, int NS,
                      int8_t *Q8, int NQ, const float *Mx, float *Q48,
                      float *K48, float *V48, float *Bt, float *G48,
                      const float *B48, const float *A48, const float *AL,
                      const float *DT, float *Att, const float *Z,
                      const float *NG, float *Y, const float *A, const float *B,
                      int NR) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(NS), ScalesMax{X, SQ, NS});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(NQ), Quantize{X, SQ, Q8});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(6144), SplitRepeat{Mx, Q48, K48, V48});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(96), L2NormQK{Q48, K48});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(48), BetaG{Bt, G48, B48, A48, AL, DT});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(48), RmsInv{Bt, Mx});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(6144), NormGated{Att, Mx, Z, NG, Bt});
  }).wait();
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(NR), ResAddF{Y, A, B});
  }).wait();
  {
    uint16_t etab[5120] = {};
    int ctrl[4] = {};
    q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range<1>(5120), Embed{Y, etab, ctrl});
    }).wait();
    float bn[256];
    for (int j = 0; j < 256; ++j)
      bn[j] = 0.0f;
    q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range<1>(1), BatchNorm{Y, A, bn, 256});
    }).wait();
  float *qnk = const_cast<float *>(A), *gtk = const_cast<float *>(Y);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(6144), SplitQK{A, qnk, gtk});
  }).wait();
  }
}

void launch_argmax(sycl::queue &q, const float *Logits, float *PV, int *PI,
                   int *OutT) {
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> bv(256, h);
    sycl::local_accessor<int, 1> bi(256, h);
    h.parallel_for(sycl::nd_range<1>(64 * 256, 256),
                   ArgmaxS1{Logits, PV, PI, bv, bi});
  }).wait();
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> bv(256, h);
    sycl::local_accessor<int, 1> bi(256, h);
    h.parallel_for(sycl::nd_range<1>(256, 256), ArgmaxS2{PV, PI, OutT, bv, bi});
  }).wait();
}
