// Standalone correctness primitive for the runtime_258v 16Q/2KV topology.
// Cache is head-major; scales are token-major. This module is intentionally
// kept separate from the runtime bundle until its host-reference replay gate
// passes.
#define D 256
#define NQ 16
#define NKV 2
#define GQA 8
#define ROT 64
#define HALF 32
#define THETA 10000000.0f
#define SCALE 0.0625f

static inline char q8(float x) {
  int q = (int)(x >= 0.0f ? x + 0.5f : x - 0.5f);
  return (char)(q < -127 ? -127 : (q > 127 ? 127 : q));
}
static inline float absv(float x) { return x < 0.0f ? -x : x; }
static inline float rope(__local const float *x, int d, float c, float s) {
  if (d < HALF) return x[d] * c - x[d + HALF] * s;
  if (d < ROT) return x[d - HALF] * s + x[d] * c;
  return x[d];
}
static inline void coeff(int d, uint pos, float *c, float *s) {
  if (d < ROT) {
    int i = d < HALF ? d : d - HALF;
    float a = (float)pos / pow(THETA, (float)(2 * i) / (float)ROT);
    *c = cos(a); *s = sin(a);
  } else { *c = 1.0f; *s = 0.0f; }
}

// One workgroup of 256. The two local arrays avoid any global-memory
// producer/consumer race between rotation, max reduction, and quantization.
__kernel void kv8_append_ctrl(
    __global float *q, __global float *k, __global const float *v,
    __global char *kc, __global char *vc, __global float *ks, __global float *vs,
    __global const int *ctrl, uint max_ctx) {
  uint pos = (uint)ctrl[1];
  if (pos >= max_ctx) return;
  int d = get_local_id(0);
  __local float row[D];
  __local float rotated[D];
  float c, s; coeff(d, pos, &c, &s);
  for (int h = 0; h < NQ; ++h) {
    __global float *qh = q + h * D;
    row[d] = qh[d]; barrier(CLK_LOCAL_MEM_FENCE);
    qh[d] = rope(row, d, c, s); barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int h = 0; h < NKV; ++h) {
    __global float *kh = k + h * D;
    row[d] = kh[d]; barrier(CLK_LOCAL_MEM_FENCE);
    rotated[d] = rope(row, d, c, s); barrier(CLK_LOCAL_MEM_FENCE);
    float mk = 0.0f, mv = 0.0f;
    for (int i = 0; i < D; ++i) {
      mk = fmax(mk, absv(rotated[i]));
      mv = fmax(mv, absv(v[h * D + i]));
    }
    float sk = mk == 0.0f ? 1.0f : mk / 127.0f;
    float sv = mv == 0.0f ? 1.0f : mv / 127.0f;
    if (d == 0) { ks[pos * NKV + h] = sk; vs[pos * NKV + h] = sv; }
    barrier(CLK_LOCAL_MEM_FENCE);
    kh[d] = rotated[d];
    kc[((size_t)h * max_ctx + pos) * D + d] = q8(rotated[d] / sk);
    vc[((size_t)h * max_ctx + pos) * D + d] = q8(v[h * D + d] / sv);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void kv8_attn_ctrl(
    __global float *out, __global const float *q, __global const float *gate,
    __global const char *kc, __global const char *vc,
    __global const float *ks, __global const float *vs,
    __global const int *ctrl, uint max_ctx) {
  int qh = get_group_id(0), d = get_local_id(0);
  if (qh >= NQ) return;
  int lane = get_sub_group_local_id();
  int sg = get_sub_group_id();
  uint total = (uint)ctrl[1] + 1;
  if (total > max_ctx) total = max_ctx;
  int kv = qh / GQA;
  float qv = q[qh * D + d];

  __local float s_part[2][256];
  float mx = -1.0e30f, sum = 0.0f, acc = 0.0f;

  for (uint tb = 0; tb < total; tb += 16) {
    int buf_idx = (int)((tb >> 4) & 1);
    float part[16];
    #pragma unroll
    for (int u = 0; u < 16; ++u) {
      uint t = tb + (uint)u;
      float kval = 0.0f;
      if (t < total) {
        kval = (float)kc[((size_t)kv * max_ctx + t) * D + d] * ks[t * NKV + kv];
      }
      part[u] = qv * kval;
    }

    #pragma unroll
    for (int u = 0; u < 16; ++u) {
      float v = part[u];
      v += intel_sub_group_shuffle(v, (uint)(lane ^ 8));
      v += intel_sub_group_shuffle(v, (uint)(lane ^ 4));
      v += intel_sub_group_shuffle(v, (uint)(lane ^ 2));
      v += intel_sub_group_shuffle(v, (uint)(lane ^ 1));
      part[u] = v;
    }

    if (lane == 0) {
      #pragma unroll
      for (int u = 0; u < 16; ++u) {
        s_part[buf_idx][sg * 16 + u] = part[u];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int u = 0; u < 16; ++u) {
      float s = 0.0f;
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        s += s_part[buf_idx][i * 16 + u];
      }
      uint t = tb + (uint)u;
      if (t < total) {
        float score = s * SCALE;
        float vv = (float)vc[((size_t)kv * max_ctx + t) * D + d] * vs[t * NKV + kv];
        if (score > mx) {
          float e = exp(mx - score);
          acc = acc * e + vv;
          sum = sum * e + 1.0f;
          mx = score;
        } else {
          float e = exp(score - mx);
          acc += e * vv;
          sum += e;
        }
      }
    }
  }
  float g = gate[qh * D + d];
  out[qh * D + d] = (acc / sum) / (1.0f + exp(-g));
}

// Split-T KV8 decode attention (I3.8 follow-up, env-gated via AINFER_ATTN_SPLIT).
// Same structure as the BF16 gqa_attn_decode_split/combine pair with int8 KV
// loads + per-token scales. Partials layout: [16 heads][SMAX splits][258].
// Merge reorders FP ops (functional parity only) — hence env-gated.
#define KV8_SPLIT_MAX 8

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void kv8_attn_decode_split(
    __global float *partials,      // [16, KV8_SPLIT_MAX, 258]
    __global const float *q,       // [16, 256]
    __global const char *kc,       // [2, max_ctx, 256] int8
    __global const char *vc,       // [2, max_ctx, 256] int8
    __global const float *ks,      // [max_ctx, 2] per-token scales
    __global const float *vs,      // [max_ctx, 2]
    __global const int *ctrl,      // ctrl[1] = position
    uint max_ctx,
    int n_splits
) {
  int gid = get_group_id(0);
  int qh = gid % NQ;
  int sp = gid / NQ;
  if (qh >= NQ || sp >= n_splits) return;
  int d = get_local_id(0);
  int lane = get_sub_group_local_id();
  int sg = get_sub_group_id();

  uint total = (uint)ctrl[1] + 1;
  if (total > max_ctx) total = max_ctx;
  int kv = qh / GQA;
  float qv = q[qh * D + d];

  uint chunk = (total + (uint)n_splits - 1u) / (uint)n_splits;
  uint t0 = (uint)sp * chunk;
  uint t1 = t0 + chunk;
  if (t1 > total) t1 = total;

  __local float s_part[2][256];
  float mx = -1.0e30f, sum = 0.0f, acc = 0.0f;

  if (t0 < t1) {
    for (uint tb = t0 & ~15u; tb < t1; tb += 16) {
      int buf_idx = (int)((tb >> 4) & 1);
      float part[16];
      #pragma unroll
      for (int u = 0; u < 16; ++u) {
        uint t = tb + (uint)u;
        float kval = 0.0f;
        if (t >= t0 && t < t1) {
          kval = (float)kc[((size_t)kv * max_ctx + t) * D + d] * ks[t * NKV + kv];
        }
        part[u] = qv * kval;
      }

      #pragma unroll
      for (int u = 0; u < 16; ++u) {
        float v = part[u];
        v += intel_sub_group_shuffle(v, (uint)(lane ^ 8));
        v += intel_sub_group_shuffle(v, (uint)(lane ^ 4));
        v += intel_sub_group_shuffle(v, (uint)(lane ^ 2));
        v += intel_sub_group_shuffle(v, (uint)(lane ^ 1));
        part[u] = v;
      }

      if (lane == 0) {
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
          s_part[buf_idx][sg * 16 + u] = part[u];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      #pragma unroll
      for (int u = 0; u < 16; ++u) {
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
          s += s_part[buf_idx][i * 16 + u];
        }
        uint t = tb + (uint)u;
        if (t >= t0 && t < t1) {
          float score = s * SCALE;
          float vv = (float)vc[((size_t)kv * max_ctx + t) * D + d] * vs[t * NKV + kv];
          if (score > mx) {
            float e = exp(mx - score);
            acc = acc * e + vv;
            sum = sum * e + 1.0f;
            mx = score;
          } else {
            float e = exp(score - mx);
            acc += e * vv;
            sum += e;
          }
        }
      }
    }
  }

  __global float *dst = partials + ((size_t)qh * KV8_SPLIT_MAX + sp) * 258;
  dst[d] = acc;
  if (d == 0) {
    dst[256] = mx;
    dst[257] = sum;
  }
}

__kernel void kv8_attn_combine(
    __global float *out,           // [16, 256]
    __global const float *gate,    // [16, 256]
    __global const float *partials,// [16, KV8_SPLIT_MAX, 258]
    int n_splits
) {
  int qh = get_group_id(0);
  if (qh >= NQ) return;
  int d = get_local_id(0);

  float m = -1.0e30f;
  for (int s = 0; s < n_splits; ++s) {
    float ms = partials[((size_t)qh * KV8_SPLIT_MAX + s) * 258 + 256];
    if (ms > m) m = ms;
  }
  float l = 0.0f, acc = 0.0f;
  for (int s = 0; s < n_splits; ++s) {
    __global const float *ps = partials + ((size_t)qh * KV8_SPLIT_MAX + s) * 258;
    float e = exp(ps[256] - m);
    l += ps[257] * e;
    acc += ps[d] * e;
  }

  float g = gate[qh * D + d];
  out[qh * D + d] = (acc / l) / (1.0f + exp(-g));
}

// B=2 verification variant. Each workgroup owns one token, so local storage
// and per-token scales are independent while both token rows land in the same
// cache before the corresponding causal attention groups execute.
__kernel void kv8_append_batch(
    __global float *q, __global float *k, __global const float *v,
    __global char *kc, __global char *vc, __global float *ks, __global float *vs,
    __global const int *ctrl, uint max_ctx, int B) {
  int b = get_group_id(0);
  if (b >= B) return;
  uint pos = (uint)ctrl[1] + (uint)b;
  if (pos >= max_ctx) return;
  int d = get_local_id(0);
  __local float row[D];
  __local float rotated[D];
  float c, s; coeff(d, pos, &c, &s);
  __global float *qb = q + (size_t)b * NQ * D;
  for (int h = 0; h < NQ; ++h) {
    __global float *qh = qb + h * D;
    row[d] = qh[d]; barrier(CLK_LOCAL_MEM_FENCE);
    qh[d] = rope(row, d, c, s); barrier(CLK_LOCAL_MEM_FENCE);
  }
  __global float *kb = k + (size_t)b * NKV * D;
  __global const float *vb = v + (size_t)b * NKV * D;
  for (int h = 0; h < NKV; ++h) {
    __global float *kh = kb + h * D;
    row[d] = kh[d]; barrier(CLK_LOCAL_MEM_FENCE);
    rotated[d] = rope(row, d, c, s); barrier(CLK_LOCAL_MEM_FENCE);
    float mk = 0.0f, mv = 0.0f;
    for (int i = 0; i < D; ++i) {
      mk = fmax(mk, absv(rotated[i]));
      mv = fmax(mv, absv(vb[h * D + i]));
    }
    float sk = mk == 0.0f ? 1.0f : mk / 127.0f;
    float sv = mv == 0.0f ? 1.0f : mv / 127.0f;
    if (d == 0) { ks[pos * NKV + h] = sk; vs[pos * NKV + h] = sv; }
    barrier(CLK_LOCAL_MEM_FENCE);
    kh[d] = rotated[d];
    kc[((size_t)h * max_ctx + pos) * D + d] = q8(rotated[d] / sk);
    vc[((size_t)h * max_ctx + pos) * D + d] = q8(vb[h * D + d] / sv);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

__kernel void kv8_attn_batch(
    __global float *out, __global const float *q, __global const float *gate,
    __global const char *kc, __global const char *vc,
    __global const float *ks, __global const float *vs,
    __global const int *ctrl, uint max_ctx, int B) {
  int id = get_group_id(0), b = id / NQ, qh = id % NQ, d = get_local_id(0);
  if (b >= B) return;
  uint total = (uint)ctrl[1] + (uint)b + 1;
  if (total > max_ctx) total = max_ctx;
  int kv = qh / GQA;
  __local float red[D];
  float mx = -1.0e30f, sum = 0.0f, acc = 0.0f;
  float qv = q[(size_t)b * NQ * D + qh * D + d];
  for (uint t = 0; t < total; ++t) {
    float kval = (float)kc[((size_t)kv * max_ctx + t) * D + d] * ks[t * NKV + kv];
    red[d] = qv * kval;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 128; step > 0; step >>= 1) {
      if (d < step) red[d] += red[d + step];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    float score = red[0] * SCALE;
    float vv = (float)vc[((size_t)kv * max_ctx + t) * D + d] * vs[t * NKV + kv];
    if (score > mx) {
      float e = exp(mx - score);
      acc = acc * e + vv;
      sum = sum * e + 1.0f;
      mx = score;
    } else {
      float e = exp(score - mx);
      acc += e * vv;
      sum += e;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float g = gate[(size_t)b * NQ * D + qh * D + d];
  out[(size_t)b * NQ * D + qh * D + d] = (acc / sum) / (1.0f + exp(-g));
}
