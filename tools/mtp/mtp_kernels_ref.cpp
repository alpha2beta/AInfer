// MTP kernel reference extract for 258V port (DO NOT BUILD here).
// Origin: tools/cmdlist/kernels.cpp @ 06f267e for M2/Concat2 line numbers;
// Int4GemvM3 block: lines 279-371 of the UNCOMMITTED working tree at time
// of extract (2026-09-19; verify against HEAD before porting).
// Extracted verbatim (byte-identical); needs SYCL/ESIMD headers + surrounding
// kernel registry (entry refs _ZTS10Int4GemvM2 / _ZTS7Concat2) to compile.


// Speculative decoding 2-token GEMV (T7.2): streams weights P and scales S
// once from memory, unpacks nibbles once, and executes dual dp4a dot products
// against two activation vectors (XQ0, SQ0) and (XQ1, SQ1) in GRF registers.
// Launch with 1 WI per L0 group (M groups).
struct Int4GemvM2 {
  float *Y0;
  float *Y1;
  const uint8_t *P;
  const uint16_t *S;
  const int8_t *XQ0;
  const float *SQ0;
  const int8_t *XQ1;
  const float *SQ1;
  int K;
  void operator()(sycl::id<1> id) const SYCL_ESIMD_KERNEL {
    int m = id[0];
    float acc0 = 0, acc1 = 0;
    int Gg = K / 128;
    for (int g = 0; g < Gg; ++g) {
      uint32_t ub = S[(size_t)m * Gg + g];
      ub <<= 16;
      float sc;
      __builtin_memcpy(&sc, &ub, 4);
      float sqg0 = SQ0[g];
      float sqg1 = SQ1[g];
      size_t base = (size_t)m * K / 2 + (size_t)g * 64;
      esimd::simd<int, 8> acc8_0(0);
      esimd::simd<int, 8> acc8_1(0);
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
        esimd::simd<int, 8> w32 = w8.bit_cast_view<int>();

        esimd::simd<int8_t, 32> xq0 =
            esimd::block_load<int8_t, 32>(XQ0 + g * 128 + j);
        esimd::simd<int8_t, 32> xq1 =
            esimd::block_load<int8_t, 32>(XQ1 + g * 128 + j);
        esimd::simd<int, 8> x32_0 = xq0.bit_cast_view<int>();
        esimd::simd<int, 8> x32_1 = xq1.bit_cast_view<int>();

        acc8_0 = esimd::dp4a<int>(acc8_0, w32, x32_0);
        acc8_1 = esimd::dp4a<int>(acc8_1, w32, x32_1);
      }
      int tmp0[8], tmp1[8];
      acc8_0.copy_to(tmp0);
      acc8_1.copy_to(tmp1);
      int gs0 = 0, gs1 = 0;
      for (int u = 0; u < 8; ++u) {
        gs0 += tmp0[u];
        gs1 += tmp1[u];
      }
      acc0 += (float)gs0 * sc * sqg0;
      acc1 += (float)gs1 * sc * sqg1;
    }
    Y0[m] = acc0;
    Y1[m] = acc1;
  }
};

void launch_int4gemvm2(sycl::queue &q, float *Y0, float *Y1, const uint8_t *P,
                       const uint16_t *S, const int8_t *XQ0, const float *SQ0,
                       const int8_t *XQ1, const float *SQ1, int M, int K) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(M),
                   Int4GemvM2{Y0, Y1, P, S, XQ0, SQ0, XQ1, SQ1, K});
  }).wait();
}


// T7.2 MTP draft: concatenate two N-vectors (fused fc input cat[e, hn]).
// Launch 2*N, 1 WI/group; all members live (TMAX lesson).
struct Concat2 {
  float *Out; // 2*N
  const float *A;
  const float *B;
  int N;
  void operator()(sycl::id<1> id) const {
    int i = id[0];
    if (i >= 2 * N)
      return;
    Out[i] = i < N ? A[i] : B[i - N];
  }
};

// Dead-strip guard (TMAX lesson): an unreferenced functor vanishes from the
// bundle and extract_spv fails the build. The raw-L0 path never calls this,
// but the reference retains the entry point.
void launch_concat2(sycl::queue &q, float *Out, const float *A,
                    const float *B, int N) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>((size_t)2 * N), Concat2{Out, A, B, N});
  }).wait();
}

// Depth-2 speculative verify 3-token GEMV (T7.2 chained drafts): M2 pattern
// with a third activation lane. Weights P and scales S still stream once;
// all three dp4a lanes share the unpacked nibbles in GRF. Launch 1 WI per
// L0 group (M groups). All members live (dead-strip guard below).
struct Int4GemvM3 {
  float *Y0;
  float *Y1;
  float *Y2;
  const uint8_t *P;
  const uint16_t *S;
  const int8_t *XQ0;
  const float *SQ0;
  const int8_t *XQ1;
  const float *SQ1;
  const int8_t *XQ2;
  const float *SQ2;
  int K;
  void operator()(sycl::id<1> id) const SYCL_ESIMD_KERNEL {
    int m = id[0];
    float acc0 = 0, acc1 = 0, acc2 = 0;
    int Gg = K / 128;
    for (int g = 0; g < Gg; ++g) {
      uint32_t ub = S[(size_t)m * Gg + g];
      ub <<= 16;
      float sc;
      __builtin_memcpy(&sc, &ub, 4);
      float sqg0 = SQ0[g];
      float sqg1 = SQ1[g];
      float sqg2 = SQ2[g];
      size_t base = (size_t)m * K / 2 + (size_t)g * 64;
      esimd::simd<int, 8> acc8_0(0);
      esimd::simd<int, 8> acc8_1(0);
      esimd::simd<int, 8> acc8_2(0);
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
        esimd::simd<int, 8> w32 = w8.bit_cast_view<int>();

        esimd::simd<int8_t, 32> xq0 =
            esimd::block_load<int8_t, 32>(XQ0 + g * 128 + j);
        esimd::simd<int8_t, 32> xq1 =
            esimd::block_load<int8_t, 32>(XQ1 + g * 128 + j);
        esimd::simd<int8_t, 32> xq2 =
            esimd::block_load<int8_t, 32>(XQ2 + g * 128 + j);
        esimd::simd<int, 8> x32_0 = xq0.bit_cast_view<int>();
        esimd::simd<int, 8> x32_1 = xq1.bit_cast_view<int>();
        esimd::simd<int, 8> x32_2 = xq2.bit_cast_view<int>();

        acc8_0 = esimd::dp4a<int>(acc8_0, w32, x32_0);
        acc8_1 = esimd::dp4a<int>(acc8_1, w32, x32_1);
        acc8_2 = esimd::dp4a<int>(acc8_2, w32, x32_2);
      }
      int tmp0[8], tmp1[8], tmp2[8];
      acc8_0.copy_to(tmp0);
      acc8_1.copy_to(tmp1);
      acc8_2.copy_to(tmp2);
      int gs0 = 0, gs1 = 0, gs2 = 0;
      for (int u = 0; u < 8; ++u) {
        gs0 += tmp0[u];
        gs1 += tmp1[u];
        gs2 += tmp2[u];
      }
      acc0 += (float)gs0 * sc * sqg0;
      acc1 += (float)gs1 * sc * sqg1;
      acc2 += (float)gs2 * sc * sqg2;
    }
    Y0[m] = acc0;
    Y1[m] = acc1;
    Y2[m] = acc2;
  }
};

// Dead-strip guard (Concat2 lesson): raw-L0 never calls this, but the
// reference retains the entry point in the bundle for extract_spv.
void launch_int4gemvm3(sycl::queue &q, float *Y0, float *Y1, float *Y2,
                       const uint8_t *P, const uint16_t *S,
                       const int8_t *XQ0, const float *SQ0,
                       const int8_t *XQ1, const float *SQ1,
                       const int8_t *XQ2, const float *SQ2, int M, int K) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(M),
                   Int4GemvM3{Y0, Y1, Y2, P, S, XQ0, SQ0, XQ1, SQ1, XQ2, SQ2,
                              K});
  }).wait();
}
