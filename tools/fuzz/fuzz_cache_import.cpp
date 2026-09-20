// T9.4: in-process fuzzer for AInferRuntime258V::import_diagnostic_cache.
// Inits the runtime ONCE, exports one valid cache as seed (after a tiny
// prefill), then loops over mutated cache files. Contract: import must
// ALWAYS return promptly with true/false — never crash, hang, or throw.
// Build with ASan+UBSan (see run_sanitizers.sh / T9.3) for memory findings.
// Usage: fuzz_cache_import <model.binfer> <all_kernels.spv> [iters] [seed]
#include "../decode/runtime_258v.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace ainfer;

static uint64_t rng64(std::mt19937_64 &r) { return r(); }

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "Usage: %s <model.binfer> <all_kernels.spv> [iters] [seed]\n", argv[0]);
    return 2;
  }
  int iters = (argc > 3) ? std::atoi(argv[3]) : 2000;
  uint64_t seed = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 777001ULL;
  if (iters < 1) iters = 1;

  AInferRuntime258V rt;
  if (!rt.init(argv[1], argv[2], 2048)) {
    std::fprintf(stderr, "runtime init failed\n");
    return 2;
  }

  // Seed: tiny prefill then export a VALID cache file.
  {
    std::vector<int> prompt = {151644, 8948, 198, 2610, 525, 264, 10925, 151645};
    int tok = 0;
    rt.reset_state();
    if (!rt.prefill(prompt, &tok)) {
      std::fprintf(stderr, "seed prefill failed\n");
      return 2;
    }
    if (!rt.export_diagnostic_cache("/tmp/opencode/fuzz_cache_seed.bin", 8)) {
      std::fprintf(stderr, "seed export failed\n");
      return 2;
    }
  }
  std::ifstream sf("/tmp/opencode/fuzz_cache_seed.bin", std::ios::binary);
  std::vector<uint8_t> seed_bytes((std::istreambuf_iterator<char>(sf)),
                                  std::istreambuf_iterator<char>());
  std::printf("[fuzz] seed cache: %zu bytes\n", seed_bytes.size());
  if (seed_bytes.size() < sizeof(DiagnosticCacheHeader) + 64) {
    std::fprintf(stderr, "seed file implausibly small\n");
    return 2;
  }

  std::mt19937_64 rng(seed);
  int findings = 0;
  double worst_s = 0.0;
  const double HANG_BUDGET_S = 10.0;

  for (int i = 0; i < iters; ++i) {
    std::vector<uint8_t> m = seed_bytes;
    int kind = (int)(rng64(rng) % 10);
    if (kind == 0) { // sparse byte flips anywhere
      int nflip = 1 + (int)(rng64(rng) % 8);
      for (int k = 0; k < nflip; ++k) m[(size_t)(rng64(rng) % m.size())] = (uint8_t)(rng64(rng) & 0xFF);
    } else if (kind == 1) { // truncation (incl. sub-header)
      size_t cut = 0;
      switch (rng64(rng) % 6) {
        case 0: cut = 0; break;
        case 1: cut = 15; break;
        case 2: cut = sizeof(DiagnosticCacheHeader) - 1; break;
        case 3: cut = sizeof(DiagnosticCacheHeader); break;
        default: cut = (size_t)(rng64(rng) % m.size()); break;
      }
      m.resize(cut);
    } else if (kind == 2) { // header u32/u64 field overwrite (magic/version/geometry/position/crc)
      static const size_t offs[] = {0, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 76};
      size_t off = offs[rng64(rng) % (sizeof(offs) / sizeof(offs[0]))];
      uint64_t v = rng64(rng);
      if (rng64(rng) & 1) v = (uint64_t)(rng64(rng) & 0xFFFFFFFFULL);
      if (off + 8 <= m.size()) std::memcpy(m.data() + off, &v, 8);
    } else if (kind == 3) { // huge geometry (fast-reject path)
      uint64_t huge = (rng64(rng) & 1) ? 0xFFFFFFFFFFFFFFFFULL : (uint64_t)(rng64(rng) % 1000000) * 1000000ULL;
      size_t off = 40 + 8 * (rng64(rng) % 3); // kv/ssm/conv totals
      if (off + 8 <= m.size()) std::memcpy(m.data() + off, &huge, 8);
    } else if (kind == 4) { // huge position (hdr.position @ offset 24)
      uint32_t p = (rng64(rng) & 1) ? 0xFFFFFFFFu : (uint32_t)(rng64(rng) & 0xFFFFFFFFu);
      if (24 + 4 <= m.size()) std::memcpy(m.data() + 24, &p, 4);
    } else if (kind == 5) { // payload corruption (CRC path, full-size reads)
      size_t pos = sizeof(DiagnosticCacheHeader) + (size_t)(rng64(rng) % (m.size() - sizeof(DiagnosticCacheHeader)));
      for (int k = 0; k < 16; ++k) m[(pos + k) % m.size()] ^= (uint8_t)(1 << (rng64(rng) & 7));
    } else if (kind == 6) { // appended garbage
      size_t extra = 1 + (size_t)(rng64(rng) % 1024);
      for (size_t k = 0; k < extra; ++k) m.push_back((uint8_t)(rng64(rng) & 0xFF));
    } else if (kind == 7) { // all-zeros / all-0xFF header
      uint8_t fill = (rng64(rng) & 1) ? 0x00 : 0xFF;
      for (size_t k = 0; k < sizeof(DiagnosticCacheHeader) && k < m.size(); ++k) m[k] = fill;
    } else if (kind == 8) { // swap two 64-byte blocks (structural shuffle)
      if (m.size() > 256) {
        size_t a = (size_t)(rng64(rng) % (m.size() - 128)) & ~63ULL;
        size_t b = (size_t)(rng64(rng) % (m.size() - 128)) & ~63ULL;
        for (int k = 0; k < 64; ++k) std::swap(m[a + k], m[b + k]);
      }
    } else { // single-bit flips
      for (int k = 0, n = 1 + (int)(rng64(rng) % 4); k < n; ++k)
        m[(size_t)(rng64(rng) % m.size())] ^= (uint8_t)(1 << (rng64(rng) & 7));
    }

    std::string path = "/tmp/opencode/fuzz_cache_case.bin";
    {
      std::ofstream of(path, std::ios::binary | std::ios::trunc);
      of.write((const char *)m.data(), m.size());
    }
    uint32_t pos_out = 0xDEADu;
    auto t0 = std::chrono::steady_clock::now();
    bool ok = rt.import_diagnostic_cache(path, &pos_out); // must not crash/hang/throw
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    worst_s = std::max(worst_s, dt);
    (void)ok; // both outcomes legal; only crash/hang is a finding (caught by ASan/harness timeout)
    if (dt > HANG_BUDGET_S) {
      std::printf("[FINDING] iter=%d hang %.1fs\n", i, dt);
      ++findings;
      if (findings >= 20) break;
    }
    if ((i + 1) % 500 == 0) {
      std::printf("[%d/%d] worst %.2fs, findings %d\n", i + 1, iters, worst_s, findings);
      std::fflush(stdout);
    }
  }
  std::printf("=== fuzz_cache_import: %d iters, %d findings, worst %.2fs ===\n",
              iters, findings, worst_s);
  return findings == 0 ? 0 : 1;
}
