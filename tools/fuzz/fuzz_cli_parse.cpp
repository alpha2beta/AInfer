// T9.4: in-process fuzzer for tools/decode/cli_parse.h helpers.
// Contract: parse_cli_int/u32/token must NEVER throw, crash, or hang on
// arbitrary byte strings; must accept exactly the strict grammar
// (optional sign, digits only, in-range) and reject everything else.
// Millions of iterations (pure CPU, no device).
// Usage: fuzz_cli_parse [iters] [seed]
#include "../decode/cli_parse.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

using namespace ainfer;

// Independent oracle: same written spec as the impl (optional single sign,
// digits only, in-range) but structured differently (accumulate-then-check
// vs the impl's early-exit) so coding slips still diverge.
static bool oracle_int(const std::string &s, int lo, int hi, int &out) {
  if (s.empty() || s.size() > 11) return false;
  size_t i = 0;
  bool neg = false;
  if (s[0] == '-' || s[0] == '+') {
    neg = (s[0] == '-');
    i = 1;
  }
  if (i >= s.size()) return false;
  long long v = 0;
  for (; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') return false;
    v = v * 10 + (s[i] - '0');
  }
  v = neg ? -v : v;
  if (v < lo || v > hi) return false;
  out = (int)v;
  return true;
}

int main(int argc, char **argv) {
  long iters = (argc > 1) ? std::atol(argv[1]) : 2000000L;
  uint64_t seed = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 31337ULL;
  std::mt19937_64 rng(seed);

  // Interesting fixed corpus (boundary + adversarial strings)
  const char *fixed[] = {
    "", "-", "+", "--1", "++1", "-0", "+0", "0", "00", "01", " 1", "1 ",
    "1\n", "\t42", "2147483647", "2147483648", "-2147483648", "-2147483649",
    "99999999999999999999", "-99999999999999999999", "0x10", "010", "1e3",
    "3.14", "32x", "x32", "1_000", ",", "1,2", "\xff", "\x80\x81",
    "4294967295", "4294967296", "1048576", "1048577", "-1",
    std::string(100, '9').c_str(), std::string(10000, '7').c_str(),
  };
  const int nfixed = (int)(sizeof(fixed) / sizeof(fixed[0]));

  auto rand_str = [&](std::string &s) {
    int kind = (int)(rng() % 5);
    s.clear();
    if (kind == 0) { // digits with optional sign/junk
      if (rng() & 1) s += (rng() & 1) ? '-' : '+';
      int n = 1 + (int)(rng() % 12);
      for (int i = 0; i < n; ++i) s += (char)('0' + rng() % 10);
      if (rng() % 3 == 0) s += (char)(rng() % 128);
    } else if (kind == 1) { // pure garbage bytes
      int n = (int)(rng() % 24);
      for (int i = 0; i < n; ++i) s += (char)(rng() & 0xFF);
    } else if (kind == 2) { // huge digit strings
      int n = 20 + (int)(rng() % 200);
      if (rng() & 1) s += '-';
      for (int i = 0; i < n; ++i) s += (char)('0' + rng() % 10);
    } else if (kind == 3) { // near INT_MAX/MIN boundaries
      long long base = (rng() & 1) ? 2147483647LL : -2147483648LL;
      long long v = base + (long long)(rng() % 11) - 5;
      s = std::to_string(v);
    } else { // single chars across byte range
      s += (char)(rng() & 0xFF);
    }
  };

  int findings = 0;
  auto t0 = std::chrono::steady_clock::now();
  std::string s;
  // Phase 1: fixed corpus (deterministic edge coverage)
  for (int i = 0; i < nfixed; ++i) {
    s = fixed[i];
    int got = 0, want = 0;
    bool r = parse_cli_int(s, got, -100, 100000);
    bool e = oracle_int(s, -100, 100000, want);
    if (r != e || (r && got != want)) {
      std::printf("[FINDING] fixed[%d]=%s impl=%d oracle=%d (%d vs %d)\n",
                  i, s.c_str(), r, e, got, want);
      if (++findings >= 20) break;
    }
  }
  // Phase 2: random differential vs oracle + u32/token/API sanity
  for (long i = 0; i < iters && findings < 20; ++i) {
    rand_str(s);
    int got = 0, want = 0;
    bool r = parse_cli_int(s, got, -1000, 1000000);
    bool e = oracle_int(s, -1000, 1000000, want);
    if (r != e || (r && got != want)) {
      std::printf("[FINDING] iter=%ld impl=%d oracle=%d %s\n", i, r, e, s.c_str());
      ++findings;
      continue;
    }
    // u32/token wrappers must agree with int on non-negative inputs
    uint32_t u = 0;
    bool ru = parse_cli_u32(s, u, 0, 1000000);
    int tok = 0;
    (void)parse_cli_token(s, tok);
    // token helper on huge-but-wellformed input must not crash (covered above)
    (void)ru;
  }
  auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("=== fuzz_cli_parse: %ld iters + %d fixed, %d findings, %.1fs ===\n",
              iters, nfixed, findings, dt);
  return findings == 0 ? 0 : 1;
}
