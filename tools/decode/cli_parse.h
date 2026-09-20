// T9.4: shared exception-safe CLI/file integer parsing for AInfer tools.
// std::stoi/stoul throw invalid_argument/out_of_range on garbage, which
// previously escaped main() and aborted via std::terminate (SIGABRT).
// All helpers return false (caller prints usage error, exits 2) instead.
#pragma once

#include <climits>
#include <cstdint>
#include <string>

namespace ainfer {

// Strict integer in [lo, hi]: optional single leading sign, then 1+ ASCII
// digits, nothing else. Deliberately NOT based on stol/stoul (which skip
// leading whitespace and accept '+'/'-' quirks); explicit scan also bounds
// length up front so no overflow is representable (>11 chars can never fit
// an int result). Never throws.
inline bool parse_cli_int(const std::string &s, int &out, int lo, int hi) {
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
    if (!neg && v > hi) return false;
    if (neg && -v < lo) return false;
  }
  v = neg ? -v : v;
  if (v < lo || v > hi) return false;
  out = (int)v;
  return true;
}

// Strict uint32 in [lo, hi] (for --max-ctx style args): digits only, no sign.
inline bool parse_cli_u32(const std::string &s, uint32_t &out, uint32_t lo,
                          uint32_t hi) {
  if (s.empty() || s.size() > 10) return false;
  unsigned long long v = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') return false;
    v = v * 10 + (unsigned)(s[i] - '0');
    if (v > hi) return false;
  }
  if (v < lo) return false;
  out = (uint32_t)v;
  return true;
}

// Token IDs: full int range here (range-checked later by StepGuard);
// the only requirement is crash-free parsing.
inline bool parse_cli_token(const std::string &s, int &out) {
  return parse_cli_int(s, out, INT_MIN, INT_MAX);
}

} // namespace ainfer
