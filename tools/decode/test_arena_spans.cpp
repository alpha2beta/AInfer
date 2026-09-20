// T9.1 unit tests: ArenaSpan bounds checks, CheckedArena overflow ledger,
// checked_mul_add overflow detection. Host-only (no Level Zero device).
// Pass criteria: all CHECKs green, OOB cases diagnosed (stderr) without crash.
#include "../decode/runtime_258v.h"

#include <cstdio>
#include <vector>

using namespace ainfer;

static int failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
      ++failures;                                                              \
    } else {                                                                   \
      std::printf("ok: %s\n", (msg));                                          \
    }                                                                          \
  } while (0)

int main() {
  // Silence expected OOB diagnostics during negative tests is not needed;
  // they go to stderr and prove the checks fire.
  std::printf("--- ArenaSpan ---\n");
  float buf[16];
  for (int i = 0; i < 16; ++i) buf[i] = (float)i;
  ArenaSpan<float> sp{buf, 16, "test"};

  CHECK(sp.at(0) == &buf[0], "at(0) valid");
  CHECK(sp.at(15) == &buf[15], "at(15) valid");
  CHECK(sp.at(16) == nullptr, "at(16) OOB -> null");
  CHECK(sp.at(SIZE_MAX) == nullptr, "at(SIZE_MAX) OOB -> null");
  ArenaSpan<float> null_sp{nullptr, 0, "null"};
  CHECK(null_sp.at(0) == nullptr, "at() on null span -> null");

  auto s1 = sp.slice(4, 8, "s1");
  CHECK(s1.get() == &buf[4] && s1.count == 8, "slice(4,8) valid");
  CHECK(s1.at(7) == &buf[11], "slice element maps to parent");
  auto s2 = sp.slice(10, 7, "s2");
  CHECK(s2.get() == nullptr && s2.count == 0, "slice(10,7) overruns -> empty");
  auto s3 = sp.slice(16, 0, "s3");
  CHECK(s3.get() != nullptr && s3.count == 0, "slice(16,0) empty-but-anchored ok");
  auto s4 = sp.slice(17, 0, "s4");
  CHECK(s4.get() == nullptr, "slice(17,0) off>count -> empty");
  auto s5 = sp.slice(0, 17, "s5");
  CHECK(s5.get() == nullptr, "slice(0,17) n>count -> empty");
  // SIZE_MAX arithmetic must not wrap into a valid slice
  auto s6 = sp.slice(SIZE_MAX - 4, 8, "s6");
  CHECK(s6.get() == nullptr, "slice near SIZE_MAX -> empty (no wraparound)");
  CHECK(sp.size_bytes() == 16 * sizeof(float), "size_bytes");

  std::printf("--- CheckedArena ---\n");
  std::vector<uint8_t> backing(1024);
  CheckedArena ar;
  ar.reset(backing.data(), backing.size());
  void *a1 = ar.bump(100, sizeof(float), "a1"); // 400B -> 448 aligned
  CHECK(a1 != nullptr, "bump within budget ok");
  void *a2 = ar.bump(100, sizeof(float), "a2");
  CHECK(a2 != nullptr && a2 != a1, "second bump distinct");
  CHECK(ar.ledger.size() == 2, "ledger records 2 entries");
  void *a3 = ar.bump(10000, sizeof(float), "too-big");
  CHECK(a3 == nullptr, "oversize bump rejected");
  CHECK(ar.ledger.size() == 2, "rejected bump not ledgered");
  // Fill to exactly the edge: 1024 - 896 = 128 left
  void *a4 = ar.bump(32, sizeof(float), "edge"); // 128B exact fit
  CHECK(a4 != nullptr, "exact-fit bump ok");
  void *a5 = ar.bump(1, sizeof(float), "over");
  CHECK(a5 == nullptr, "1-byte-past-full rejected");
  ArenaSpan<float> lk;
  CHECK(ar.span_f32("a1", &lk) && lk.count * sizeof(float) >= 100 * sizeof(float),
        "span_f32 lookup by name");
  CHECK(!ar.span_f32("missing", &lk), "span_f32 unknown name -> false");
  // Mul-overflow in bump size computation
  void *a6 = ar.bump(SIZE_MAX / 2, 8, "mul-overflow");
  CHECK(a6 == nullptr, "bump size mul-overflow rejected");

  std::printf("--- checked_mul_add ---\n");
  size_t out = 0;
  CHECK(checked_mul_add(10, 20, 5, &out) && out == 205, "10*20+5=205 ok");
  CHECK(!checked_mul_add(SIZE_MAX, 2, 0, &out), "mul overflow detected");
  CHECK(!checked_mul_add(SIZE_MAX - 1, 1, 5, &out), "add overflow detected");
  CHECK(checked_mul_add(0, 0, 0, &out) && out == 0, "0*0+0 ok");

  if (failures == 0) {
    std::printf("ALL ARENA SPAN TESTS PASSED\n");
    return 0;
  }
  std::printf("%d FAILURES\n", failures);
  return 1;
}
