// T9.2 unit tests: StepGuard step-parameter validation and prompt-ID range
// scan. Pure host logic (no Level Zero device). OOB cases must fail CLOSED
// (return false + diagnostic) without crashing.
#include "../decode/runtime_258v.h"

#include <cstdio>

using namespace ainfer;
using Guard = AInferRuntime258V::StepGuard;

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
  char err[256];
  const uint32_t CTX = 2048;

  std::printf("--- StepGuard::check ---\n");
  CHECK(Guard::check(0, 1, 0, 1, CTX, err, sizeof(err)), "origin step valid");
  CHECK(Guard::check(2047, 2048, 248319, 512, CTX, err, sizeof(err)),
        "max-boundary step valid");
  CHECK(Guard::check(100, 132, 151644, 32, CTX, err, sizeof(err)),
        "mid-context chunk valid");

  CHECK(!Guard::check(-1, 1, 0, 1, CTX, err, sizeof(err)), "position -1 rejected");
  CHECK(!Guard::check(2048, 2048, 0, 1, CTX, err, sizeof(err)),
        "position == max_ctx rejected (KV OOB)");
  CHECK(!Guard::check(1000000, 1000001, 0, 1, CTX, err, sizeof(err)),
        "huge position rejected");
  CHECK(Guard::check(100, 100, 0, 1, CTX, err, sizeof(err)),
        "active_length == position valid (post-import state)");
  CHECK(!Guard::check(100, 99, 0, 1, CTX, err, sizeof(err)),
        "active_length < position rejected");
  CHECK(!Guard::check(0, 2049, 0, 1, CTX, err, sizeof(err)),
        "active_length > max_ctx rejected");
  CHECK(Guard::check(0, 0, 0, 1, CTX, err, sizeof(err)),
        "fresh-init (0,0) valid (degenerate but memory-safe)");
  CHECK(!Guard::check(5, 0, 0, 1, CTX, err, sizeof(err)),
        "active_length == 0 with position > 0 rejected");
  CHECK(!Guard::check(100, 132, -5, 32, CTX, err, sizeof(err)),
        "negative token_id rejected");
  CHECK(!Guard::check(100, 132, 248320, 32, CTX, err, sizeof(err)),
        "token_id == VOCAB_SIZE rejected (embed OOB)");
  CHECK(!Guard::check(100, 132, 1000000, 32, CTX, err, sizeof(err)),
        "huge token_id rejected");
  CHECK(!Guard::check(100, 132, 42, 0, CTX, err, sizeof(err)),
        "chunk_tokens == 0 rejected");
  CHECK(!Guard::check(100, 132, 42, 513, CTX, err, sizeof(err)),
        "chunk_tokens > MAX rejected");
  // Null-err tolerance: must still return the right verdict, no crash.
  CHECK(!Guard::check(-1, 1, 0, 1, CTX, nullptr, 0), "null err buffer tolerated");
  CHECK(Guard::check(5, 6, 7, 1, CTX, nullptr, 0), "null err valid case ok");

  std::printf("--- StepGuard::check_prompt_ids ---\n");
  const int good[] = {0, 151644, 248319, 198};
  CHECK(Guard::check_prompt_ids(good, 4, err, sizeof(err)), "valid prompt ids pass");
  const int neg[] = {10, -1, 20};
  CHECK(!Guard::check_prompt_ids(neg, 3, err, sizeof(err)), "negative id rejected");
  const int big[] = {10, 9999999};
  CHECK(!Guard::check_prompt_ids(big, 2, err, sizeof(err)), "huge id rejected");
  const int edge[] = {248319};
  CHECK(Guard::check_prompt_ids(edge, 1, err, sizeof(err)), "VOCAB-1 accepted");
  const int over[] = {248320};
  CHECK(!Guard::check_prompt_ids(over, 1, err, sizeof(err)), "VOCAB rejected");
  CHECK(Guard::check_prompt_ids(nullptr, 0, err, sizeof(err)), "empty scan vacuous pass");
  CHECK(!Guard::check_prompt_ids(nullptr, 3, err, sizeof(err)), "null ids rejected");

  if (failures == 0) {
    std::printf("ALL EXEC GUARD TESTS PASSED\n");
    return 0;
  }
  std::printf("%d FAILURES\n", failures);
  return 1;
}
