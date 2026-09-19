#include "runtime_258v.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ainfer;

int main(int argc, char **argv) {
  const char *binfer = (argc > 1) ? argv[1] : "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv = (argc > 2) ? argv[2] : "tools/kernels_258v/all_kernels.spv";

  AInferRuntime258V rt;
  if (!rt.init(binfer, spv, 2048)) {
    std::fprintf(stderr, "rt.init failed\n");
    return 1;
  }

  std::vector<int> prompt = {151644, 8948, 198, 2610, 525, 264, 10925, 151645};
  int first_tok = 0;
  rt.prefill(prompt, &first_tok);
  std::printf("First decode token: %d\n", first_tok);

  for (int i = 0; i < 5; ++i) {
    int next_tok = 0;
    rt.decode_step(&next_tok);
    std::printf("Step %d token: %d\n", i + 1, next_tok);
  }

  return 0;
}
