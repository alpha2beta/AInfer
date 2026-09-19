#include "runtime_258v.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ainfer;

int main() {
  const char *binfer = "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv = "tools/kernels_258v/all_kernels.spv";

  AInferRuntime258V rt;
  if (!rt.init(binfer, spv, 2048)) {
    std::fprintf(stderr, "rt.init failed\n");
    return 1;
  }

  // Run 1 prefill step
  std::vector<int> prompt = {151644};
  int first_tok = 0;
  rt.prefill(prompt, &first_tok);
  std::printf("Prompt [151644] -> first token: %d\n", first_tok);

  return 0;
}
