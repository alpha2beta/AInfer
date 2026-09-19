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

  std::vector<int> prompt = {151644, 8948, 198, 2610, 525, 264, 10925, 151645};
  int first_tok = 0;
  rt.prefill(prompt, &first_tok);
  std::printf("Prefill first_tok: %d\n", first_tok);

  return 0;
}
