#include "../decode/runtime_258v.h"
#include <chrono>
#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace ainfer;

int main(int argc, char **argv) {
  const char *model_path = (argc > 1) ? argv[1] : "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv_path = (argc > 2) ? argv[2] : "tools/kernels_258v/all_kernels.spv";

  AInferRuntime258V rt;
  if (!rt.init(model_path, spv_path, 2048)) {
    std::fprintf(stderr, "Failed to init runtime\n");
    return 1;
  }

  const int B = 256;
  std::vector<int> prompt(B, 151644);
  rt.reset_state();
  int dummy = 0;
  // Warmup prefill to populate chunk activations and router top_idx
  rt.prefill(prompt, &dummy);

  std::printf("\n=================================================================\n");
  std::printf("--- MoE Microbenchmark at B = %d (Arc 140V Xe2) ----------------\n", B);
  std::printf("=================================================================\n");

  bool ok = rt.profile_moe_shootout(B);
  if (!ok) {
    std::fprintf(stderr, "WARNING: Numerical parity check failed in shootout!\n");
  }

  rt.profile_prefill_breakdown(B);

  return ok ? 0 : 1;
}
