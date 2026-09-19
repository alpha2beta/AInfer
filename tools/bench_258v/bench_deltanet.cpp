#include "../decode/runtime_258v.h"

#include <chrono>
#include <cstdio>
#include <vector>

using namespace ainfer;

int main(int argc, char **argv) {
  const char *model_path = (argc > 1) ? argv[1] : "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv_path = (argc > 2) ? argv[2] : "tools/kernels_258v/all_kernels.spv";

  AInferRuntime258V rt;
  if (!rt.init(model_path, spv_path, 2048)) {
    std::fprintf(stderr, "Failed to init runtime\n");
    return 1;
  }

  std::vector<int> prompt = {1, 15043, 29892, 1024};
  int first_tok = 0;
  rt.prefill(prompt, &first_tok);

  double embed_ms, layers_ms, tail_ms, step_ms;
  std::vector<double> l_ms;
  rt.profile_step_breakdown(embed_ms, layers_ms, tail_ms, step_ms, l_ms);

  std::printf("=================================================================\n");
  std::printf("--- Layer-by-Layer Latency Profile (Arc 140V Xe2) ---\n");
  std::printf("=================================================================\n");
  for (int l = 0; l < TOTAL_LAYERS; ++l) {
    bool is_full = ((l + 1) % FULL_ATTN_INTERVAL == 0);
    std::printf("  Layer %2d (%s): %6.3f ms\n",
                l, is_full ? "Full-Attention" : "DeltaNet-Linear", l_ms[l]);
  }
  std::printf("=================================================================\n");

  return 0;
}
