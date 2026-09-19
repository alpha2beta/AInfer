#include "../decode/runtime_258v.h"
#include <chrono>
#include <cstdio>
#include <vector>

using namespace ainfer;

// We can time the prefill chunk list for B=2 and compare it to decode step
int main() {
  const char *model_path = "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv_path   = "tools/kernels_258v/all_kernels.spv";

  AInferRuntime258V runtime;
  if (!runtime.init(model_path, spv_path, 2048)) return 1;

  double embed_ms = 0.0, layers_ms = 0.0, tail_ms = 0.0, step_ms = 0.0;
  std::vector<double> layer_times;
  runtime.profile_step_breakdown(embed_ms, layers_ms, tail_ms, step_ms, layer_times);

  std::printf("--- Baseline Decode (B=1) Breakdown ---\n");
  std::printf("  Embed:   %6.2f ms\n", embed_ms);
  std::printf("  Layers:  %6.2f ms (40 layers, avg %.2f ms/layer)\n", layers_ms, layers_ms / 40.0);
  std::printf("  Tail:    %6.2f ms\n", tail_ms);
  std::printf("  Total:   %6.2f ms\n\n", step_ms);

  // DeltaNet vs Full Attention layer times in baseline
  double delta_sum = 0.0, full_sum = 0.0;
  for (int l = 0; l < 40; ++l) {
    if (l == 3 || l == 7 || l == 11 || l == 15 || l == 19 || l == 23 || l == 27 || l == 31 || l == 35 || l == 39) {
      full_sum += layer_times[l];
    } else {
      delta_sum += layer_times[l];
    }
  }
  std::printf("  DeltaNet (30 layers): %6.2f ms (avg %.2f ms/layer)\n", delta_sum, delta_sum / 30.0);
  std::printf("  FullAttn (10 layers): %6.2f ms (avg %.2f ms/layer)\n\n", full_sum, full_sum / 10.0);

  return 0;
}
