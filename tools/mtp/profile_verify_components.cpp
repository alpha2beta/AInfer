// profile_verify_components.cpp — Detailed breakdown of verification execution on Arc 140V

#include "../decode/runtime_258v.h"

#include <chrono>
#include <cstdio>
#include <vector>

using namespace ainfer;

int main() {
  const char *model_path = "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv_path   = "tools/kernels_258v/all_kernels.spv";

  AInferRuntime258V runtime;
  if (!runtime.init(model_path, spv_path, 2048)) {
    std::fprintf(stderr, "Init base runtime failed\n");
    return 1;
  }
  if (!runtime.init_speculative_verification()) {
    std::fprintf(stderr, "Init speculative verification failed\n");
    return 1;
  }

  std::vector<int> prompt = {151644, 8948, 198, 2610, 525, 264, 10925, 151645, 151648, 198, 936, 31920, 9467, 7, 82, 25, 1146, 1000, 1146, 512};
  int first_tok = 0;
  runtime.reset_state();
  runtime.prefill(prompt, &first_tok);

  // Warmup speculative step
  int t1, t2, nem;
  bool acc;
  double round_us;
  runtime.speculative_step(&t1, &t2, &nem, &acc, &round_us);

  // Now measure multiple speculative steps
  std::printf("=================================================================\n");
  std::printf("Speculative Step Measurement (10 rounds):\n");
  std::printf("=================================================================\n");

  double total_round_ms = 0.0;
  for (int i = 0; i < 10; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    runtime.speculative_step(&t1, &t2, &nem, &acc, &round_us);
    auto t1_time = std::chrono::high_resolution_clock::now();
    double step_ms = std::chrono::duration<double, std::milli>(t1_time - t0).count();
    total_round_ms += step_ms;
    std::printf("  Round %d: %6.2f ms | Accepted: %s | Emitted: %d\n",
                i, step_ms, acc ? "YES" : "NO ", nem);
  }
  std::printf("Average Round Latency: %.2f ms\n\n", total_round_ms / 10.0);

  return 0;
}
