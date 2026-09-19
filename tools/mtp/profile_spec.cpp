#include "../decode/runtime_258v.h"
#include <chrono>
#include <cstdio>
#include <vector>

using namespace ainfer;

int main() {
  const char *model_path = "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv_path   = "tools/kernels_258v/all_kernels.spv";

  AInferRuntime258V runtime;
  if (!runtime.init(model_path, spv_path, 2048)) return 1;
  if (!runtime.init_speculative_verification()) return 1;

  std::vector<int> prompt = {151644, 8948, 198, 2610, 525, 264, 10925, 151645, 151648, 198, 936, 31920, 9467, 7, 82, 25, 1146, 1000, 1146, 512};
  int first_tok = 0;
  runtime.reset_state();
  runtime.prefill(prompt, &first_tok);

  // Measure single baseline decode steps
  std::vector<double> base_us;
  for (int i = 0; i < 10; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int tok = 0;
    runtime.decode_step(&tok);
    auto t1 = std::chrono::high_resolution_clock::now();
    base_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  // Reset and measure speculative steps with component breakdown
  runtime.reset_state();
  runtime.prefill(prompt, &first_tok);
  std::vector<double> spec_us;
  std::vector<int> n_emit;
  std::vector<bool> acc;
  std::vector<double> v_time, d_time, r_time;

  for (int i = 0; i < 10; ++i) {
    int tok1 = 0, tok2 = 0, nem = 0;
    bool accepted = false;
    double round_us = 0.0;
    runtime.speculative_step(&tok1, &tok2, &nem, &accepted, &round_us);
    spec_us.push_back(round_us);
    n_emit.push_back(nem);
    acc.push_back(accepted);
  }

  std::printf("--- Baseline Decode Steps ---\n");
  for (size_t i = 0; i < base_us.size(); ++i) {
    std::printf("  Step %zu: %.2f ms (%.2f tok/s)\n", i, base_us[i] / 1000.0, 1e6 / base_us[i]);
  }

  std::printf("\n--- Speculative Steps ---\n");
  for (size_t i = 0; i < spec_us.size(); ++i) {
    std::printf("  Round %zu: %.2f ms | Emitted: %d | Accepted: %s | Effective: %.2f tok/s\n",
                i, spec_us[i] / 1000.0, n_emit[i], acc[i] ? "YES" : "NO ", (n_emit[i] * 1e6) / spec_us[i]);
  }

  return 0;
}
