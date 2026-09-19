// Profile prefill kernel categories on Arc 140V
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

  std::vector<int> test_B = {32, 64, 128, 256};

  std::printf("\n=================================================================\n");
  std::printf("--- Prefill Chunk Profiling (Arc 140V Xe2) ----------------------\n");
  std::printf("=================================================================\n");

  for (int B : test_B) {
    std::vector<int> prompt(B, 151644);
    rt.reset_state();
    int dummy = 0;
    // Warmup
    rt.prefill(prompt, &dummy);

    // Measure 3 runs
    const int RUNS = 3;
    double total_ms = 0.0;
    for (int r = 0; r < RUNS; ++r) {
      rt.reset_state();
      auto t0 = std::chrono::high_resolution_clock::now();
      rt.prefill(prompt, &dummy);
      auto t1 = std::chrono::high_resolution_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double avg_ms = total_ms / RUNS;
    double tps = (double)B / (avg_ms * 1e-3);
    std::printf("B = %-3d: %8.2f ms (Throughput: %6.2f tok/s, %6.2f ms/tok)\n",
                B, avg_ms, tps, avg_ms / B);
  }
  std::printf("=================================================================\n\n");

  rt.profile_prefill_breakdown(32);
  rt.profile_prefill_breakdown(128);
  rt.profile_prefill_breakdown(256);

  return 0;
}
