// Benchmark prefill throughput scaling across prompt lengths
#include "../decode/runtime_258v.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace ainfer;

int main(int argc, char **argv) {
  const char *binfer = (argc > 1) ? argv[1] : "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv = (argc > 2) ? argv[2] : "tools/kernels_258v/all_kernels.spv";
  const char *report = (argc > 3) ? argv[3] : "tools/bench_258v/report_prefill_scaling.json";

  AInferRuntime258V rt;
  if (!rt.init(binfer, spv, 2048)) {
    std::fprintf(stderr, "rt.init failed\n");
    return 1;
  }

  std::vector<int> sample_tokens = {151644, 8948, 198, 2610, 525, 264, 10925, 151645};
  std::vector<int> test_lengths = {8, 16, 32, 64, 128, 256, 512, 1024, 2048};

  std::printf("\n=================================================================\n");
  std::printf("--- AInfer Prefill Scaling Benchmark (Arc 140V Xe2) -------------\n");
  std::printf("=================================================================\n");
  std::printf("%-8s | %-12s | %-16s | %-12s\n", "Prompt", "Latency", "Throughput", "ms / tok");
  std::printf("-----------------------------------------------------------------\n");

  std::ofstream out(report);
  out << "{\n  \"device\": \"" << rt.get_device_name() << "\",\n  \"results\": [\n";

  for (size_t i = 0; i < test_lengths.size(); ++i) {
    int P = test_lengths[i];
    std::vector<int> prompt;
    while ((int)prompt.size() < P) {
      for (int t : sample_tokens) {
        if ((int)prompt.size() < P) prompt.push_back(t);
      }
    }

    // Warmup
    rt.reset_state();
    int dummy = 0;
    rt.prefill(prompt, &dummy);

    // 3 measured runs
    double total_ms = 0.0;
    const int RUNS = 3;
    for (int r = 0; r < RUNS; ++r) {
      rt.reset_state();
      auto t0 = std::chrono::steady_clock::now();
      int out_tok = 0;
      rt.prefill(prompt, &out_tok);
      auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double avg_ms = total_ms / RUNS;
    double tps = (double)P / (avg_ms * 1e-3);
    double ms_per_tok = avg_ms / P;

    std::printf("P = %-4d | %8.2f ms | %10.2f tok/s | %8.2f ms/tok\n",
                P, avg_ms, tps, ms_per_tok);

    out << "    {\n"
        << "      \"prompt_tokens\": " << P << ",\n"
        << "      \"prefill_latency_ms\": " << avg_ms << ",\n"
        << "      \"prefill_tok_per_s\": " << tps << ",\n"
        << "      \"ms_per_tok\": " << ms_per_tok << "\n"
        << "    }" << (i + 1 < test_lengths.size() ? ",\n" : "\n");
  }

  out << "  ]\n}\n";
  out.close();

  std::printf("=================================================================\n");
  std::printf("Report saved to: %s\n\n", report);

  return 0;
}
