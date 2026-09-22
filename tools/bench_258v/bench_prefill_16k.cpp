// T7.2 follow-up: real 16K prefill wall-clock measurement on Arc 140V.
// One warm-up establishes resident pages; the timed run measures only the
// in-memory runtime prefill. No MTP, decode, or extrapolated model is used.
#include "../decode/runtime_258v.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace ainfer;

int main(int argc, char **argv) {
  const char *binfer = argc > 1 ? argv[1] : "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv = argc > 2 ? argv[2] : "tools/kernels_258v/all_kernels.spv";
  const char *report = argc > 3 ? argv[3] : "tools/bench_258v/report_prefill_16k_258v.json";
  const int P = argc > 4 ? std::atoi(argv[4]) : 16384;
  const int MAXCTX = P + 256;

  std::vector<int> pattern = {151644, 8948, 198, 2610, 525, 264, 10925,
                              151645, 198, 248045, 74455, 198, 248068, 271,
                              248069, 271};
  std::vector<int> prompt;
  prompt.reserve(P);
  for (int i = 0; i < P; ++i) prompt.push_back(pattern[i % pattern.size()]);

  AInferRuntime258V runtime;
  auto init0 = std::chrono::steady_clock::now();
  if (!runtime.init(binfer, spv, MAXCTX)) {
    std::fprintf(stderr, "runtime init failed\n");
    return 1;
  }
  auto init1 = std::chrono::steady_clock::now();

  int first = 0;
  const bool skip_warm = argc > 5 && std::atoi(argv[5]) != 0;
  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  double warm_ms = 0.0;
  if (!skip_warm) {
    runtime.reset_state();
    auto warm0 = std::chrono::steady_clock::now();
    if (!runtime.prefill(prompt, &first)) {
      std::fprintf(stderr, "warm-up prefill failed\n");
      return 1;
    }
    auto warm1 = std::chrono::steady_clock::now();
    warm_ms = ms(warm0, warm1);
  }

  runtime.reset_state();
  auto run0 = std::chrono::steady_clock::now();
  if (!runtime.prefill(prompt, &first)) {
    std::fprintf(stderr, "timed prefill failed\n");
    return 1;
  }
  auto run1 = std::chrono::steady_clock::now();

  double init_ms = ms(init0, init1);
  double measured_ms = ms(run0, run1);
  double tok_s = P / (measured_ms * 1e-3);
  std::printf("%d prefill: %.2f ms (%.3f tok/s), warm %.2f ms, first=%d\n",
              P, measured_ms, tok_s, warm_ms, first);

  std::ofstream out(report);
  if (!out) return 1;
  out << "{\n"
      << "  \"task\": \"T7.2-16K-real\",\n"
      << "  \"device\": \"" << runtime.get_device_name() << "\",\n"
      << "  \"model\": \"" << binfer << "\",\n"
      << "  \"prompt_tokens\": " << P << ",\n"
      << "  \"max_context\": " << MAXCTX << ",\n"
      << "  \"init_ms\": " << init_ms << ",\n"
      << "  \"warmup_prefill_ms\": " << warm_ms << ",\n"
      << "  \"measured_prefill_ms\": " << measured_ms << ",\n"
      << "  \"measured_prefill_tok_per_s\": " << tok_s << ",\n"
      << "  \"first_token\": " << first << ",\n"
      << "  \"method\": \"real_wall_clock_single_process_bf16_kv_no_mtp\",\n"
      << "  \"status\": \"PASSED\"\n"
      << "}\n";
  return 0;
}
