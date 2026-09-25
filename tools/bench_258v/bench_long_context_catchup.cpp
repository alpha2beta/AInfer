// Benchmark Long-Context (6.7K) Prefill & Decode Catchup
// Directly compares performance against baseline and llama.cpp figures in claim_correction.md.
// Emits tools/bench_258v/report_long_context_catchup.json.

#include "../decode/runtime_258v.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace ainfer;

int main(int argc, char **argv) {
  const char *binfer = (argc > 1) ? argv[1] : "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv = (argc > 2) ? argv[2] : "tools/kernels_258v/all_kernels.spv";
  const char *report_path = (argc > 3) ? argv[3] : "tools/bench_258v/report_long_context_catchup.json";
  const int P = (argc > 4) ? std::atoi(argv[4]) : 6720;
  const int DECODE_TOKENS = (argc > 5) ? std::atoi(argv[5]) : 32;
  const int MAX_CTX = 8192;

  std::printf("=================================================================\n");
  std::printf("--- AInfer Long-Context Catchup Benchmark (P=%d, Arc 140V) ---\n", P);
  std::printf("=================================================================\n");
  std::printf("  Model:         %s\n", binfer);
  std::printf("  SPIR-V:        %s\n", spv);
  std::printf("  Prompt Tokens: %d\n", P);
  std::printf("  Decode Tokens: %d\n", DECODE_TOKENS);
  std::printf("  Max Context:   %d\n", MAX_CTX);
  std::printf("-----------------------------------------------------------------\n\n");

  std::vector<int> sample_pattern = {
      151644, 8948, 198, 2610, 525, 264, 10925, 151645,
      198, 248045, 74455, 198, 248068, 271, 248069, 271
  };
  std::vector<int> prompt;
  prompt.reserve(P);
  for (int i = 0; i < P; ++i) {
    prompt.push_back(sample_pattern[i % sample_pattern.size()]);
  }

  AInferRuntime258V rt;
  std::printf("[1/4] Initializing Unified Runtime (max_ctx=%d)...\n", MAX_CTX);
  auto t_init0 = std::chrono::steady_clock::now();
  if (!rt.init(binfer, spv, MAX_CTX)) {
    std::fprintf(stderr, "FATAL: rt.init failed\n");
    return 1;
  }
  auto t_init1 = std::chrono::steady_clock::now();
  double init_ms = std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();
  std::printf("  Runtime initialized in %.2f ms (%.2f s)\n", init_ms, init_ms / 1000.0);
  std::printf("  KV Cache Arena: %.2f MiB (10 full layers @ %d ctx)\n\n",
              rt.get_kv_cache_bytes() / (1024.0 * 1024.0), MAX_CTX);

  // [2/4] Warm-up run at long context to ensure static GPU pages are fully resident
  std::printf("[2/4] Warm-up Prefill at P=%d...\n", P);
  rt.reset_state();
  int warmup_first_token = 0;
  auto t_warm0 = std::chrono::steady_clock::now();
  if (!rt.prefill(prompt, &warmup_first_token)) {
    std::fprintf(stderr, "FATAL: Warm-up prefill failed\n");
    return 1;
  }
  auto t_warm1 = std::chrono::steady_clock::now();
  double warm_ms = std::chrono::duration<double, std::milli>(t_warm1 - t_warm0).count();
  std::printf("  Warm-up completed in %.2f ms (%.2f tok/s), first token=%d\n\n",
              warm_ms, (double)P / (warm_ms * 1e-3), warmup_first_token);

  // [3/4] Measured Long-Context Prefill & Greedy Decode
  std::printf("[3/4] Measured Long-Context Prefill & Greedy Decode (%d runs)...\n", 2);
  const int MEASURED_RUNS = 2;
  std::vector<double> prefill_latencies_ms;
  std::vector<double> prefill_tok_per_s;
  std::vector<double> decode_tok_per_s;
  std::vector<int> greedy_tokens;

  for (int r = 0; r < MEASURED_RUNS; ++r) {
    rt.reset_state();

    // Measured Prefill
    int first_token = 0;
    auto t_p0 = std::chrono::steady_clock::now();
    if (!rt.prefill(prompt, &first_token)) {
      std::fprintf(stderr, "FATAL: Measured prefill failed on run %d\n", r);
      return 1;
    }
    auto t_p1 = std::chrono::steady_clock::now();
    double pref_ms = std::chrono::duration<double, std::milli>(t_p1 - t_p0).count();
    double pref_tps = (double)P / (pref_ms * 1e-3);
    prefill_latencies_ms.push_back(pref_ms);
    prefill_tok_per_s.push_back(pref_tps);

    // Measured Greedy Decode at Context P
    std::vector<int> tokens_this_run;
    tokens_this_run.push_back(first_token);

    auto t_dec0 = std::chrono::steady_clock::now();
    for (int step = 1; step < DECODE_TOKENS; ++step) {
      int next_tok = 0;
      if (!rt.decode_step(&next_tok)) {
        std::fprintf(stderr, "FATAL: Decode step %d failed on run %d\n", step, r);
        return 1;
      }
      tokens_this_run.push_back(next_tok);
    }
    auto t_dec1 = std::chrono::steady_clock::now();
    double dec_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
    double dec_tps = (double)(DECODE_TOKENS - 1) / (dec_ms * 1e-3);
    decode_tok_per_s.push_back(dec_tps);

    if (r == 0) {
      greedy_tokens = tokens_this_run;
    }

    std::printf("  Run %d: Prefill %.2f ms (%.2f tok/s) | Decode %d tok in %.2f ms (%.2f tok/s)\n",
                r + 1, pref_ms, pref_tps, DECODE_TOKENS - 1, dec_ms, dec_tps);
  }

  double avg_prefill_ms = std::accumulate(prefill_latencies_ms.begin(), prefill_latencies_ms.end(), 0.0) / MEASURED_RUNS;
  double avg_prefill_tps = std::accumulate(prefill_tok_per_s.begin(), prefill_tok_per_s.end(), 0.0) / MEASURED_RUNS;
  double avg_decode_tps = std::accumulate(decode_tok_per_s.begin(), decode_tok_per_s.end(), 0.0) / MEASURED_RUNS;

  std::printf("\n  Average Long-Context Prefill: %.2f ms (%.2f tok/s)\n", avg_prefill_ms, avg_prefill_tps);
  std::printf("  Average Long-Context Decode:  %.2f tok/s (%.2f ms/tok)\n\n",
              avg_decode_tps, 1000.0 / avg_decode_tps);

  // [4/4] Speculative Verification at Long Context
  std::printf("[4/4] Initializing MTP Speculative Verification at P=%d...\n", P);
  double spec_decode_tps = 0.0;
  double spec_pref_ms = 0.0;
  double spec_speedup = 0.0;
  double acceptance_rate = 0.0;
  bool spec_parity = false;
  std::vector<int> spec_tokens;

  if (rt.init_speculative_verification()) {
    std::printf("  MTP verification engine initialized.\n");

    if (rt.generate_speculative(prompt, DECODE_TOKENS, spec_tokens, &spec_pref_ms, &spec_decode_tps, &acceptance_rate)) {
      spec_speedup = spec_decode_tps / avg_decode_tps;
      spec_parity = (spec_tokens.size() == greedy_tokens.size());
      if (spec_parity) {
        for (size_t i = 0; i < spec_tokens.size(); ++i) {
          if (spec_tokens[i] != greedy_tokens[i]) {
            spec_parity = false;
            break;
          }
        }
      }
      std::printf("  Speculative Decode at P=%d: %.2f tok/s (%.2fx vs greedy, alpha=%.1f%%, parity=%s)\n\n",
                  P, spec_decode_tps, spec_speedup, acceptance_rate * 100.0, spec_parity ? "MATCH" : "DIFF");
    } else {
      std::fprintf(stderr, "  WARNING: generate_speculative failed\n");
    }
  }

  // Baseline Comparison Table against claim_correction.md
  const double PRIOR_A4_PREFILL = 84.3;
  const double PRIOR_A4_DECODE = 9.4;
  const double LLAMA_PREFILL = 183.2;
  const double LLAMA_DECODE = 23.0;

  double prefill_vs_prior = avg_prefill_tps / PRIOR_A4_PREFILL;
  double prefill_vs_llama = avg_prefill_tps / LLAMA_PREFILL;
  double decode_vs_prior = avg_decode_tps / PRIOR_A4_DECODE;
  double decode_vs_llama = avg_decode_tps / LLAMA_DECODE;

  std::printf("========================================================================================\n");
  std::printf("--- LONG-CONTEXT PERFORMANCE CATCHUP EVALUATION (P ~ 6.7K) ----------------------------\n");
  std::printf("========================================================================================\n");
  std::printf("%-24s | %-14s | %-14s | %-14s | %-10s\n",
              "Metric", "Prior AInfer", "llama.cpp", "New AInfer", "vs llama.cpp");
  std::printf("----------------------------------------------------------------------------------------\n");
  std::printf("%-24s | %10.1f tps | %10.1f tps | %10.1f tps | %7.2fx %s\n",
              "Prefill Throughput", PRIOR_A4_PREFILL, LLAMA_PREFILL, avg_prefill_tps,
              prefill_vs_llama, (prefill_vs_llama >= 1.0 ? "WIN" : "GAP"));
  std::printf("%-24s | %10.1f tps | %10.1f tps | %10.1f tps | %7.2fx %s\n",
              "Greedy Decode", PRIOR_A4_DECODE, LLAMA_DECODE, avg_decode_tps,
              decode_vs_llama, (decode_vs_llama >= 1.0 ? "WIN" : "GAP"));
  if (spec_decode_tps > 0.0) {
    double spec_vs_llama = spec_decode_tps / LLAMA_DECODE;
    std::printf("%-24s | %14s | %14s | %10.1f tps | %7.2fx %s\n",
                "Speculative Decode", "N/A", "N/A", spec_decode_tps,
                spec_vs_llama, (spec_vs_llama >= 1.0 ? "WIN" : "GAP"));
  }
  std::printf("========================================================================================\n\n");

  // Output JSON report
  std::ofstream out(report_path);
  out << "{\n"
      << "  \"task\": \"I3.7-long-context-catchup\",\n"
      << "  \"device\": \"" << rt.get_device_name() << "\",\n"
      << "  \"model\": \"" << binfer << "\",\n"
      << "  \"prompt_tokens\": " << P << ",\n"
      << "  \"decode_tokens\": " << DECODE_TOKENS << ",\n"
      << "  \"max_context\": " << MAX_CTX << ",\n"
      << "  \"init_ms\": " << init_ms << ",\n"
      << "  \"warmup_prefill_ms\": " << warm_ms << ",\n"
      << "  \"measured_prefill_ms\": " << avg_prefill_ms << ",\n"
      << "  \"measured_prefill_tok_per_s\": " << avg_prefill_tps << ",\n"
      << "  \"measured_greedy_decode_tok_per_s\": " << avg_decode_tps << ",\n"
      << "  \"measured_spec_decode_tok_per_s\": " << spec_decode_tps << ",\n"
      << "  \"spec_acceptance_rate_pct\": " << acceptance_rate << ",\n"
      << "  \"spec_parity\": " << (spec_parity ? "true" : "false") << ",\n"
      << "  \"claim_comparison\": {\n"
      << "    \"prior_ainfer_prefill_tok_per_s\": " << PRIOR_A4_PREFILL << ",\n"
      << "    \"llama_cpp_prefill_tok_per_s\": " << LLAMA_PREFILL << ",\n"
      << "    \"prefill_speedup_vs_prior\": " << prefill_vs_prior << ",\n"
      << "    \"prefill_ratio_vs_llama\": " << prefill_vs_llama << ",\n"
      << "    \"prior_ainfer_decode_tok_per_s\": " << PRIOR_A4_DECODE << ",\n"
      << "    \"llama_cpp_decode_tok_per_s\": " << LLAMA_DECODE << ",\n"
      << "    \"decode_speedup_vs_prior\": " << decode_vs_prior << ",\n"
      << "    \"decode_ratio_vs_llama\": " << decode_vs_llama << "\n"
      << "  }\n"
      << "}\n";

  std::printf("Written long-context report to %s\n", report_path);
  return 0;
}
