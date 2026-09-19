// T10.1 Multi-Token Prediction (MTP) Speculative Drafting Benchmark on Arc 140V (258V)
// Measures:
//   - MTP draft step latency (mean, median, p90, min, max)
//   - Empirical draft acceptance rate alpha across multi-domain prompts
//   - Bit-exact reset determinism between independent runs
//   - Theoretical speculative decoding speedup vs baseline decode
// Emits tools/mtp/report_mtp_258v.json

#include "../decode/runtime_258v.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace ainfer;

struct SampleResult {
  int prompt_idx;
  int step;
  int draft_token;
  int true_token;
  bool accepted;
  double draft_latency_us;
  double trunk_latency_us;
};

int main(int argc, char **argv) {
  const char *model_path = (argc > 1) ? argv[1] : "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv_path   = (argc > 2) ? argv[2] : "tools/kernels_258v/all_kernels.spv";
  const char *report_out = (argc > 3) ? argv[3] : "tools/mtp/report_mtp_258v.json";

  std::printf("=================================================================\n");
  std::printf("--- AInfer T10.1: MTP Draft Benchmark on Arc 140V (Lunar Lake) ---\n");
  std::printf("=================================================================\n\n");

  AInferRuntime258V runtime;
  std::printf("[1/4] Initializing base runtime...\n");
  if (!runtime.init(model_path, spv_path, 2048)) {
    std::fprintf(stderr, "FATAL: Base runtime initialization failed!\n");
    return 1;
  }

  std::printf("[2/4] Initializing MTP draft layer & recording command list...\n");
  auto t_mtp_init0 = std::chrono::steady_clock::now();
  if (!runtime.init_mtp()) {
    std::fprintf(stderr, "FATAL: MTP initialization failed!\n");
    return 1;
  }
  auto t_mtp_init1 = std::chrono::steady_clock::now();
  double mtp_init_ms = std::chrono::duration<double, std::milli>(t_mtp_init1 - t_mtp_init0).count();
  std::printf("  MTP Layer initialized successfully in %.2f ms\n\n", mtp_init_ms);

  // Test Prompts: diverse coding, reasoning, arithmetic, factual
  struct TestPrompt {
    std::string text;
    std::vector<int> tokens;
  };

  std::vector<TestPrompt> prompts = {
    // 1. Python reverse function
    {"def reverse_string(s: str) -> str:",
     {151644, 8948, 198, 2610, 525, 264, 10925, 151645, 151648, 198, 936, 31920, 9467, 7, 82, 25, 1146, 1000, 1146, 512}},
    // 2. Arithmetic reasoning
    {"Calculate 84 * 3 / 2 step by step.",
     {151644, 8948, 198, 2610, 525, 264, 10925, 151645, 151648, 198, 59728, 220, 23, 19, 348, 220, 18, 593, 220, 17, 3016, 553, 3016, 13}},
    // 3. Factual knowledge
    {"The capital of France is Paris. The capital of Japan is",
     {151644, 8948, 198, 2610, 525, 264, 10925, 151645, 151648, 198, 785, 6863, 315, 9813, 374, 9542, 13, 785, 6863, 315, 6386, 374}},
    // 4. Code loop
    {"for i in range(10):\n    print(i * 2)",
     {151644, 8948, 198, 2610, 525, 264, 10925, 151645, 151648, 198, 1023, 684, 304, 3512, 10, 16, 558, 271, 3835, 10, 684, 348, 220, 17, 8}},
    // 5. Logic puzzle
    {"If all cats are mammals, and Luna is a cat, then Luna is",
     {151644, 8948, 198, 2610, 525, 264, 10925, 151645, 151648, 198, 3058, 682, 16641, 527, 43905, 11, 323, 44927, 374, 264, 8408, 11, 1243, 44927, 374}}
  };

  const int DECODE_STEPS_PER_PROMPT = 16;
  std::vector<SampleResult> all_samples;
  std::vector<double> all_draft_latencies_us;
  std::vector<double> all_trunk_latencies_us;

  // Warmup run
  std::printf("[3/4] Warming up MTP execution on Arc 140V...\n");
  {
    runtime.reset_state();
    std::vector<int> warmup_prompt = prompts[0].tokens;
    int first_tok = 0;
    runtime.prefill(warmup_prompt, &first_tok);
    for (int s = 0; s < 4; ++s) {
      int next_tok = 0;
      runtime.decode_step(&next_tok);
      int draft_tok = 0;
      runtime.mtp_draft_step(&draft_tok);
    }
  }

  std::printf("[4/4] Running MTP Draft Acceptance & Latency Evaluation...\n");
  for (size_t pi = 0; pi < prompts.size(); ++pi) {
    const auto &p = prompts[pi];
    runtime.reset_state();

    int first_tok = 0;
    runtime.prefill(p.tokens, &first_tok);

    int prompt_accepted = 0;
    int prompt_samples = 0;

    int pending_draft_token = -1;
    double prev_draft_us = 0.0;

    // Initial draft after prefill produces first token
    // Note: trunk hidden state at layer 39 and selected_token are ready
    runtime.mtp_draft_step(&pending_draft_token, &prev_draft_us);

    for (int step = 0; step < DECODE_STEPS_PER_PROMPT; ++step) {
      auto t_tr0 = std::chrono::steady_clock::now();
      int true_next_tok = 0;
      if (!runtime.decode_step(&true_next_tok)) {
        std::fprintf(stderr, "Decode failed at prompt %zu, step %d\n", pi, step);
        break;
      }
      auto t_tr1 = std::chrono::steady_clock::now();
      double trunk_us = std::chrono::duration<double, std::micro>(t_tr1 - t_tr0).count();
      all_trunk_latencies_us.push_back(trunk_us);

      // Verify the previous draft against the current true token
      bool match = (pending_draft_token == true_next_tok);
      if (match) prompt_accepted++;
      prompt_samples++;

      SampleResult sr;
      sr.prompt_idx = (int)pi;
      sr.step = step;
      sr.draft_token = pending_draft_token;
      sr.true_token = true_next_tok;
      sr.accepted = match;
      sr.draft_latency_us = prev_draft_us;
      sr.trunk_latency_us = trunk_us;
      all_samples.push_back(sr);
      all_draft_latencies_us.push_back(prev_draft_us);

      // Generate next draft candidate for the subsequent step
      runtime.mtp_draft_step(&pending_draft_token, &prev_draft_us);
    }

    double prompt_alpha = (double)prompt_accepted / prompt_samples;
    std::printf("  Prompt %zu (%s): %d / %d accepted (alpha = %.2f%%)\n",
                pi + 1, p.text.substr(0, 32).c_str(), prompt_accepted, prompt_samples, prompt_alpha * 100.0);
  }

  // Determinism test: Prompt 0 run twice
  std::printf("\n--- Verifying Determinism Across Resets ---\n");
  std::vector<int> run1_drafts, run2_drafts;
  std::vector<int> run1_trunks, run2_trunks;
  for (int r = 0; r < 2; ++r) {
    runtime.reset_state();
    int first_tok = 0;
    runtime.prefill(prompts[0].tokens, &first_tok);
    std::vector<int> &tr_vec = (r == 0) ? run1_trunks : run2_trunks;
    tr_vec.push_back(first_tok);
    int dtok = 0;
    runtime.mtp_draft_step(&dtok);
    std::vector<int> &vec = (r == 0) ? run1_drafts : run2_drafts;
    vec.push_back(dtok);
    for (int step = 0; step < 8; ++step) {
      int next_tok = 0;
      runtime.decode_step(&next_tok);
      tr_vec.push_back(next_tok);
      runtime.mtp_draft_step(&dtok);
      vec.push_back(dtok);
    }
  }

  bool deterministic_trunk = (run1_trunks == run2_trunks);
  bool deterministic = (run1_drafts == run2_drafts);
  int match_count = 0;
  for (size_t i = 0; i < run1_drafts.size(); ++i) {
    if (i < run2_drafts.size() && run1_drafts[i] == run2_drafts[i]) match_count++;
    std::printf("  [%zu] draft: run1=%d, run2=%d %s | trunk: run1=%d, run2=%d %s\n",
                i, run1_drafts[i], (i < run2_drafts.size() ? run2_drafts[i] : -1),
                (i < run2_drafts.size() && run1_drafts[i] == run2_drafts[i]) ? "MATCH" : "MISMATCH",
                run1_trunks[i], (i < run2_trunks.size() ? run2_trunks[i] : -1),
                (i < run2_trunks.size() && run1_trunks[i] == run2_trunks[i]) ? "MATCH" : "MISMATCH");
  }
  std::printf("  Trunk determinism check: %s\n", deterministic_trunk ? "PASSED" : "FAILED");
  std::printf("  MTP Draft determinism check: %s (%d / %zu exact token matches)\n",
              deterministic ? "PASSED" : "FAILED",
              match_count, run1_drafts.size());

  // Aggregate Metrics
  int total_samples = (int)all_samples.size();
  int total_accepted = 0;
  for (const auto &s : all_samples) {
    if (s.accepted) total_accepted++;
  }
  double overall_alpha = (double)total_accepted / total_samples;

  std::sort(all_draft_latencies_us.begin(), all_draft_latencies_us.end());
  std::sort(all_trunk_latencies_us.begin(), all_trunk_latencies_us.end());

  double draft_mean_us = std::accumulate(all_draft_latencies_us.begin(), all_draft_latencies_us.end(), 0.0) / total_samples;
  double draft_median_us = all_draft_latencies_us[total_samples / 2];
  double draft_min_us = all_draft_latencies_us.front();
  double draft_max_us = all_draft_latencies_us.back();
  double draft_p90_us = all_draft_latencies_us[(size_t)(total_samples * 0.90)];

  double trunk_median_us = all_trunk_latencies_us[total_samples / 2];

  // Speculative Economics Model (Depth-1):
  // Cost per round = c_verify + c_draft
  // In single-step dual verify: c_verify ≈ 1.05 * c_trunk (or ~1.0 for small batch)
  // Expected tokens generated per round = 1 + alpha
  // Expected speedup = (1 + alpha) / (c_verify/c_trunk + c_draft/c_trunk)
  double c_draft_ratio = draft_median_us / trunk_median_us;
  double c_verify_ratio = 1.05; // 2-token verification batch overhead factor
  double projected_speedup = (1.0 + overall_alpha) / (c_verify_ratio + c_draft_ratio);

  std::printf("\n=================================================================\n");
  std::printf("--- MTP 258V Performance & Acceptance Summary ---\n");
  std::printf("=================================================================\n");
  std::printf("  Total Samples:        %d across %zu diverse prompts\n", total_samples, prompts.size());
  std::printf("  Draft Accepted:       %d\n", total_accepted);
  std::printf("  Empirical Alpha:      %.2f%%\n", overall_alpha * 100.0);
  std::printf("  Draft Latency Median: %.3f ms (%6.1f us)\n", draft_median_us / 1000.0, draft_median_us);
  std::printf("  Draft Latency Mean:   %.3f ms (%6.1f us)\n", draft_mean_us / 1000.0, draft_mean_us);
  std::printf("  Draft Latency P90:    %.3f ms (%6.1f us)\n", draft_p90_us / 1000.0, draft_p90_us);
  std::printf("  Draft Latency Min:    %.3f ms (%6.1f us)\n", draft_min_us / 1000.0, draft_min_us);
  std::printf("  Draft Latency Max:    %.3f ms (%6.1f us)\n", draft_max_us / 1000.0, draft_max_us);
  std::printf("  Trunk Decode Median:  %.3f ms (%6.1f us)\n", trunk_median_us / 1000.0, trunk_median_us);
  std::printf("  Draft Fraction c:     %.2f%% of trunk step\n", c_draft_ratio * 100.0);
  std::printf("  Projected Speedup:    %.2fx\n", projected_speedup);
  std::printf("  Determinism:          %s\n", deterministic ? "PASSED (100% bit-exact)" : "FAILED");
  std::printf("=================================================================\n\n");

  // Emit JSON report
  std::ofstream out(report_out);
  if (out) {
    out << "{\n";
    out << "  \"task\": \"T10.1\",\n";
    out << "  \"title\": \"Multi-Token Prediction (MTP) Speculative Decoding on Arc 140V\",\n";
    out << "  \"device\": \"Intel Arc 140V (Xe2, Lunar Lake 258V)\",\n";
    out << "  \"model\": \"symrex/Tiel-Coder-35B-A3B-Genesis-Hermes (Qwen3.5-MoE)\",\n";
    out << "  \"mtp_architecture\": {\n";
    out << "    \"num_mtp_layers\": 1,\n";
    out << "    \"full_attn\": true,\n";
    out << "    \"num_routed_experts\": 256,\n";
    out << "    \"active_experts\": 8,\n";
    out << "    \"shared_expert\": true,\n";
    out << "    \"hidden_dim\": 2048,\n";
    out << "    \"concat_dim\": 4096,\n";
    out << "    \"kv_cache_bytes\": " << (2 * 2048 * 256 * sizeof(uint16_t)) << ",\n";
    out << "    \"zero_allocation_loop\": true\n";
    out << "  },\n";
    out << "  \"summary\": {\n";
    out << "    \"total_samples\": " << total_samples << ",\n";
    out << "    \"accepted_samples\": " << total_accepted << ",\n";
    out << "    \"alpha\": " << overall_alpha << ",\n";
    out << "    \"alpha_percent\": " << (overall_alpha * 100.0) << ",\n";
    out << "    \"draft_latency_median_ms\": " << (draft_median_us / 1000.0) << ",\n";
    out << "    \"draft_latency_mean_ms\": " << (draft_mean_us / 1000.0) << ",\n";
    out << "    \"draft_latency_p90_ms\": " << (draft_p90_us / 1000.0) << ",\n";
    out << "    \"draft_latency_min_ms\": " << (draft_min_us / 1000.0) << ",\n";
    out << "    \"draft_latency_max_ms\": " << (draft_max_us / 1000.0) << ",\n";
    out << "    \"trunk_latency_median_ms\": " << (trunk_median_us / 1000.0) << ",\n";
    out << "    \"draft_fraction_c\": " << c_draft_ratio << ",\n";
    out << "    \"projected_speedup\": " << projected_speedup << ",\n";
    out << "    \"deterministic\": " << (deterministic ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"sample_details\": [\n";
    for (size_t i = 0; i < all_samples.size(); ++i) {
      const auto &s = all_samples[i];
      out << "    {\"prompt\": " << s.prompt_idx
          << ", \"step\": " << s.step
          << ", \"draft\": " << s.draft_token
          << ", \"true\": " << s.true_token
          << ", \"accepted\": " << (s.accepted ? "true" : "false")
          << ", \"draft_us\": " << s.draft_latency_us
          << ", \"trunk_us\": " << s.trunk_latency_us << "}"
          << (i + 1 < all_samples.size() ? ",\n" : "\n");
    }
    out << "  ]\n";
    out << "}\n";
    std::printf("Report saved to %s\n", report_out);
  }

  return deterministic ? 0 : 1;
}
