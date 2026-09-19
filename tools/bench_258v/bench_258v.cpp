// AInfer High-Precision Benchmark Harness for Intel Core Ultra 7 258V (Arc 140V)
// Measures isolated timing fields per Task T7.1:
//   - model load time (ms)
//   - cold TTFT (ms)
//   - warm TTFT (ms)
//   - prefill time (ms) and prefill throughput (tok/s)
//   - first decode latency (ms)
//   - per-token decode step latencies
//   - sustained decode throughput (tok/s)
//   - p50/p90/p95/p99 inter-token jitter (ms)
//   - static arena memory footprint and RSS telemetry

#include "../decode/runtime_258v.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <vector>

using namespace ainfer;

static long get_current_rss_kb() {
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    return ru.ru_maxrss;
  }
  return 0;
}

struct LatencyStats {
  double min_ms = 0.0;
  double max_ms = 0.0;
  double mean_ms = 0.0;
  double stddev_ms = 0.0;
  double p50_ms = 0.0;
  double p90_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
};

static LatencyStats compute_stats(std::vector<double> latencies_ms) {
  LatencyStats st;
  if (latencies_ms.empty()) return st;

  std::sort(latencies_ms.begin(), latencies_ms.end());
  size_t n = latencies_ms.size();

  st.min_ms = latencies_ms.front();
  st.max_ms = latencies_ms.back();

  double sum = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
  st.mean_ms = sum / n;

  double sq_sum = 0.0;
  for (double x : latencies_ms) {
    sq_sum += (x - st.mean_ms) * (x - st.mean_ms);
  }
  st.stddev_ms = std::sqrt(sq_sum / n);

  auto get_percentile = [&](double pct) -> double {
    if (n == 1) return latencies_ms[0];
    double rank = (pct / 100.0) * (n - 1);
    size_t low = (size_t)std::floor(rank);
    size_t high = (size_t)std::ceil(rank);
    double weight = rank - low;
    return latencies_ms[low] * (1.0 - weight) + latencies_ms[high] * weight;
  };

  st.p50_ms = get_percentile(50.0);
  st.p90_ms = get_percentile(90.0);
  st.p95_ms = get_percentile(95.0);
  st.p99_ms = get_percentile(99.0);

  return st;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "Usage: %s <model.binfer> [options]\n"
                 "Options:\n"
                 "  --spv=<file>             SPIR-V module (default: tools/kernels_258v/all_kernels.spv)\n"
                 "  --max-ctx=<N>            Max context capacity (default: 4096)\n"
                 "  --ids=<csv>              Prompt token IDs (default: 16-tok greeting)\n"
                 "  --ids-file=<path>        Read prompt token IDs from file\n"
                 "  --prompt-len=<N>         Generate synthetic repeated prompt of length N\n"
                 "  --decode-tokens=<N>      Number of tokens to decode (default: 64)\n"
                 "  --warmup-runs=<N>        Warmup runs (default: 1)\n"
                 "  --measured-runs=<N>      Measured runs for steady state (default: 3)\n"
                 "  --report=<path>          Output JSON report path\n",
                 argv[0]);
    return 1;
  }

  std::string binfer_path = argv[1];
  std::string spv_path = "tools/kernels_258v/all_kernels.spv";
  std::string report_path = "tools/bench_258v/report_bench_t71.json";
  uint32_t max_ctx = 4096;
  int decode_tokens = 64;
  int warmup_runs = 1;
  int measured_runs = 3;
  std::vector<int> prompt_ids;
  int synth_prompt_len = 0;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--spv=", 0) == 0) {
      spv_path = arg.substr(6);
    } else if (arg.rfind("--max-ctx=", 0) == 0) {
      max_ctx = std::stoul(arg.substr(10));
    } else if (arg.rfind("--decode-tokens=", 0) == 0) {
      decode_tokens = std::stoi(arg.substr(16));
    } else if (arg.rfind("--warmup-runs=", 0) == 0) {
      warmup_runs = std::stoi(arg.substr(14));
    } else if (arg.rfind("--measured-runs=", 0) == 0) {
      measured_runs = std::stoi(arg.substr(16));
    } else if (arg.rfind("--report=", 0) == 0) {
      report_path = arg.substr(9);
    } else if (arg.rfind("--prompt-len=", 0) == 0) {
      synth_prompt_len = std::stoi(arg.substr(13));
    } else if (arg.rfind("--ids=", 0) == 0) {
      std::stringstream ss(arg.substr(6));
      std::string item;
      while (std::getline(ss, item, ',')) {
        if (!item.empty()) prompt_ids.push_back(std::stoi(item));
      }
    } else if (arg.rfind("--ids-file=", 0) == 0) {
      std::ifstream f(arg.substr(11));
      if (f) {
        std::string line;
        while (std::getline(f, line)) {
          std::stringstream ss(line);
          std::string item;
          while (std::getline(ss, item, ',')) {
            if (!item.empty()) prompt_ids.push_back(std::stoi(item));
          }
        }
      }
    }
  }

  if (synth_prompt_len > 0) {
    prompt_ids.clear();
    // Repeating pattern of standard tokens
    int pattern[] = {151644, 8948, 198, 2610, 525, 264, 10925, 151645, 198, 248045, 74455, 198, 248068, 271, 248069, 271};
    for (int k = 0; k < synth_prompt_len; ++k) {
      prompt_ids.push_back(pattern[k % 16]);
    }
  }

  if (prompt_ids.empty()) {
    // Default 16-token prompt
    prompt_ids = {151644, 8948, 198, 2610, 525, 264, 10925, 151645, 198, 248045, 74455, 198, 248068, 271, 248069, 271};
  }

  std::printf("=================================================================\n");
  std::printf("--- AInfer Isolated Benchmark Harness for Core Ultra 7 258V ---\n");
  std::printf("=================================================================\n");
  std::printf("  Model:         %s\n", binfer_path.c_str());
  std::printf("  SPIR-V:        %s\n", spv_path.c_str());
  std::printf("  Max Context:   %u tokens\n", max_ctx);
  std::printf("  Prompt Tokens: %zu tokens\n", prompt_ids.size());
  std::printf("  Decode Tokens: %d tokens\n", decode_tokens);
  std::printf("  Warmup Runs:   %d, Measured Runs: %d\n", warmup_runs, measured_runs);
  std::printf("-----------------------------------------------------------------\n\n");

  long rss_before_init = get_current_rss_kb();

  // 1. Measure Model Load Time
  AInferRuntime258V runtime;
  std::printf("[Stage 1] Initializing Level Zero runtime and uploading weights...\n");
  auto t_load0 = std::chrono::steady_clock::now();
  if (!runtime.init(binfer_path, spv_path, max_ctx)) {
    std::fprintf(stderr, "FATAL: Failed to initialize runtime\n");
    return 1;
  }
  auto t_load1 = std::chrono::steady_clock::now();
  double model_load_time_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();
  long rss_after_init = get_current_rss_kb();

  std::printf("  Model Load Time: %.2f ms (%.3f s)\n", model_load_time_ms, model_load_time_ms / 1000.0);
  std::printf("  Host RSS:        %ld KB -> %ld KB (delta: %ld KB)\n",
              rss_before_init, rss_after_init, rss_after_init - rss_before_init);

  uint64_t pay_bytes = runtime.get_payload_arena_bytes();
  uint64_t sc_bytes = runtime.get_scale_arena_bytes();
  uint64_t kv_bytes = runtime.get_kv_cache_bytes();
  uint64_t ssm_bytes = runtime.get_ssm_state_bytes();
  uint64_t work_bytes = runtime.get_workspace_bytes();
  uint64_t total_arena_bytes = pay_bytes + sc_bytes + kv_bytes + ssm_bytes + work_bytes;

  std::printf("  Payload Arena:   %.2f GiB\n", pay_bytes / (1024.0 * 1024.0 * 1024.0));
  std::printf("  Scale Arena:     %.2f MiB\n", sc_bytes / (1024.0 * 1024.0));
  std::printf("  KV Cache Arena:  %.2f MiB\n", kv_bytes / (1024.0 * 1024.0));
  std::printf("  SSM State Arena: %.2f MiB\n", ssm_bytes / (1024.0 * 1024.0));
  std::printf("  Workspace Arena: %.2f MiB\n", work_bytes / (1024.0 * 1024.0));
  std::printf("  Total Committed: %.2f GiB\n\n", total_arena_bytes / (1024.0 * 1024.0 * 1024.0));

  // 2. Measure Cold TTFT and Cold First Decode Latency
  std::printf("[Stage 2] Cold Start Benchmark (first execution right after init)...\n");
  int cold_first_token = 0;
  auto t_cold_pref0 = std::chrono::steady_clock::now();
  if (!runtime.prefill(prompt_ids, &cold_first_token)) {
    std::fprintf(stderr, "FATAL: Cold prefill failed\n");
    return 1;
  }
  auto t_cold_pref1 = std::chrono::steady_clock::now();
  double cold_ttft_ms = std::chrono::duration<double, std::milli>(t_cold_pref1 - t_cold_pref0).count();

  int cold_second_token = 0;
  auto t_cold_dec0 = std::chrono::steady_clock::now();
  if (!runtime.decode_step(&cold_second_token)) {
    std::fprintf(stderr, "FATAL: Cold first decode step failed\n");
    return 1;
  }
  auto t_cold_dec1 = std::chrono::steady_clock::now();
  double cold_first_decode_ms = std::chrono::duration<double, std::milli>(t_cold_dec1 - t_cold_dec0).count();

  std::printf("  Cold TTFT:              %.2f ms (%.2f ms/tok for %zu tokens)\n",
              cold_ttft_ms, cold_ttft_ms / prompt_ids.size(), prompt_ids.size());
  std::printf("  Cold 1st Decode Latency: %.2f ms\n\n", cold_first_decode_ms);

  // 3. Warmup runs
  std::printf("[Stage 3] Executing %d warmup runs...\n", warmup_runs);
  for (int w = 0; w < warmup_runs; ++w) {
    runtime.reset_state();
    int dummy_tok = 0;
    runtime.prefill(prompt_ids, &dummy_tok);
    for (int step = 1; step < std::min(8, decode_tokens); ++step) {
      runtime.decode_step(&dummy_tok);
    }
  }

  // 4. Measured Warm Runs for Isolated Steady-State Metrics
  std::printf("[Stage 4] Executing %d measured steady-state runs...\n", measured_runs);
  std::vector<double> run_warm_ttft_ms;
  std::vector<double> run_prefill_ms;
  std::vector<double> run_prefill_tok_per_s;
  std::vector<double> run_first_decode_ms;
  std::vector<double> run_sustained_decode_tok_per_s;
  std::vector<double> all_inter_token_latencies_ms;

  for (int run = 0; run < measured_runs; ++run) {
    runtime.reset_state();

    // Measure Warm TTFT / Prefill Time
    int first_token = 0;
    auto t_pref0 = std::chrono::steady_clock::now();
    if (!runtime.prefill(prompt_ids, &first_token)) {
      std::fprintf(stderr, "FATAL: Prefill failed on run %d\n", run);
      return 1;
    }
    auto t_pref1 = std::chrono::steady_clock::now();
    double pref_ms = std::chrono::duration<double, std::milli>(t_pref1 - t_pref0).count();
    double pref_tps = (double)prompt_ids.size() / (pref_ms * 1e-3);

    run_warm_ttft_ms.push_back(pref_ms);
    run_prefill_ms.push_back(pref_ms);
    run_prefill_tok_per_s.push_back(pref_tps);

    // Measure First Decode Step Latency
    int next_token = 0;
    auto t_d0 = std::chrono::steady_clock::now();
    if (!runtime.decode_step(&next_token)) {
      std::fprintf(stderr, "FATAL: First decode failed on run %d\n", run);
      return 1;
    }
    auto t_d1 = std::chrono::steady_clock::now();
    double first_dec_ms = std::chrono::duration<double, std::milli>(t_d1 - t_d0).count();
    run_first_decode_ms.push_back(first_dec_ms);
    all_inter_token_latencies_ms.push_back(first_dec_ms);

    // Measure remaining decode steps individually to record inter-token jitter
    std::vector<double> run_step_latencies_ms;
    run_step_latencies_ms.push_back(first_dec_ms);

    auto t_sustained_start = std::chrono::steady_clock::now();
    for (int step = 2; step < decode_tokens; ++step) {
      auto t_step0 = std::chrono::steady_clock::now();
      if (!runtime.decode_step(&next_token)) break;
      auto t_step1 = std::chrono::steady_clock::now();
      double step_ms = std::chrono::duration<double, std::milli>(t_step1 - t_step0).count();
      run_step_latencies_ms.push_back(step_ms);
      all_inter_token_latencies_ms.push_back(step_ms);
    }
    auto t_sustained_end = std::chrono::steady_clock::now();
    double sustained_s = std::chrono::duration<double>(t_sustained_end - t_sustained_start).count();
    double sustained_tps = (run_step_latencies_ms.size() > 1 && sustained_s > 0)
                               ? (double)(run_step_latencies_ms.size() - 1) / sustained_s
                               : 0.0;
    run_sustained_decode_tok_per_s.push_back(sustained_tps);

    std::printf("  Run %d: warm_ttft=%.2f ms, prefill=%.2f tok/s, 1st_dec=%.2f ms, sustained_dec=%.2f tok/s\n",
                run + 1, pref_ms, pref_tps, first_dec_ms, sustained_tps);
  }

  // Aggregate stats across measured runs
  auto avg_vec = [](const std::vector<double> &v) -> double {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  };

  double mean_warm_ttft_ms = avg_vec(run_warm_ttft_ms);
  double mean_prefill_ms = avg_vec(run_prefill_ms);
  double mean_prefill_tok_per_s = avg_vec(run_prefill_tok_per_s);
  double mean_first_decode_ms = avg_vec(run_first_decode_ms);
  double mean_sustained_decode_tok_per_s = avg_vec(run_sustained_decode_tok_per_s);

  LatencyStats jitter_stats = compute_stats(all_inter_token_latencies_ms);
  long rss_final = get_current_rss_kb();

  std::printf("\n=================================================================\n");
  std::printf("--- Standardized Performance Summary (Task T7.1) ---\n");
  std::printf("=================================================================\n");
  std::printf("  Model Load Time:          %.2f ms (%.3f s)\n", model_load_time_ms, model_load_time_ms / 1000.0);
  std::printf("  Cold TTFT:                %.2f ms\n", cold_ttft_ms);
  std::printf("  Warm TTFT (mean):         %.2f ms\n", mean_warm_ttft_ms);
  std::printf("  Prefill Latency:          %.2f ms (for %zu tokens)\n", mean_prefill_ms, prompt_ids.size());
  std::printf("  Prefill Throughput:       %.2f tok/s\n", mean_prefill_tok_per_s);
  std::printf("  First Decode Latency:     %.2f ms\n", mean_first_decode_ms);
  std::printf("  Sustained Decode:         %.2f tok/s\n", mean_sustained_decode_tok_per_s);
  std::printf("  Inter-Token Latency (p50):%.2f ms\n", jitter_stats.p50_ms);
  std::printf("  Inter-Token Latency (p90):%.2f ms\n", jitter_stats.p90_ms);
  std::printf("  Inter-Token Latency (p95):%.2f ms\n", jitter_stats.p95_ms);
  std::printf("  Inter-Token Latency (p99):%.2f ms\n", jitter_stats.p99_ms);
  std::printf("  Inter-Token StdDev:       %.2f ms\n", jitter_stats.stddev_ms);
  std::printf("  Host Peak RSS:            %ld KB\n", rss_final);
  std::printf("=================================================================\n\n");

  // Emit standardized JSON report
  std::ofstream out(report_path);
  if (!out) {
    std::fprintf(stderr, "Failed to write report to %s\n", report_path.c_str());
    return 1;
  }

  out << "{\n";
  out << "  \"task\": \"T7.1\",\n";
  out << "  \"device\": \"" << runtime.get_device_name() << "\",\n";
  out << "  \"model\": \"" << binfer_path << "\",\n";
  out << "  \"max_context\": " << max_ctx << ",\n";
  out << "  \"prompt_tokens\": " << prompt_ids.size() << ",\n";
  out << "  \"decode_tokens_per_run\": " << decode_tokens << ",\n";
  out << "  \"measured_runs\": " << measured_runs << ",\n";
  out << "  \"timing_fields\": {\n";
  out << "    \"model_load_time_ms\": " << model_load_time_ms << ",\n";
  out << "    \"cold_ttft_ms\": " << cold_ttft_ms << ",\n";
  out << "    \"warm_ttft_ms\": " << mean_warm_ttft_ms << ",\n";
  out << "    \"prefill_time_ms\": " << mean_prefill_ms << ",\n";
  out << "    \"prefill_tok_per_s\": " << mean_prefill_tok_per_s << ",\n";
  out << "    \"first_decode_latency_ms\": " << mean_first_decode_ms << ",\n";
  out << "    \"sustained_decode_tok_per_s\": " << mean_sustained_decode_tok_per_s << ",\n";
  out << "    \"inter_token_jitter\": {\n";
  out << "      \"min_ms\": " << jitter_stats.min_ms << ",\n";
  out << "      \"max_ms\": " << jitter_stats.max_ms << ",\n";
  out << "      \"mean_ms\": " << jitter_stats.mean_ms << ",\n";
  out << "      \"stddev_ms\": " << jitter_stats.stddev_ms << ",\n";
  out << "      \"p50_ms\": " << jitter_stats.p50_ms << ",\n";
  out << "      \"p90_ms\": " << jitter_stats.p90_ms << ",\n";
  out << "      \"p95_ms\": " << jitter_stats.p95_ms << ",\n";
  out << "      \"p99_ms\": " << jitter_stats.p99_ms << "\n";
  out << "    }\n";
  out << "  },\n";
  out << "  \"memory_footprint\": {\n";
  out << "    \"payload_arena_bytes\": " << pay_bytes << ",\n";
  out << "    \"scale_arena_bytes\": " << sc_bytes << ",\n";
  out << "    \"kv_cache_bytes\": " << kv_bytes << ",\n";
  out << "    \"ssm_state_bytes\": " << ssm_bytes << ",\n";
  out << "    \"workspace_bytes\": " << work_bytes << ",\n";
  out << "    \"total_committed_bytes\": " << total_arena_bytes << ",\n";
  out << "    \"total_committed_gib\": " << (double)total_arena_bytes / (1024.0 * 1024.0 * 1024.0) << ",\n";
  out << "    \"rss_before_init_kb\": " << rss_before_init << ",\n";
  out << "    \"rss_after_init_kb\": " << rss_after_init << ",\n";
  out << "    \"rss_final_kb\": " << rss_final << "\n";
  out << "  },\n";
  out << "  \"raw_runs\": [\n";
  for (size_t r = 0; r < run_warm_ttft_ms.size(); ++r) {
    out << "    {\n";
    out << "      \"run_index\": " << r + 1 << ",\n";
    out << "      \"warm_ttft_ms\": " << run_warm_ttft_ms[r] << ",\n";
    out << "      \"prefill_time_ms\": " << run_prefill_ms[r] << ",\n";
    out << "      \"prefill_tok_per_s\": " << run_prefill_tok_per_s[r] << ",\n";
    out << "      \"first_decode_ms\": " << run_first_decode_ms[r] << ",\n";
    out << "      \"sustained_decode_tok_per_s\": " << run_sustained_decode_tok_per_s[r] << "\n";
    out << "    }" << (r + 1 < run_warm_ttft_ms.size() ? "," : "") << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  out.close();

  std::printf("Report saved to: %s\n", report_path.c_str());
  return 0;
}
