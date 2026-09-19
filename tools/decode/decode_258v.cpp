// AInfer Single-Process In-Memory Recorded Loop Runtime CLI for Intel Arc 140V (258V)
// Usage: decode_258v <model.binfer> [options]
// Options:
//   --ids=<csv>              Prompt token IDs as comma-separated integers
//   --ids-file=<file>        Prompt token IDs from file
//   --max-new=<N>            Maximum number of tokens to decode (default: 32)
//   --max-ctx=<N>            Maximum context capacity in tokens (default: 2048)
//   --spv=<file>             Path to SPIR-V module (default: tools/kernels_258v/all_kernels.spv)
//   --report=<file>          Write execution metrics to JSON report
//   --benchmark=<N>          Benchmark steady-state decode across N tokens

#include "runtime_258v.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace ainfer;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <model.binfer> [--ids=..] [--max-new=N] [--report=out.json]\n", argv[0]);
    return 1;
  }

  std::string model_path = argv[1];
  std::string spv_path = "tools/kernels_258v/all_kernels.spv";
  std::string report_path = "";
  std::vector<int> prompt_ids;
  int max_new = 32;
  int max_ctx = 2048;
  int bench_tokens = 0;

  std::string batch_file = "";
  std::string tf_file = "";
  std::string out_file = "";

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--ids=", 0) == 0) {
      std::stringstream ss(arg.substr(6));
      std::string item;
      while (std::getline(ss, item, ',')) {
        if (!item.empty()) prompt_ids.push_back(std::stoi(item));
      }
    } else if (arg.rfind("--ids-file=", 0) == 0) {
      std::ifstream ff(arg.substr(11));
      if (ff) {
        std::string line;
        while (std::getline(ff, line)) {
          std::stringstream ss(line);
          std::string item;
          while (std::getline(ss, item, ',')) {
            if (!item.empty()) prompt_ids.push_back(std::stoi(item));
          }
        }
      }
    } else if (arg.rfind("--batch-file=", 0) == 0) {
      batch_file = arg.substr(13);
    } else if (arg.rfind("--teacher-forced-file=", 0) == 0) {
      tf_file = arg.substr(22);
    } else if (arg.rfind("--out-file=", 0) == 0) {
      out_file = arg.substr(11);
    } else if (arg.rfind("--max-new=", 0) == 0) {
      max_new = std::stoi(arg.substr(10));
    } else if (arg.rfind("--max-ctx=", 0) == 0) {
      max_ctx = std::stoi(arg.substr(10));
    } else if (arg.rfind("--spv=", 0) == 0) {
      spv_path = arg.substr(6);
    } else if (arg.rfind("--report=", 0) == 0) {
      report_path = arg.substr(9);
    } else if (arg.rfind("--benchmark=", 0) == 0) {
      bench_tokens = std::stoi(arg.substr(12));
    }
  }

  AInferRuntime258V runtime;
  if (!runtime.init(model_path, spv_path, max_ctx)) {
    std::fprintf(stderr, "Failed to initialize runtime\n");
    return 1;
  }

  // Batch evaluation mode
  if (!batch_file.empty()) {
    std::ifstream bf(batch_file);
    if (!bf) {
      std::fprintf(stderr, "Cannot open batch file: %s\n", batch_file.c_str());
      return 1;
    }

    std::ofstream out;
    if (!out_file.empty()) {
      out.open(out_file);
      out << "{\n";
      out << "  \"device\": \"" << runtime.get_device_name() << "\",\n";
      out << "  \"results\": [\n";
    }

    std::string line;
    int case_idx = 0;
    while (std::getline(bf, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::stringstream ss(line);
      std::string case_id;
      std::string max_new_str;
      std::getline(ss, case_id, ',');
      std::getline(ss, max_new_str, ',');
      int case_max_new = std::stoi(max_new_str);

      std::vector<int> c_prompt;
      std::string tok_str;
      while (std::getline(ss, tok_str, ',')) {
        if (!tok_str.empty()) c_prompt.push_back(std::stoi(tok_str));
      }

      runtime.reset_state();
      double pref_ms = 0.0, dec_tok_s = 0.0;
      std::vector<int> gen_ids;
      runtime.generate(c_prompt, case_max_new, gen_ids, &pref_ms, &dec_tok_s);

      std::printf("[%d] %s: pref=%.2f ms, gen=%zu tok (%.2f tok/s)\n",
                  case_idx + 1, case_id.c_str(), pref_ms, gen_ids.size(), dec_tok_s);

      if (out.is_open()) {
        if (case_idx > 0) out << ",\n";
        out << "    {\n";
        out << "      \"id\": \"" << case_id << "\",\n";
        out << "      \"prompt_tokens\": " << c_prompt.size() << ",\n";
        out << "      \"generated_tokens\": " << gen_ids.size() << ",\n";
        out << "      \"prefill_ms\": " << pref_ms << ",\n";
        out << "      \"decode_tok_per_s\": " << dec_tok_s << ",\n";
        out << "      \"generated_ids\": [";
        for (size_t k = 0; k < gen_ids.size(); ++k) {
          out << gen_ids[k] << (k + 1 < gen_ids.size() ? ", " : "");
        }
        out << "]\n";
        out << "    }";
        out.flush();
      }
      case_idx++;
    }

    if (out.is_open()) {
      out << "\n  ]\n}\n";
      out.close();
    }
    std::printf("[Batch] Finished evaluating %d cases.\n", case_idx);
    return 0;
  }

  // Teacher-forced evaluation mode (T6.4)
  if (!tf_file.empty()) {
    std::ifstream tff(tf_file);
    if (!tff) {
      std::fprintf(stderr, "Cannot open teacher-forced file: %s\n", tf_file.c_str());
      return 1;
    }

    std::ofstream out;
    if (!out_file.empty()) {
      out.open(out_file);
      out << "{\n";
      out << "  \"device\": \"" << runtime.get_device_name() << "\",\n";
      out << "  \"results\": [\n";
    }

    std::string line;
    int case_idx = 0;
    while (std::getline(tff, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::stringstream ss(line);
      std::string case_id;
      std::getline(ss, case_id, ',');

      std::vector<int> c_tokens;
      std::string tok_str;
      while (std::getline(ss, tok_str, ',')) {
        if (!tok_str.empty()) c_tokens.push_back(std::stoi(tok_str));
      }

      runtime.reset_state();
      std::vector<int> preds;
      auto t0 = std::chrono::steady_clock::now();
      runtime.teacher_forced_eval(c_tokens, preds);
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

      std::printf("[TF %d] %s: %zu tokens evaluated in %.2f ms (%.2f tok/s)\n",
                  case_idx + 1, case_id.c_str(), c_tokens.size(), ms, c_tokens.size() / (ms * 1e-3));

      if (out.is_open()) {
        if (case_idx > 0) out << ",\n";
        out << "    {\n";
        out << "      \"id\": \"" << case_id << "\",\n";
        out << "      \"tokens\": " << c_tokens.size() << ",\n";
        out << "      \"ms\": " << ms << ",\n";
        out << "      \"predicted_ids\": [";
        for (size_t k = 0; k < preds.size(); ++k) {
          out << preds[k] << (k + 1 < preds.size() ? ", " : "");
        }
        out << "]\n";
        out << "    }";
        out.flush();
      }
      case_idx++;
    }

    if (out.is_open()) {
      out << "\n  ]\n}\n";
      out.close();
    }
    std::printf("[TeacherForced] Finished evaluating %d cases.\n", case_idx);
    return 0;
  }

  if (prompt_ids.empty()) {
    // Default prompt: <|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    prompt_ids = {248045, 846, 198, 9419, 0, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271};
  }

  std::printf("\n[Execution] Running in-memory prefill for %zu tokens...\n", prompt_ids.size());
  double prefill_ms = 0.0;
  double decode_tok_per_s = 0.0;
  std::vector<int> generated_ids;

  int tokens_to_gen = (bench_tokens > 0) ? bench_tokens : max_new;
  if (!runtime.generate(prompt_ids, tokens_to_gen, generated_ids, &prefill_ms, &decode_tok_per_s)) {
    std::fprintf(stderr, "Generation failed\n");
    return 1;
  }

  std::printf("\n=======================================================\n");
  std::printf("--- Generation Results ---\n");
  std::printf("=======================================================\n");
  std::printf("  Prefill latency:   %.2f ms (%.2f ms/tok)\n", prefill_ms, prefill_ms / prompt_ids.size());
  std::printf("  Tokens generated:  %zu\n", generated_ids.size());
  std::printf("  Decode throughput: %.2f tok/s\n", decode_tok_per_s);
  std::printf("  Output tokens:     [");
  for (size_t i = 0; i < generated_ids.size(); ++i) {
    std::printf("%d%s", generated_ids[i], i + 1 < generated_ids.size() ? ", " : "");
  }
  std::printf("]\n");
  std::printf("=======================================================\n");

  if (!report_path.empty()) {
    std::ofstream rpt(report_path);
    rpt << "{\n";
    rpt << "  \"device\": \"" << runtime.get_device_name() << "\",\n";
    rpt << "  \"prompt_tokens\": " << prompt_ids.size() << ",\n";
    rpt << "  \"generated_tokens\": " << generated_ids.size() << ",\n";
    rpt << "  \"prefill_ms\": " << prefill_ms << ",\n";
    rpt << "  \"decode_tok_per_s\": " << decode_tok_per_s << ",\n";
    rpt << "  \"generated_ids\": [";
    for (size_t i = 0; i < generated_ids.size(); ++i) {
      rpt << generated_ids[i] << (i + 1 < generated_ids.size() ? ", " : "");
    }
    rpt << "]\n";
    rpt << "}\n";
    rpt.close();
  }

  return 0;
}
