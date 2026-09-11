// T6.3 activation-quantization diagnostic (CPU, real L0 MLP weights).
// Q: does per-tensor INT8 (current device policy) destroy the unnormalized
//    silu-mul vector dG17 (17408-wide, down_proj input), and does per-group-128
//    INT8 fix it? Also checks dH-class (post-norm GEMV input) for contrast.
// Compares: zero-rate, clip-rate, input SNR, downstream down_proj output error.
// Magnitudes are swept x1/x2/x4/x8 to emulate long-context growth (dXmax 300-419).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ---- minimal safetensors reader (one tensor, BF16) ----
static uint64_t rd64le(const char *p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
static uint32_t rd32le(const char *p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
static float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float x; std::memcpy(&x, &u, 4); return x; }

// find tensor data offset in shard file; returns file offset of raw bytes
static bool find_tensor(const std::string &shard, const std::string &want,
                        uint64_t &data_off, std::vector<int64_t> &shape, std::string &dtype) {
  std::ifstream f(shard, std::ios::binary);
  if (!f) return false;
  char tmp[8]; f.read(tmp, 8);
  uint64_t hlen = rd64le(tmp);
  std::string hdr(hlen, 0);
  f.read(hdr.data(), hlen);
  // naive JSON scan for "want":{"dtype":..,"shape":..,"data_offsets":[a,b]}
  std::string key = "\"" + want + "\"";
  size_t k = hdr.find(key);
  if (k == std::string::npos) return false;
  size_t dd = hdr.find("\"dtype\"", k); size_t dc = hdr.find(':', dd);
  size_t dq = hdr.find('"', dc); dtype = hdr.substr(dq + 1, hdr.find('"', dq + 1) - dq - 1);
  size_t ds = hdr.find("\"shape\"", k); size_t db = hdr.find('[', ds); size_t de = hdr.find(']', db);
  shape.clear();
  { std::string s = hdr.substr(db + 1, de - db - 1); size_t p = 0;
    while (p < s.size()) { size_t c = s.find(',', p); std::string n = s.substr(p, c == std::string::npos ? c : c - p);
      if (!n.empty() && n != " ") shape.push_back(std::stoll(n)); if (c == std::string::npos) break; p = c + 1; } }
  size_t dof = hdr.find("\"data_offsets\"", k); size_t ob = hdr.find('[', dof); size_t cm = hdr.find(',', ob);
  uint64_t a = std::stoull(hdr.substr(ob + 1, cm - ob - 1));
  data_off = 8 + hlen + a;
  return true;
}

static std::vector<float> load_bf16(const std::string &shard, const std::string &name) {
  uint64_t off; std::vector<int64_t> shape; std::string dtype;
  if (!find_tensor(shard, name, off, shape, dtype)) { std::fprintf(stderr, "missing %s\n", name.c_str()); std::exit(1); }
  if (dtype != "BF16") { std::fprintf(stderr, "dtype %s\n", dtype.c_str()); std::exit(1); }
  size_t n = 1; for (auto d : shape) n *= (size_t)d;
  std::vector<uint16_t> raw(n);
  std::ifstream f(shard, std::ios::binary);
  f.seekg((std::streamoff)off); f.read((char *)raw.data(), n * 2);
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(raw[i]);
  return out;
}

// minimal JSON array-of-arrays reader for mlp_layer0.json "in" (2x5120 floats)
static std::vector<float> load_mlp_in(const std::string &path, int &rows, int &cols) {
  std::ifstream f(path, std::ios::binary);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  size_t k = s.find("\"in\"");
  size_t b = s.find('[', k);
  rows = 0; cols = -1;
  std::vector<float> out;
  size_t p = b; int depth = 0; std::string num;
  auto flush = [&]() { if (!num.empty()) { out.push_back(std::stof(num)); num.clear(); } };
  for (; p < s.size(); ++p) {
    char c = s[p];
    if (c == '[') { depth++; if (depth == 2) rows++; }
    else if (c == ']') { flush(); depth--; if (depth == 0) break; }
    else if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') num += c;
    else if (c == ',' || c == ' ' || c == '\n' || c == '\t' || c == '\r') flush();
    else if (depth == 0 && out.size() > 0) break;
  }
  cols = (int)(out.size() / (size_t)rows);
  return out;
}

struct QStats { double zero_rate, clip_rate, snr_db, out_meanrel, out_maxrel; };

static QStats eval_policy(const std::vector<float> &vec, const std::vector<float> &Wdown,
                          int M, int K, bool per_group) {
  int G = K / 128;
  std::vector<float> sc(per_group ? G : 1);
  if (per_group) {
    for (int g = 0; g < G; ++g) {
      float mx = 0; for (int j = 0; j < 128; ++j) mx = std::max(mx, std::fabs(vec[g * 128 + j]));
      sc[g] = mx / 127.0f; if (sc[g] == 0) sc[g] = 1.0f;
    }
  } else {
    float mx = 0; for (float v : vec) mx = std::max(mx, std::fabs(v));
    sc[0] = mx / 127.0f; if (sc[0] == 0) sc[0] = 1.0f;
  }
  std::vector<float> dq(K);
  long zeros = 0, clips = 0;
  double sig = 0, noise = 0;
  for (int j = 0; j < K; ++j) {
    float s = per_group ? sc[j / 128] : sc[0];
    float v = vec[j] / s;
    int qi = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
    int qc = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
    if (qc != qi) clips++;
    if (qc == 0) zeros++;
    float r = (float)qc * s;
    dq[j] = r;
    sig += (double)vec[j] * vec[j]; noise += (double)(vec[j] - r) * (vec[j] - r);
  }
  // downstream: y = Wdown @ vec  vs  Wdown @ dq  (M rows)
  double num = 0, den = 0, mxr = 0, refmax = 0;
  std::vector<double> yref(M, 0.0);
  for (int m = 0; m < M; ++m) {
    double a = 0; for (int j = 0; j < K; ++j) a += (double)Wdown[(size_t)m * K + j] * vec[j];
    yref[m] = a; refmax = std::max(refmax, std::fabs(a));
  }
  for (int m = 0; m < M; ++m) {
    double a = 0; for (int j = 0; j < K; ++j) a += (double)Wdown[(size_t)m * K + j] * dq[j];
    double d = std::fabs(a - yref[m]);
    num += d; den += std::fabs(yref[m]); mxr = std::max(mxr, d / (refmax > 0 ? refmax : 1));
  }
  QStats st;
  st.zero_rate = (double)zeros / K; st.clip_rate = (double)clips / K;
  st.snr_db = 10 * std::log10(sig / (noise > 0 ? noise : 1e-30));
  st.out_meanrel = num / (den > 0 ? den : 1); st.out_maxrel = mxr;
  return st;
}

int main(int argc, char **argv) {
  const std::string shard = "models/Qwen3.8-27B/model-00001-of-00018.safetensors";
  std::printf("loading L0 MLP weights from %s ...\n", shard.c_str());
  auto Wgate = load_bf16(shard, "model.language_model.layers.0.mlp.gate_proj.weight"); // 17408x5120
  auto Wup = load_bf16(shard, "model.language_model.layers.0.mlp.up_proj.weight");
  auto Wdown = load_bf16(shard, "model.language_model.layers.0.mlp.down_proj.weight"); // 5120x17408
  auto Wnorm = load_bf16(shard, "model.language_model.layers.0.input_layernorm.weight"); // 5120
  int rows, cols;
  auto Xin = load_mlp_in("reference/mlp_layer0.json", rows, cols);
  std::printf("gate %zu up %zu down %zu norm %zu | in %d x %d\n",
              Wgate.size(), Wup.size(), Wdown.size(), Wnorm.size(), rows, cols);
  const int H = 5120, I = 17408;
  std::vector<float> gate(I), up(I), dg(I), h(H);
  std::string json = "{\"class\":\"dG17-unorm\",\"scales\":[1,2,4,8],\"rows\":[";
  bool first = true;
  for (int scl : {1, 2, 4, 8}) {
    // token 0, scaled to emulate long-context magnitude growth
    for (int j = 0; j < H; ++j) h[j] = Xin[j] * (float)scl;
    // rmsnorm (1+w)
    double ss = 0; for (int j = 0; j < H; ++j) ss += (double)h[j] * h[j];
    float inv = 1.0f / std::sqrt((float)(ss / H) + 1e-6f);
    std::vector<float> hn(H);
    for (int j = 0; j < H; ++j) hn[j] = h[j] * inv * (1.0f + Wnorm[j]);
    // gate/up matvecs (fp32, BF16 weights = clean reference path)
    for (int m = 0; m < I; ++m) {
      double a = 0, b = 0;
      for (int j = 0; j < H; ++j) { a += (double)Wgate[(size_t)m * H + j] * hn[j]; b += (double)Wup[(size_t)m * H + j] * hn[j]; }
      gate[m] = (float)a; up[m] = (float)b;
    }
    for (int m = 0; m < I; ++m) dg[m] = (gate[m] / (1.0f + std::exp(-gate[m]))) * up[m];
    float dgmax = 0; for (float v : dg) dgmax = std::max(dgmax, std::fabs(v));
    float hnmax = 0; for (float v : hn) hnmax = std::max(hnmax, std::fabs(v));
    QStats pt = eval_policy(dg, Wdown, H, I, false);
    QStats pg = eval_policy(dg, Wdown, H, I, true);
    // dH-class contrast: quantize hn itself through gate_proj (use Wgate as M=I,K=H)
    QStats ht = eval_policy(hn, Wgate, I, H, false);
    QStats hg = eval_policy(hn, Wgate, I, H, true);
    std::printf("[x%d] dgmax %.2f hnmax %.2f | dG17 per-tensor: zero %.3f clip %.4f snr %5.1fdB out-meanrel %.4f out-maxrel %.4f\n",
                scl, dgmax, hnmax, pt.zero_rate, pt.clip_rate, pt.snr_db, pt.out_meanrel, pt.out_maxrel);
    std::printf("[x%d] dgmax %.2f hnmax %.2f | dG17 per-group : zero %.3f clip %.4f snr %5.1fdB out-meanrel %.4f out-maxrel %.4f\n",
                scl, dgmax, hnmax, pg.zero_rate, pg.clip_rate, pg.snr_db, pg.out_meanrel, pg.out_maxrel);
    std::printf("[x%d]              | dH   per-tensor: zero %.3f clip %.4f snr %5.1fdB out-meanrel %.4f | per-group: zero %.3f snr %5.1fdB out-meanrel %.4f\n",
                scl, ht.zero_rate, ht.clip_rate, ht.snr_db, ht.out_meanrel, hg.zero_rate, hg.snr_db, hg.out_meanrel);
    char row[1024];
    std::snprintf(row, sizeof row,
                  "%s{\"scale\":%d,\"dgmax\":%.3f,\"pt_zero\":%.4f,\"pt_clip\":%.5f,\"pt_snr\":%.2f,\"pt_meanrel\":%.5f,\"pt_maxrel\":%.5f,"
                  "\"pg_zero\":%.4f,\"pg_clip\":%.5f,\"pg_snr\":%.2f,\"pg_meanrel\":%.5f,\"pg_maxrel\":%.5f,"
                  "\"ht_meanrel\":%.5f,\"hg_meanrel\":%.5f}",
                  first ? "" : ",", scl, dgmax,
                  pt.zero_rate, pt.clip_rate, pt.snr_db, pt.out_meanrel, pt.out_maxrel,
                  pg.zero_rate, pg.clip_rate, pg.snr_db, pg.out_meanrel, pg.out_maxrel,
                  ht.out_meanrel, hg.out_meanrel);
    json += row; first = false;
  }
  json += "]}";
  std::string out = (argc > 1) ? argv[1] : "tools/t63/report_actquant.json";
  FILE *o = std::fopen(out.c_str(), "w");
  std::fprintf(o, "%s\n", json.c_str()); std::fclose(o);
  std::printf("wrote %s\n", out.c_str());
  return 0;
}
