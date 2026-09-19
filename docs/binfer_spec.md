# `.binfer` Container Format — v1.1 MoE Extension (T3.1)

Versioned, hardware-oriented model container for AInfer. Single file, little-endian,
all sections 64-byte aligned. Scope is **text-only** INT4 weight-only models
(dense hybrid Qwen3.8-27B in v1.0, sparse MoE hybrid Qwen3.5-MoE / Tiel-Coder-35B-A3B in v1.1);
vision weights are excluded and rejected by loaders.

---

## 1. Design principles

- **Backward-Compatible Extension:** Existing 192-byte tensor directory entries remain untouched. MoE topology is isolated in a dedicated section (Section 6) rather than bloating non-expert entries.
- **Independent Addressability:** Every tensor and expert matrix is individually addressable with static offsets.
- **Layout Registry:** Every tensor carries a `layout_id` (0 = row-major dense; 3D tensors `[num_experts, out_dim, in_dim]` represent batched expert banks).
- **Deterministic Conversion:** Identical source weights + quantization policy produce byte-identical containers with matching CRCs.
- **Validation Before Allocation:** Header, section table, CRC32, tensor offsets, routing dimensions, and payload bounds are validated before Level Zero memory allocation.
- **64-Byte Alignment:** All sections, scales, and tensor payloads start at 64-byte boundaries (Xe2 cache line friendly; zeroed padding).

---

## 2. File layout

| # | Section | Alignment | Contents |
|---|---|---|---|
| 0 | File header | 0 | 128 bytes, §3 |
| 1 | Model identity | 64 | source repo/revision, conversion timestamp, §4 |
| 2 | Architecture metadata | 64 | Layer pattern, hidden dims, attention/DeltaNet parameters, §5 |
| 3 | Tokenizer descriptor | 64 | Vocabulary size, token hashes, added tokens, §6 |
| 4 | Quantization policy | 64 | INT4 group-128 policy, exception rules, §7 |
| 5 | Tensor directory | 64 | $N \times 192$-byte entries (up to 8D shapes), §8 |
| 6 | MoE metadata *(v1.1+)* | 64 | Expert counts, routing dimensions, active $k$, shared experts, §9 |
| 7 | Scale pool | 64 | Concatenated BF16 scales, §7 |
| 8 | Tensor payloads | 64 each | Static INT4 / BF16 weight matrices, §10 |
| 9 | File checksum | — | Trailing 32-byte SHA-256 over bytes `[0, checksum_offset)` |

All offsets are absolute from file start. The section table locates sections 1 through 6; payload and scale locations are indexed in the tensor directory.

---

## 3. File header (128 bytes, LE)

| Offset | Size | Field | Value / notes |
|---|---|---|---|
| 0 | 8 | `magic` | `42 49 4E 46 45 52 00 01` (`"BINFER\0\1"`) |
| 8 | 4 | `version` | `1` (minor additions use flags and section table) |
| 12 | 4 | `flags` | bit 0 = text-only (must be 1); **bit 1 = MoE model enabled (1 for MoE, 0 for dense)**; bits 2–31 reserved (0) |
| 16 | 8 | `tensor_count` | Total number of indexed tensors (e.g. 1045 for Qwen3.5-MoE 35B) |
| 24 | 8 | `section_table_offset` | Absolute file offset of the section table |
| 32 | 4 | `section_count` | 5 for v1.0 dense models; **6 for v1.1 MoE models** |
| 36 | 4 | `alignment` | 64 (bytes) |
| 40 | 8 | `total_file_bytes` | Exact file size in bytes |
| 48 | 80 | `reserved` | Zeroes |

### Section table entry (32 bytes, LE)
- `id u32`: Section ID (1=Model Identity, 2=Architecture, 3=Tokenizer, 4=Quant Policy, 5=Tensor Directory, 6=MoE Metadata)
- `offset u64`: Absolute offset in file (must be 64-byte aligned)
- `bytes u64`: Exact length of section in bytes
- `crc32 u32`: CRC32 checksum over the section bytes
- `reserved u32 × 2`: Zeroes

Loaders must CRC-verify all metadata sections before trusting any offset or dimension contained within them.

---

## 4. Model identity section (Section 1)

- `source_repo`: NUL-terminated UTF-8 string (e.g. `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized`)
- `source_revision`: 40-char git commit SHA
- `source_total_bytes u64`: Source SafeTensors declared total size (e.g. 71,955,936,224)
- `converter`: Name of exporter (e.g. `AInfer binfer.py`)
- `converter_version`: Exporter version string
- `created_date`: ISO 8601 date string (YYYY-MM-DD)

---

## 5. Architecture metadata section (Section 2)

Key-value store (`u32 count` + entries: `u16 key_len`, `key`, `u8 type`, `value`) containing:
```text
arch = qwen3_5_moe
text_layers = 40
linear_layers = 30
full_layers = 10
full_attention_interval = 4
hidden_size = 2048
vocab_size = 248320
full_attention = {q_heads: 16, kv_heads: 2, head_dim: 256}
linear_attention = {key_heads: 16, value_heads: 32, key_dim: 128, value_dim: 128, conv_kernel: 4, ssm_dtype: fp32}
rope = {type: mrope_interleaved, section: [11, 11, 10], theta: 10000000, partial: 0.25}
rms_norm_eps = 1e-6
mlp_act = silu
source_dtype = bf16
vision = status:deferred
max_context_native = 262144
```

---

## 6. Tokenizer descriptor section (Section 3)

- `tokenizer_format`: `hf-json-v1`
- `vocab_size u32`: 248,320
- `bos_token_id u32`: 248044
- `eos_token_id u32`: 248044
- `sha256` hashes for `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, `chat_template.jinja`
- Added special tokens registry (`<|im_start|>`, `<|im_end|>`, `<|thought|>`, etc.)

---

## 7. Quantization policy section (Section 4) + Scale Pool (Section 7)

- `default_scheme u8`: `1` = INT4 symmetric, group size 128, BF16 scales
- `group_size u32`: 128
- `scale_dtype u8`: `0` = BF16 (2 bytes per 128 weights)
- **Quantization Exception Rules:**
  - `model.language_model.embed_tokens.weight` $\to$ BF16 unquantized
  - All RMSNorm weights $\to$ source precision (BF16/FP32)
  - Router gate weights (`*.mlp.gate.weight`, `*.shared_expert_gate.weight`) $\to$ FP32
  - Linear attention parameters (`*.linear_attn.A_log`, `dt_bias`, `conv1d.weight`) $\to$ FP32 / source precision
  - Visual encoder tensors $\to$ excluded (reject if present)
  - All MoE routed expert weights (`*.mlp.experts.gate_up_proj`, `*.mlp.experts.down_proj`) $\to$ INT4 symmetric group-128
  - All shared expert weights (`*.mlp.shared_expert.*`) $\to$ INT4 symmetric group-128

Scales are stored contiguously in the Scale Pool. Each directory entry provides `scale_offset` and `scale_bytes = ceil(num_elements / 128) * 2`.

---

## 8. Tensor directory (Section 5, 192 bytes per entry)

| Field | Size (bytes) | Type | Description |
|---|---|---|---|
| `name` | 64 | `char[64]` | NUL-terminated tensor name (redundant `model.language_model.` prefix stripped to fit $\le 63$ chars) |
| `ndim` | 1 | `u8` | Number of dimensions (1 to 8; 3 for batched expert banks) |
| `reserved_dim` | 7 | `u8[7]` | Zeroes |
| `shape` | 64 | `u64[8]` | Row-major logical dimensions (unused slots = 0) |
| `logical_dtype` | 1 | `u8` | Enum: 0=BF16, 1=FP16, 2=FP32, 3=INT4_SYM_G128, 4=INT8 |
| `storage_dtype` | 1 | `u8` | Stored physical data type enum |
| `layout_id` | 2 | `u16` | 0 = row-major dense; 1–15 = reserved hardware swizzles |
| `group_size` | 4 | `u32` | 128 for INT4, 0 for unquantized |
| `scale_offset` | 8 | `u64` | Byte offset into Scale Pool (Section 7) |
| `scale_bytes` | 8 | `u64` | Byte length of scales (0 if unquantized) |
| `data_offset` | 8 | `u64` | Absolute byte offset in file to payload |
| `data_bytes` | 8 | `u64` | Byte length of payload |
| `crc32` | 4 | `u32` | CRC32 of payload bytes |
| `reserved` | 12 | `u8[12]` | Zeroes |

---

## 9. MoE architecture & routing metadata (Section 6, v1.1+)

This section is present whenever `flags & 0x02 != 0`.

| Field | Size (bytes) | Type | Description |
|---|---|---|---|
| `num_experts` | 4 | `u32` | Total routed experts per layer (e.g. 256) |
| `num_experts_per_tok` | 4 | `u32` | Active routed experts selected per token ($k=8$) |
| `moe_intermediate_size` | 4 | `u32` | Intermediate dimension of each routed expert (512) |
| `shared_expert_intermediate_size` | 4 | `u32` | Intermediate dimension of shared expert (512) |
| `shared_expert_count` | 4 | `u32` | Number of shared experts per layer (1) |
| `routing_gate_dtype` | 1 | `u8` | 2 = FP32 (router gate matrices are FP32) |
| `norm_topk_prob` | 1 | `u8` | 1 = softmax probabilities renormalized over top-$k$ |
| `expert_tensor_layout` | 2 | `u16` | 0 = 3D packed bank `[num_experts, out_dim, in_dim]` |
| `layer_count` | 4 | `u32` | Number of MoE layers (40) |
| `reserved` | 36 | `u8[36]` | Zeroes |

Following the 64-byte header is an array of $40 \times 32$-byte layer routing descriptors:
- `layer_index u32`: 0 to 39
- `router_tensor_id u32`: Tensor directory index for `mlp.gate.weight`
- `shared_gate_tensor_id u32`: Tensor directory index for `mlp.shared_expert_gate.weight`
- `experts_gate_up_id u32`: Tensor directory index for `mlp.experts.gate_up_proj`
- `experts_down_id u32`: Tensor directory index for `mlp.experts.down_proj`
- `shared_expert_down_id u32`: Tensor directory index for `mlp.shared_expert.down_proj.weight`
- `shared_expert_gate_id u32`: Tensor directory index for `mlp.shared_expert.gate_proj.weight`
- `shared_expert_up_id u32`: Tensor directory index for `mlp.shared_expert.up_proj.weight`

---

## 10. Payloads & INT4 Packing (Section 8)

- All tensor payloads begin at 64-byte boundaries with zeroed padding.
- **INT4 Packing:** Two 4-bit values per byte. Low nibble (bits 0–3) = even index; high nibble (bits 4–7) = odd index within each group of 128.
- For 3D expert matrices `[num_experts, out_dim, in_dim]`: Expert $e$ begins at `data_offset + e * (out_dim * in_dim / 2)` and its scales begin at `scale_offset + e * (out_dim * in_dim / 128 * 2)`.

---

## 11. Loader Validation Requirements

A conforming AInfer Level Zero loader must reject a `.binfer` file before memory allocation if:
1. `magic` does not match `"BINFER\0\1"` or `version != 1`.
2. Bit 0 of `flags` is not 1 (non-text model).
3. If bit 1 of `flags` is set, Section 6 (MoE Metadata) must exist and pass CRC32 check.
4. Any section offset or payload range exceeds `total_file_bytes`.
5. Any payload or scale range overlaps another.
6. `num_experts` does not equal 256 or `num_experts_per_tok != 8`.
7. Any tensor directory payload CRC32 fails.
8. Any unexpected vision tensor is present in the directory.
