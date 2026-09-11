# `.binfer` Container Format — v1.0 (T2.1)

Versioned, hardware-oriented model container for AInfer. Single file, little-endian,
all sections 64-byte aligned. v1 scope is **text-only** INT4 weight-only models
(Qwen3.8-27B first); vision weights are excluded and rejected by v1 loaders.

## 1. Design principles

- One file, independently addressable tensors (no tar-style streaming dependency).
- No physical swizzle baked in: every tensor carries a `layout_id`; loader rejects unknown IDs.
- Deterministic conversion: identical source + policy ⇒ byte-identical file.
- Validation before allocation: every offset/length/checksum verified before any device malloc.
- 64-byte alignment everywhere (Xe2 cache-line friendly; zeroed padding).

## 2. File layout

| # | Section | Alignment | Contents |
| - | ------- | --------- | -------- |
| 0 | File header | 0 | 128 bytes, §3 |
| 1 | Model identity | 64 | source repo/revision, §4 |
| 2 | Architecture metadata | 64 | §5 |
| 3 | Tokenizer descriptor | 64 | identity + assets or hashes, §6 |
| 4 | Quantization policy | 64 | §7 |
| 5 | Tensor directory | 64 | N × 192-byte entries, §8 |
| 6 | Scale pool | 64 | concatenated per-tensor scales, §7 |
| 7 | Tensor payloads | 64 each | weights, §9 |
| 8 | File checksum | — | trailing 32-byte SHA-256 over bytes [0, checksum_offset) |

All offsets are absolute from file start. `section_count` in the header locates sections 1–5;
payload/scale locations come from the tensor directory.

## 3. File header (128 bytes, LE)

| Offset | Size | Field | Value / notes |
| ------ | ---- | ----- | ------------- |
| 0 | 8 | `magic` | `42 49 4E 46 45 52 00 01` (`"BINFER\0\1"`) |
| 8 | 4 | `version` | `1` |
| 12 | 4 | `flags` | bit0 = text-only (must be 1 in v1); bits 1–31 reserved, must be 0 |
| 16 | 8 | `tensor_count` | e.g. 866 for Qwen3.8-27B text v1 |
| 24 | 8 | `section_table_offset` | absolute offset of section table |
| 32 | 4 | `section_count` | 5 (sections 1–5) |
| 36 | 4 | `alignment` | 64 |
| 40 | 8 | `total_file_bytes` | exact file size |
| 48 | 80 | reserved | zeros |

Section table entry (32 bytes): `id u32`, `offset u64`, `bytes u64`, `crc32 u32`, `reserved u32 × 2`.
Loader CRC-checks sections 1–5 before trusting any offset inside them.

## 4. Model identity section

- `source_repo` (NUL-terminated string, e.g. `Qwen/Qwen3.8-27B`)
- `source_revision` (40-char git SHA, e.g. `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`)
- `source_total_bytes u64` (SafeTensors declared total, e.g. 55562855904)
- `converter` string + `converter_version`, `created_date` (YYYY-MM-DD)
- Loader rejects on revision mismatch unless an explicit override flag is set (override path is for debugging only and must be logged).

## 5. Architecture metadata section

Key-value store (u32 count + entries: u16 key_len, key, u8 type, value) with at minimum:

```text
arch=qwen3_5, text_layers=64, linear_layers=48, full_layers=16,
hidden=5120, intermediate=17408, vocab=248320,
q_heads=24, kv_heads=4, head_dim=256,
rope={type:mrope_interleaved, section:[11,11,10], theta:10000000, partial:0.25},
rms_eps=1e-6, mlp_act=silu, output_gate=swish, attn_output_gate=true,
tied_embeddings=false, source_dtype=bf16, ssm_dtype=fp32,
mtp_layers=1, vision=status:deferred, max_context_native=262144, v1_context_cap=4096
```

## 6. Tokenizer descriptor section

- `tokenizer_format` = `hf-json-v1`; `vocab_size u32` (e.g. 248077 core + specials)
- SHA-256 of `tokenizer.json`, `merges.txt`, `chat_template.jinja`, `tokenizer_config.json`
- Added-special-token list (required: T1.4 showed `tokenizer.json` alone drops
  `<|im_start|>`-family markup; T4.3 must register these before any comparison)
- Either embedded assets or the hashes above + pinned source revision (embed by default).

## 7. Quantization policy section + scale pool

- `default_scheme u8`: `1` = INT4 symmetric, group-size 128, BF16 scales
- `group_size u32` = 128; `scale_dtype` = BF16 (2 bytes/group)
- Exception table (tensor-name patterns kept at source precision):
  - `embed_tokens.weight` → BF16 (scenario A)
  - all 1-D norm weights → source precision
  - `linear_attn.A_log`, `linear_attn.dt_bias`, `linear_attn.conv1d.weight` → source precision
  - everything visual → absent (reject if present)
- All other 2-D text weights (505 tensors for Qwen3.8-27B) → INT4 symmetric g128.
- Scales for one tensor are contiguous in the scale pool; entry gives
  `scale_offset`/`scale_bytes` (= `ceil(elem/128) * 2`). Zero-point fields exist in the
  entry but must be zero-length for symmetric v1 (asymmetric reserved for future versions).

## 8. Tensor directory (section 5: raw entries, no length prefix; table CRC covers raw bytes)

Entry (192 bytes):

| Field | Size | Notes |
| ----- | ---- | ----- |
| `name` | 64 | NUL-terminated (longest Qwen3.8-27B name is 62 chars) |
| `ndim` | 1 | + 7 reserved |
| `shape` | 64 | 8 × u64, row-major logical order, unused lanes 0 |
| `logical_dtype` | 1 | enum: 0=BF16 1=FP16 2=FP32 3=INT4_SYM_G128 4=INT8 5=FP8_E4M3 |
| `storage_dtype` | 1 | same enum (as stored) |
| `layout_id` | 2 | §10 |
| `group_size` | 4 | 128 for INT4, else 0 |
| `scale_offset` / `scale_bytes` | 16 | into scale pool; 0/0 if unquantized |
| `data_offset` / `data_bytes` | 16 | payload location |
| `crc32` | 4 | of payload bytes |
| reserved | 12 | zeros (covers future zero-point refs) |

## 9. Payloads

Raw dense bytes, each tensor starting at a 64-byte boundary, zeroed padding between tensors.
INT4 packing v1: two 4-bit values per byte, low nibble = even index (little-endian within
group of 128). Row-major logical order at `layout_id 0`; swizzled layouts keep the same
entry shape with a different `layout_id` and identical CRC scope (CRC covers stored bytes).

## 10. Layout ID registry (v1)

| ID | Meaning | Status |
| -- | ------- | ------ |
| 0 | row-major dense | required |
| 1–15 | reserved Xe2 swizzles | TBD by T3.3 benchmarks |
| 16+ | experimental | loader must reject |

## 11. Loader validation (normative, implements T2.2 hooks)

Reject before allocating if any of: bad magic/version/flags; section CRC fail;
`tensor_count` mismatch (866 for this model); revision mismatch; unknown `layout_id`;
any offset/length out of `[0, total_file_bytes)`; overlap between payloads/scales;
payload CRC mismatch; visual-namespace tensor present; unsupported dtype enum.
Report the first failure with tensor name + expected/actual values.

## 12. Versioning

- Major bump (new magic byte 8 / `version`) = breaking; old loaders reject.
- Minor additions use reserved fields only. Asymmetric quantization, FP8 weights, and
  multimodal sections are explicitly v2+ scope, not v1.

## 13. Reference numbers — Qwen3.8-27B text v1 (from `manifest.json` + `memory_budget.json`)

- Text tensors: 866 (505 quantized 2-D + embed BF16 + 360 high-precision 0-D/1-D/conv/SSM scalars)
- BF16 source (text): 50.889 GiB → INT4 payload + scales: 12.508 GiB
- File estimate: ~14.9 GiB weights + ~0.2 MiB directory + scales included above + assets
- Excluded: 333 visual tensors (~0.86 GiB), 262K context (v1 cap 4K)

## 14. Open items

- Swizzle selection (T3.3) assigns IDs 1–15; baseline layout 0 always ships.
- Single-file vs sharded emission: single file decided (independently addressable payloads make sharding unnecessary).
