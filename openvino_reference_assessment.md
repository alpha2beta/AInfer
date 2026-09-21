# OpenVINO as T7.5 Reference Runtime — Assessment (saved, not actioned)

**Date:** 2026-09-20 · **Status:** SKIPPED for now (user decision). This doc preserves
the feasibility assessment so the spike can be started later without re-deriving it.

## Question
Should OpenVINO GenAI replace/supplement llama.cpp Vulkan as the T7.5
controlled-comparison baseline on the 258V box?

## Support status (researched 2026-09-20 — favorable)
- **optimum-intel #1689 (merged):** Qwen3.5 / Qwen3.5-MoE / Qwen3.6 support —
  `RecurrentAttentionCellOp` conversion rule for GatedDeltaNet patching,
  3-GEMM MoE pattern (separate gate/up, transformation-friendly for fusion
  into the internal MoE op), stateful pipeline support.
- **OpenVINO 2026.x release notes:** BF16/FP16 GatedDeltaNet JIT kernels,
  CausalConv1D support, hybrid KV + linear-state cache for SDPA and PA
  backends, Xe2/Xe3 XAttention preview. The "linear-attention kernels missing"
  gap from the B70-cookbook era is closed upstream.
- **Known rough edge (optimum-intel #1721):** `qwen3_5_moe` export entry-point
  registers `image-text-to-text` only. Workarounds exist: VLM export runs
  text-only generation fine on Intel GPU (confirmed on B70), or extract the
  text backbone via transformers first and export with `text-generation`.
  Manageable, not blocking.
- Related: Optimum #1754 documents an OV-traceable `qwen3_5_moe` port
  (functional state IO, explicit KV cache, compute-all+mask MoE option).

## Hard parts (honest)
1. **Exporting OUR fine-tune (71 GB BF16) on a 30 GB RAM box is the go/no-go
   gate.** NNCF INT4 export normally loads full weights first. Needs low-memory
   tricks (disk-offloaded `device_map`, or FP16→INT4 streaming) — multi-hour,
   possibly OOM-killed experiment. Nothing else matters if this fails.
2. **Isolation mandatory.** OV GenAI + optimum-intel + transformers≥5.2 are
   fast-moving deps that must NOT go into the pinned `~/.venvs/ainfer` venv.
   Use a separate throwaway venv.
3. **Comparability caveats even on success:** OpenVINO-INT4 vs our INT4-g128
   (different quantization), Linux compute-runtime vs Windows 101.8826 driver
   stack behind the external 525 tok/s table. Still strictly better than the
   current llama.cpp-Vulkan-only baseline.

## Proposed spike (when resumed — time-boxed, mostly unattended)
1. Isolated venv: `openvino`, `openvino-genai`, `optimum-intel`,
   `transformers>=5.2`; snapshot versions BEFORE/AFTER.
2. Attempt INT4 export of `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes` →
   **go/no-go decision point, report back before spending more**.
3. If export succeeds: GenAI `LLMPipeline` on GPU, matched P=441 + sweep,
   record as the new T7.5 reference.
- Preconditions at assessment time: 184 GB free on /home (fine); 30 GB RAM
  (the risk); no OpenVINO anywhere on the box (venv or system).
