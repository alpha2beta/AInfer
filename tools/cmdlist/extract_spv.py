#!/usr/bin/env python3
"""T5.3 build-time SPIR-V extraction (stdlib only).

Flow (replicates the icpx spir64 driver path for ESIMD code):
  kernels.o --objcopy--> bundle.bc --sycl-post-link(-split-esimd -lower-esimd)-->
  spl images --llvm-spirv(-allow-unknown-intrinsics=llvm.genx.)--> .spv modules.

Images are selected BY ENTRY-POINT CONTENT (post-link image numbering is not
stable across kernel-set changes), then copied to fixed output names, one
module per kernel entry:
  SiluMul -> silumul.spv, ControlAdd -> control.spv, RMSNormW -> norm.spv,
  ResAdd -> res.spv, Int4Gemv -> gemv.spv, AttnCore -> attn.spv,
  SsmConv -> ssmconv.spv, SsmRecur -> ssmrecur.spv, RopeApply -> rope.spv,
  KvAppend -> kvappend.spv, ArgmaxS1 -> argmax1.spv, ArgmaxS2 -> argmax2.spv,
  ScalesMax/Quantize/SplitRepeat/L2NormQK/BetaG/RmsInv/NormGated/ResAddF likewise,
  SplitQK -> splitqk.spv, BatchNorm -> batchnorm.spv, Embed -> embed.spv,
  ChunkGemm -> chunkgemm.spv, ChunkAttn -> chunkattn.spv,
  CvtF32F16 -> cvtf32f16.spv.
  ChunkSsmConv -> chunkssmconv.spv, ChunkSsmRecur -> chunkssmrecur.spv.

Usage: extract_spv.py <kernels.o> <outdir> <sycl-post-link> <llvm-spirv>
"""
import glob
import os
import shutil
import struct
import subprocess
import sys

BUNDLE_SECTION = "__CLANG_OFFLOAD_BUNDLE__sycl-spir64-unknown-unknown"
WANTED = {
    "_ZTS7SiluMul": "silumul.spv",
    "_ZTS10ControlAdd": "control.spv",
    "_ZTS8RMSNormW": "norm.spv",
    "_ZTS6ResAdd": "res.spv",
    "_ZTS8Int4Gemv": "gemv.spv",
    "_ZTS8AttnCore": "attn.spv",
    "_ZTS7SsmConv": "ssmconv.spv",
    "_ZTS8SsmRecur": "ssmrecur.spv",
    "_ZTS9RopeApply": "rope.spv",
    "_ZTS8KvAppend": "kvappend.spv",
    "_ZTS8ArgmaxS1": "argmax1.spv",
    "_ZTS8ArgmaxS2": "argmax2.spv",
    "_ZTS9ScalesMax": "scalesmax.spv",
    "_ZTS8Quantize": "quantize.spv",
    "_ZTS11SplitRepeat": "splitrepeat.spv",
    "_ZTS8L2NormQK": "l2normqk.spv",
    "_ZTS5BetaG": "betag.spv",
    "_ZTS6RmsInv": "rmsinv.spv",
    "_ZTS9NormGated": "normgated.spv",
    "_ZTS7ResAddF": "resaddf.spv",
    "_ZTS7SplitQK": "splitqk.spv",
    "_ZTS9BatchNorm": "batchnorm.spv",
    "_ZTS5Embed": "embed.spv",
    "_ZTS9ChunkGemm": "chunkgemm.spv",
    "_ZTS9ChunkAttn": "chunkattn.spv",
    "_ZTS12ChunkSsmConv": "chunkssmconv.spv",
    "_ZTS13ChunkSsmRecur": "chunkssmrecur.spv",
    "_ZTS9CvtF32F16": "cvtf32f16.spv",
    "_ZTS7Concat2": "concat.spv",
    "_ZTS10KvAppendI8": "kvappendi8.spv",
    "_ZTS10AttnCoreI8": "attni8.spv",
    "_ZTS9TiledAttn": "tiledattn.spv",
    "_ZTS6QkGemm": "qkgemm.spv",
    "_ZTS6WvGemm": "wvgemm.spv",
    "_ZTS10SoftmaxRow": "softmaxrow.spv",
    "_ZTS7GateMul": "gatemul.spv",
    "_ZTS13ChunkKvAppend": "chunkkvappend.spv",
    "_ZTS9ChunkRope": "chunkrope.spv",
    "_ZTS11ChunkQkGemm": "chunkqkgemm.spv",
    "_ZTS11ChunkWvGemm": "chunkwvgemm.spv",
    "_ZTS15ChunkSoftmaxRow": "chunksoftmaxrow.spv",
}
SPV_MAGIC = b"\x03\x02\x23\x07"
# Flag set captured from the icpx spir64 driver link (2026.1.1); the two
# allow-unknown-intrinsics repetitions are verbatim from the driver.
SPV_EXT = ("-all,+SPV_INTEL_vector_compute,+SPV_INTEL_subgroups,"
           "+SPV_INTEL_media_block_io,+SPV_INTEL_joint_matrix,"
           "+SPV_KHR_cooperative_matrix,"
           "+SPV_INTEL_bfloat16_conversion,+SPV_INTEL_fp_fast_math_mode,"
           "+SPV_INTEL_arbitrary_precision_integers,"
           "+SPV_INTEL_variable_length_array,+SPV_INTEL_long_composites,"
           "+SPV_INTEL_optnone,+SPV_KHR_bit_instructions,"
           "+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr")


def entries(path):
    with open(path, "rb") as f:
        d = f.read()
    if d[:4] != SPV_MAGIC:
        return []
    w = struct.unpack("<" + "I" * (len(d) // 4), d)
    names = []
    for i in range(5, len(w)):
        op = w[i] & 0xFFFF
        ln = w[i] >> 16
        if op == 15 and ln > 3:
            raw = b"".join(struct.pack("<I", x) for x in w[i + 3:i + ln])
            name = raw.split(b"\0")[0].decode(errors="replace")
            if name.startswith("_ZTS") and "RoundedRange" not in name:
                names.append(name)
    return names


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED:", " ".join(cmd), file=sys.stderr)
        print(r.stderr[-3000:], file=sys.stderr)
        sys.exit(1)
    return r


def main():
    obj, outdir, postlink, spirv = sys.argv[1:5]
    work = os.path.join(outdir, "spvwork")
    os.makedirs(work, exist_ok=True)
    bundle = os.path.join(work, "bundle.bc")
    run(["objcopy", "--dump-section",
         "%s=%s" % (BUNDLE_SECTION, bundle), obj, os.devnull])
    run([postlink, "-O2", "-split=auto", "-split-esimd", "-lower-esimd",
         "-emit-only-kernels-as-entry-points",
         "-o", os.path.join(work, "spl.table"), bundle], )
    # Translate every split image (run in work dir; outputs land next to .bc).
    for bc in sorted(glob.glob(os.path.join(work, "spl_*.bc"))):
        run([spirv, "--spirv-allow-unknown-intrinsics=llvm.genx.",
             "--spirv-allow-unknown-intrinsics=llvm.genx.",
             "--spirv-ext=" + SPV_EXT, bc, "-o", bc[:-3] + ".spv"])
    found = {}
    for spv in sorted(glob.glob(os.path.join(work, "spl_*.spv"))):
        # Prefer ESIMD images: a vectorized entry exists in BOTH its
        # spl_esimd_* image (real implementation) and a scalar image
        # (outline/stub), and sorted order would otherwise let the scalar
        # copy win (this silently ran ChunkSsmRecur scalar at 193 ms/chunk
        # despite a vector source — always verify genx markers + timing).
        esimd = "esimd" in os.path.basename(spv)
        for e in entries(spv):
            if e not in found or esimd:
                found[e] = spv
    for entry, name in WANTED.items():
        if entry not in found:
            print("entry %s not found in any image; saw %s"
                  % (entry, sorted(found)), file=sys.stderr)
            sys.exit(1)
        dst = os.path.join(outdir, name)
        shutil.copyfile(found[entry], dst)
        print("%s <- %s (%s)" % (name, os.path.basename(found[entry]), entry))


if __name__ == "__main__":
    main()
