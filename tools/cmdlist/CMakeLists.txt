add_executable(bench_lists bench_lists.cpp)
target_include_directories(bench_lists PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(bench_lists PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
target_compile_options(bench_lists PRIVATE -fsycl)
target_link_options(bench_lists PRIVATE -fsycl)

# NOTE: graph_replay (SYCL Graph) is intentionally NOT built: the pinned
# toolchain's SYCL-Graph extension rejects this stack's backend
# ("opencl backend is not supported by SYCL Graph extension").
# The record/replay evidence path is kernel_replay (raw L0 lists + SPIR-V).

# Build-time SPIR-V (tools/cmdlist/extract_spv.py, replicating the icpx spir64
# driver path): objcopy bundle -> sycl-post-link (-split-esimd -lower-esimd,
# REQUIRED once kernels.cpp holds ESIMD code; plain llvm-spirv dies on 32-wide
# vectors) -> llvm-spirv (genx-intrinsics allowed) -> one module per kernel
# entry, selected by content. See docs/toolchain.md T5.3/GEMV-port notes.
get_filename_component(COMP_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
find_program(LLVM_SPIRV llvm-spirv
  PATHS ${COMP_BIN_DIR}/compiler /opt/intel/oneapi/compiler/2026.1/bin/compiler
  NO_DEFAULT_PATH)
find_program(SYCL_POST_LINK sycl-post-link
  PATHS ${COMP_BIN_DIR}/compiler /opt/intel/oneapi/compiler/2026.1/bin/compiler
  NO_DEFAULT_PATH)
if(NOT LLVM_SPIRV OR NOT SYCL_POST_LINK)
  message(FATAL_ERROR "llvm-spirv/sycl-post-link not found (needed for T5.3 SPIR-V)")
endif()
set(SPV_MODS silumul control norm res gemv attn ssmconv ssmrecur rope kvappend argmax1 argmax2 scalesmax quantize splitrepeat l2normqk betag rmsinv normgated resaddf splitqk batchnorm embed chunkgemm chunkgemmdb chunkattn chunkssmconv chunkssmrecur cvtf32f16 concat gemvm2 gemvm3 chunkkvappend chunkrope kvappendi8 attni8 tiledattn qkgemm wvgemm softmaxrow gatemul chunkqkgemm chunkwvgemm chunksoftmaxrow)
set(SPV_FILES "")
foreach(m IN LISTS SPV_MODS)
  list(APPEND SPV_FILES ${CMAKE_CURRENT_BINARY_DIR}/${m}.spv)
endforeach()
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels.o
  COMMAND ${CMAKE_CXX_COMPILER} -fsycl -fsycl-targets=spir64 -c
          ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cpp
          -o ${CMAKE_CURRENT_BINARY_DIR}/kernels.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cpp
  COMMENT "Compiling L0 kernel module (spir64 device image)")
add_custom_command(
  OUTPUT ${SPV_FILES}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/extract_spv.py
          ${CMAKE_CURRENT_BINARY_DIR}/kernels.o
          ${CMAKE_CURRENT_BINARY_DIR}
          ${SYCL_POST_LINK} ${LLVM_SPIRV}
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernels.o
          ${CMAKE_CURRENT_SOURCE_DIR}/extract_spv.py
  COMMENT "Extracting per-kernel SPIR-V modules")
add_custom_target(spv_mods ALL DEPENDS ${SPV_FILES})

add_executable(kernel_replay kernel_replay.cpp)
target_include_directories(kernel_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(kernel_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(kernel_replay spv_mods)

add_executable(control_replay control_replay.cpp)
target_include_directories(control_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(control_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(control_replay spv_mods)

# T5.3 layer-class prototype: 3-kernel recorded list (norm->silu->res),
# loading the content-selected module for each entry point.
add_executable(list_replay list_replay.cpp)
target_include_directories(list_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(list_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(list_replay spv_mods)

enable_testing()
add_test(NAME bench_lists
  COMMAND bench_lists ${CMAKE_CURRENT_BINARY_DIR}/report_cli.json)
set_tests_properties(bench_lists PROPERTIES TIMEOUT 1200)
add_test(NAME kernel_replay
  COMMAND kernel_replay ${CMAKE_CURRENT_BINARY_DIR}/silumul.spv
                         ${CMAKE_CURRENT_BINARY_DIR}/report_kernel.json)
set_tests_properties(kernel_replay PROPERTIES TIMEOUT 1200)
add_test(NAME control_replay
  COMMAND control_replay ${CMAKE_CURRENT_BINARY_DIR}/control.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/report_control.json)
set_tests_properties(control_replay PROPERTIES TIMEOUT 1200)
add_test(NAME list_replay
  COMMAND list_replay ${CMAKE_CURRENT_BINARY_DIR}/norm.spv
                       ${CMAKE_CURRENT_BINARY_DIR}/silumul.spv
                       ${CMAKE_CURRENT_BINARY_DIR}/res.spv
                       ${CMAKE_CURRENT_BINARY_DIR}/report_list.json)
set_tests_properties(list_replay PROPERTIES TIMEOUT 1200)

# T5.3 GEMV port: decode-exact ESIMD dp4a INT4 GEMV in a recorded list.
add_executable(gemv_replay gemv_replay.cpp)
target_include_directories(gemv_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(gemv_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(gemv_replay spv_mods)
add_test(NAME gemv_replay
  COMMAND gemv_replay ${CMAKE_CURRENT_BINARY_DIR}/gemv.spv
                       ${CMAKE_CURRENT_BINARY_DIR}/report_gemv.json)
set_tests_properties(gemv_replay PROPERTIES TIMEOUT 1200)

# T5.3 attention port: decode-exact GQA+gate core, control-bound context.
add_executable(attn_replay attn_replay.cpp)
target_include_directories(attn_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(attn_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(attn_replay spv_mods)
add_test(NAME attn_replay
  COMMAND attn_replay ${CMAKE_CURRENT_BINARY_DIR}/attn.spv
                       ${CMAKE_CURRENT_BINARY_DIR}/report_attn.json)
set_tests_properties(attn_replay PROPERTIES TIMEOUT 1200)

# T6.3 INT8 KV port: append + attention over per-token INT8 caches.
add_executable(attni8_replay attni8_replay.cpp)
target_include_directories(attni8_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(attni8_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(attni8_replay spv_mods)
add_test(NAME attni8_replay
  COMMAND attni8_replay ${CMAKE_CURRENT_BINARY_DIR}/kvappendi8.spv
                        ${CMAKE_CURRENT_BINARY_DIR}/attni8.spv
                        ${CMAKE_CURRENT_BINARY_DIR}/report_attni8.json)
set_tests_properties(attni8_replay PROPERTIES TIMEOUT 1200)

# T7.4 tiled attention probe: SLM-shared single-pass scan (synthetic).
add_executable(tiledattn_replay tiledattn_replay.cpp)
target_include_directories(tiledattn_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(tiledattn_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(tiledattn_replay spv_mods)
add_test(NAME tiledattn_replay
  COMMAND tiledattn_replay ${CMAKE_CURRENT_BINARY_DIR}/tiledattn.spv
                         ${CMAKE_CURRENT_BINARY_DIR}/report_tiledattn.json)
set_tests_properties(tiledattn_replay PROPERTIES TIMEOUT 1800)

# T7.4 QK-GEMM probe: fp16 x BF16-cache DPAS GEMM (GEMM-form attention step).
add_executable(qkgemm_replay qkgemm_replay.cpp)
target_include_directories(qkgemm_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(qkgemm_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(qkgemm_replay spv_mods)
add_test(NAME qkgemm_replay
  COMMAND qkgemm_replay ${CMAKE_CURRENT_BINARY_DIR}/qkgemm.spv
                       ${CMAKE_CURRENT_BINARY_DIR}/report_qkgemm.json)
set_tests_properties(qkgemm_replay PROPERTIES TIMEOUT 1200)

# T7.4 hybrid attention probe: qk-gemm + softmax + cvt + wv-gemm stages.
add_executable(hybridattn_replay hybridattn_replay.cpp)
target_include_directories(hybridattn_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(hybridattn_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(hybridattn_replay spv_mods)
add_test(NAME hybridattn_replay
  COMMAND hybridattn_replay ${CMAKE_CURRENT_BINARY_DIR}/qkgemm.spv
                         ${CMAKE_CURRENT_BINARY_DIR}/softmaxrow.spv
                         ${CMAKE_CURRENT_BINARY_DIR}/cvtf32f16.spv
                         ${CMAKE_CURRENT_BINARY_DIR}/wvgemm.spv
                         ${CMAKE_CURRENT_BINARY_DIR}/resaddf.spv
                         ${CMAKE_CURRENT_BINARY_DIR}/report_hybridattn.json)
set_tests_properties(hybridattn_replay PROPERTIES TIMEOUT 1800)

# T7.4 chunk production: full linear chunk layer on REAL weights (layer 0).
add_executable(chunklayerreal_replay chunklayerreal_replay.cpp)
target_include_directories(chunklayerreal_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(chunklayerreal_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(chunklayerreal_replay spv_mods)
add_test(NAME chunklayerreal_replay
  COMMAND chunklayerreal_replay ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                          ${CMAKE_CURRENT_BINARY_DIR}/norm.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/chunkgemm.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/chunkssmconv.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/chunkssmrecur.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/silumul.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/splitrepeat.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/l2normqk.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/betag.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/rmsinv.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/normgated.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/resaddf.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/cvtf32f16.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/report_chunklayerreal.json)
set_tests_properties(chunklayerreal_replay PROPERTIES TIMEOUT 1800)

# T7.4 chunk production: full full-attn chunk on REAL weights (layer 3).
add_executable(chunkattnreal_replay chunkattnreal_replay.cpp)
target_include_directories(chunkattnreal_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(chunkattnreal_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(chunkattnreal_replay spv_mods)
add_test(NAME chunkattnreal_replay
  COMMAND chunkattnreal_replay ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                          ${CMAKE_CURRENT_BINARY_DIR}/norm.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/cvtf32f16.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/chunkgemm.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/splitqk.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/batchnorm.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/chunkrope.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/chunkkvappend.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/chunkattn.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/silumul.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/resaddf.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/report_chunkattnreal.json)
set_tests_properties(chunkattnreal_replay PROPERTIES TIMEOUT 1800)

# T7.4 chunk production: GEMM-form full-attn chunk on REAL weights (layer 3).
add_executable(chunkqkwvreal_replay chunkqkwvreal_replay.cpp)
target_include_directories(chunkqkwvreal_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(chunkqkwvreal_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(chunkqkwvreal_replay spv_mods)
add_test(NAME chunkqkwvreal_replay
  COMMAND chunkqkwvreal_replay ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                           ${CMAKE_CURRENT_BINARY_DIR}/norm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/cvtf32f16.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkgemm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/splitqk.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/batchnorm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkrope.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkkvappend.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkqkgemm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunksoftmaxrow.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkwvgemm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/gatemul.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/silumul.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/resaddf.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/report_chunkqkwvreal.json)
set_tests_properties(chunkqkwvreal_replay PROPERTIES TIMEOUT 1800)

# T7.4 chunk production: full 64-layer chunked prefill on REAL weights.
add_executable(chunk64real_replay chunk64real_replay.cpp)
target_include_directories(chunk64real_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(chunk64real_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(chunk64real_replay spv_mods)
add_test(NAME chunk64real_replay
  COMMAND chunk64real_replay ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                           ${CMAKE_CURRENT_BINARY_DIR}/norm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkgemm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkssmconv.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkssmrecur.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/silumul.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/splitrepeat.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/l2normqk.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/betag.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/rmsinv.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/normgated.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/resaddf.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/cvtf32f16.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/splitqk.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/batchnorm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkrope.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkkvappend.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkqkgemm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunksoftmaxrow.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkwvgemm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/gatemul.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/report_chunk64real.json)
set_tests_properties(chunk64real_replay PROPERTIES TIMEOUT 3600)

# T7.4 chunk production: two-chunk 64-layer prefill (carry + continuity).
add_executable(chunk64mc_replay chunk64mc_replay.cpp)
target_include_directories(chunk64mc_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(chunk64mc_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(chunk64mc_replay spv_mods)
add_test(NAME chunk64mc_replay
  COMMAND chunk64mc_replay ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                           ${CMAKE_CURRENT_BINARY_DIR}/norm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkgemm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkssmconv.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkssmrecur.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/silumul.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/splitrepeat.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/l2normqk.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/betag.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/rmsinv.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/normgated.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/resaddf.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/cvtf32f16.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/splitqk.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/batchnorm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkrope.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkkvappend.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkqkgemm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunksoftmaxrow.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkwvgemm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/gatemul.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/report_chunk64mc.json)
set_tests_properties(chunk64mc_replay PROPERTIES TIMEOUT 3600)

# T7.4 chunked prefill: GEMM-form chunk attention probe (causal QK + softmax + WV).
add_executable(chunkqkwv_replay chunkqkwv_replay.cpp)
target_include_directories(chunkqkwv_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(chunkqkwv_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(chunkqkwv_replay spv_mods)
add_test(NAME chunkqkwv_replay
  COMMAND chunkqkwv_replay ${CMAKE_CURRENT_BINARY_DIR}/chunkqkgemm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunksoftmaxrow.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/cvtf32f16.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkwvgemm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/report_chunkqkwv.json)
set_tests_properties(chunkqkwv_replay PROPERTIES TIMEOUT 1800)

# T7.2 MTP draft slice: MTP-1 draft as one recorded list, real weights.
add_executable(mtpdraft_replay mtpdraft_replay.cpp)
target_include_directories(mtpdraft_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(mtpdraft_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(mtpdraft_replay spv_mods)
add_test(NAME mtpdraft_replay
  COMMAND mtpdraft_replay ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                          ${CMAKE_CURRENT_BINARY_DIR}/norm.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/gemv.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/concat.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/splitqk.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/batchnorm.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/rope.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/kvappend.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/attn.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/scalesmax.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/quantize.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/silumul.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/resaddf.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/argmax1.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/argmax2.spv
                          ${CMAKE_CURRENT_BINARY_DIR}/report_mtpdraft.json)
set_tests_properties(mtpdraft_replay PROPERTIES TIMEOUT 1800)

# T7.2 chained drafts: Int4GemvM3 triple-lane GEMV vs M2+M1, real weights.
add_executable(gemvm3_replay gemvm3_replay.cpp)
target_include_directories(gemvm3_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(gemvm3_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(gemvm3_replay spv_mods)
add_test(NAME gemvm3_replay
  COMMAND gemvm3_replay ${CMAKE_CURRENT_BINARY_DIR}/gemvm3.spv
                        ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                        ${CMAKE_CURRENT_BINARY_DIR}/report_gemvm3.json)
set_tests_properties(gemvm3_replay PROPERTIES TIMEOUT 1200)

# Fused FlashAttention probe: ChunkFlashAttn vs host fp32 FA reference.
add_executable(chunkflash_replay chunkflash_replay.cpp)
target_include_directories(chunkflash_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(chunkflash_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(chunkflash_replay spv_mods)
add_test(NAME chunkflash_replay
  COMMAND chunkflash_replay ${CMAKE_CURRENT_BINARY_DIR}/chunkflashattn.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/report_chunkflash.json)
set_tests_properties(chunkflash_replay PROPERTIES TIMEOUT 1200)

# T7.4 long-context scan: AttnCore at T=1K/2K/4K (same SPIR-V, 4K sizing).
add_executable(attnfar_replay attnfar_replay.cpp)
target_include_directories(attnfar_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(attnfar_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(attnfar_replay spv_mods)
add_test(NAME attnfar_replay
  COMMAND attnfar_replay ${CMAKE_CURRENT_BINARY_DIR}/attn.spv
                         ${CMAKE_CURRENT_BINARY_DIR}/report_attnfar.json)
set_tests_properties(attnfar_replay PROPERTIES TIMEOUT 1800)

# T5.3 SSM port: stateful conv-k4+silu and 48-head FP32 recurrence.
add_executable(ssm_replay ssm_replay.cpp)
target_include_directories(ssm_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(ssm_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(ssm_replay spv_mods)
add_test(NAME ssm_replay
  COMMAND ssm_replay ${CMAKE_CURRENT_BINARY_DIR}/ssmconv.spv
                      ${CMAKE_CURRENT_BINARY_DIR}/ssmrecur.spv
                      ${CMAKE_CURRENT_BINARY_DIR}/report_ssm.json)
set_tests_properties(ssm_replay PROPERTIES TIMEOUT 1200)

# T5.3 RoPE/KV port: decode-exact half-rotation + cache append.
add_executable(rope_replay rope_replay.cpp)
target_include_directories(rope_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(rope_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(rope_replay spv_mods)
add_test(NAME rope_replay
  COMMAND rope_replay ${CMAKE_CURRENT_BINARY_DIR}/rope.spv
                       ${CMAKE_CURRENT_BINARY_DIR}/kvappend.spv
                       ${CMAKE_CURRENT_BINARY_DIR}/report_rope.json)
set_tests_properties(rope_replay PROPERTIES TIMEOUT 1200)

# T7.4 far-slot KV: KvAppend into 64K-deep caches at {0, 32767, 65533}.
add_executable(kvfar_replay kvfar_replay.cpp)
target_include_directories(kvfar_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(kvfar_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(kvfar_replay spv_mods)
add_test(NAME kvfar_replay
  COMMAND kvfar_replay ${CMAKE_CURRENT_BINARY_DIR}/kvappend.spv
                       ${CMAKE_CURRENT_BINARY_DIR}/report_kvfar.json)
set_tests_properties(kvfar_replay PROPERTIES TIMEOUT 1200)

# T5.3 argmax port: decode-exact two-stage SLM reduction, local args by size.
add_executable(argmax_replay argmax_replay.cpp)
target_include_directories(argmax_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(argmax_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(argmax_replay spv_mods)
add_test(NAME argmax_replay
  COMMAND argmax_replay ${CMAKE_CURRENT_BINARY_DIR}/argmax1.spv
                         ${CMAKE_CURRENT_BINARY_DIR}/argmax2.spv
                         ${CMAKE_CURRENT_BINARY_DIR}/report_argmax.json)
set_tests_properties(argmax_replay PROPERTIES TIMEOUT 1200)

# T5.3 layer adoption: one full linear layer as a single recorded list.
add_executable(layerlin_replay layerlin_replay.cpp)
target_include_directories(layerlin_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(layerlin_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(layerlin_replay spv_mods)
add_test(NAME layerlin_replay
  COMMAND layerlin_replay ${CMAKE_CURRENT_BINARY_DIR}/norm.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/gemv.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/ssmconv.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/ssmrecur.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/silumul.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/scalesmax.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/quantize.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/splitrepeat.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/l2normqk.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/betag.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/rmsinv.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/normgated.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/resaddf.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/report_layerlin.json)
set_tests_properties(layerlin_replay PROPERTIES TIMEOUT 1200)

# T5.3 layer adoption (full-attention): one layer as a single recorded list.
add_executable(layerattn_replay layerattn_replay.cpp)
target_include_directories(layerattn_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(layerattn_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(layerattn_replay spv_mods)
add_test(NAME layerattn_replay
  COMMAND layerattn_replay ${CMAKE_CURRENT_BINARY_DIR}/norm.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/gemv.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/splitqk.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/batchnorm.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/rope.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/kvappend.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/attn.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/scalesmax.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/quantize.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/silumul.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/resaddf.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/report_layerattn.json)
set_tests_properties(layerattn_replay PROPERTIES TIMEOUT 1200)

# T7.4 chunked prefill: tiled INT4 DPAS GEMM in a recorded list.
add_executable(chunkgemm_replay chunkgemm_replay.cpp)
target_include_directories(chunkgemm_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(chunkgemm_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(chunkgemm_replay spv_mods)
add_test(NAME chunkgemm_replay
  COMMAND chunkgemm_replay ${CMAKE_CURRENT_BINARY_DIR}/chunkgemm.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/report_chunkgemm.json)
set_tests_properties(chunkgemm_replay PROPERTIES TIMEOUT 1200)

# Prefill GEMM slice pairs: ChunkGemmDB vs CPU fp32 ref (tol 1e-3).
add_executable(gemmdb_replay gemmdb_replay.cpp)
target_include_directories(gemmdb_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(gemmdb_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(gemmdb_replay spv_mods)
add_test(NAME gemmdb_replay
  COMMAND gemmdb_replay ${CMAKE_CURRENT_BINARY_DIR}/chunkgemmdb.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/report_gemmdb.json)
set_tests_properties(gemmdb_replay PROPERTIES TIMEOUT 1200)

# T7.4 chunked prefill: causal chunk-attention over a cache prefix.
add_executable(chunkattn_replay chunkattn_replay.cpp)
target_include_directories(chunkattn_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(chunkattn_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(chunkattn_replay spv_mods)
add_test(NAME chunkattn_replay
  COMMAND chunkattn_replay ${CMAKE_CURRENT_BINARY_DIR}/chunkattn.spv
                            ${CMAKE_CURRENT_BINARY_DIR}/report_chunkattn.json)
set_tests_properties(chunkattn_replay PROPERTIES TIMEOUT 1200)

# T7.4 chunked prefill: SSM chunk orchestration (persistent cross-chunk state).
add_executable(chunkssm_replay chunkssm_replay.cpp)
target_include_directories(chunkssm_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(chunkssm_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(chunkssm_replay spv_mods)
add_test(NAME chunkssm_replay
  COMMAND chunkssm_replay ${CMAKE_CURRENT_BINARY_DIR}/chunkssmconv.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/chunkssmrecur.spv
                           ${CMAKE_CURRENT_BINARY_DIR}/report_chunkssm.json)
set_tests_properties(chunkssm_replay PROPERTIES TIMEOUT 1200)

# T7.4 chunked prefill: full linear-layer chunk orchestration.
add_executable(chunklayer_replay chunklayer_replay.cpp)
target_include_directories(chunklayer_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(chunklayer_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(chunklayer_replay spv_mods)
add_test(NAME chunklayer_replay
  COMMAND chunklayer_replay ${CMAKE_CURRENT_BINARY_DIR}/norm.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/chunkgemm.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/chunkssmconv.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/chunkssmrecur.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/silumul.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/splitrepeat.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/l2normqk.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/betag.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/rmsinv.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/normgated.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/resaddf.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/cvtf32f16.spv
                             ${CMAKE_CURRENT_BINARY_DIR}/report_chunklayer.json)
set_tests_properties(chunklayer_replay PROPERTIES TIMEOUT 1200)

# T5.3 adoption on real weights: linear layer 0 from the .binfer arenas vs
# the certified SYCL binary's layer dump.
add_executable(layer0real_replay layer0real_replay.cpp)
target_include_directories(layer0real_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(layer0real_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(layer0real_replay spv_mods)

# T5.3 adoption on real weights (full-attention): layer 3 from the arenas vs
# dump slot 3 (input: slot 2). Manual run (needs model + certified dump).
add_executable(layer3real_replay layer3real_replay.cpp)
target_include_directories(layer3real_replay PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(layer3real_replay PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(layer3real_replay spv_mods)

# T8.2 release gate: strict-parse every tracked JSON evidence report (Gate A).
# Host-only (no GPU, ms-scale) — belongs in the per-change FAST gate.
add_test(NAME reports_json
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../validate_reports.py)
set_tests_properties(reports_json PROPERTIES TIMEOUT 300)

# T8.4 release gate: pinned tokenizer golden (Gate B). Host-only, needs the
# project venv (`tokenizers`, `jinja2`) — in the per-change FAST gate.
add_test(NAME tokenizer_golden
  COMMAND $ENV{HOME}/.venvs/ainfer/bin/python
          ${CMAKE_CURRENT_SOURCE_DIR}/../tokenizer/validate_golden.py)
set_tests_properties(tokenizer_golden PROPERTIES TIMEOUT 300)
