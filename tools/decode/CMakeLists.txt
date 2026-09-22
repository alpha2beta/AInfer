# B60-era SYCL decode; skipped when AINFER_ENABLE_SYCL=OFF (e.g. 258v/clang++).
# decode_l0 and the host unit tests below are SYCL-free and always built.
if(AINFER_ENABLE_SYCL)
add_executable(decode decode.cpp)
target_include_directories(decode PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(decode PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
target_compile_options(decode PRIVATE -fsycl)
target_link_options(decode PRIVATE -fsycl)

enable_testing()
add_test(NAME decode
  COMMAND decode ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                 4 2 ${CMAKE_CURRENT_BINARY_DIR}/report_decode.json)
set_tests_properties(decode PROPERTIES TIMEOUT 3600)
endif()

# X2 (258v preset): host-only unit tests, no SYCL, no device, no model.
# T9.1 arena/overflow checks + T9.2 step-guard checks (both header-only use of
# runtime_258v.h, so no Level Zero link needed).
add_executable(test_arena_spans test_arena_spans.cpp)
target_include_directories(test_arena_spans PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
add_test(NAME test_arena_spans COMMAND test_arena_spans)
set_tests_properties(test_arena_spans PROPERTIES TIMEOUT 120)

add_executable(test_exec_guards test_exec_guards.cpp)
target_include_directories(test_exec_guards PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
add_test(NAME test_exec_guards COMMAND test_exec_guards)
set_tests_properties(test_exec_guards PROPERTIES TIMEOUT 120)

# T5.3 loop adoption: full 64-layer decode as recorded raw-L0 lists.
# Manual parity runs (needs model + SPIR-V dir); the SYCL `decode` stays green.
# AINFER_ZE_LOADER mirrors tools/l0probe (Ubuntu default; 258v preset points
# at the sysroot loader since CachyOS has no system libze_loader).
set(AINFER_ZE_LOADER "/usr/lib/x86_64-linux-gnu/libze_loader.so.1"
    CACHE FILEPATH "Level Zero loader library")
add_executable(decode_l0 decode_l0.cpp)
target_include_directories(decode_l0 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(decode_l0 PRIVATE
  ${AINFER_ZE_LOADER})
if(TARGET spv_mods)
  add_dependencies(decode_l0 spv_mods)
endif()
