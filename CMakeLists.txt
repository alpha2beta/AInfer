cmake_minimum_required(VERSION 3.16)
project(ainfer CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
option(AINFER_BUILD_CMDLIST "Build SPIR-V command-list targets" ON)
add_subdirectory(tools/l0probe)
add_subdirectory(tools/esimd_check)
add_subdirectory(tools/l0load)
add_subdirectory(tools/bench)
add_subdirectory(tools/gemv)
add_subdirectory(tools/norm)
add_subdirectory(tools/prefill)
add_subdirectory(tools/attn)
add_subdirectory(tools/ssm)
add_subdirectory(tools/sample)
add_subdirectory(tools/blkexec)
add_subdirectory(tools/blkblock)
add_subdirectory(tools/decode)
if(AINFER_BUILD_CMDLIST)
  add_subdirectory(tools/cmdlist)
endif()
add_subdirectory(tools/kvfit)

# X2 (258v preset): script-driven regression gates. These wrap existing
# repeatable harnesses (no CMake-built targets of their own):
# - kv8_primitive_tests: builds kv8_primitive.spv + host/device primitive
#   checks, asserts PASSED on both JSON reports (Arc 140V, ~1 min).
# - t12_rollback: offline rootless toolchain snapshot/rollback check; needs
#   build-258v/tools/l0probe/l0_timestamp_smoke built first.
add_test(NAME kv8_primitive_tests
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tools/kernels_258v/run_kv8_primitive_tests.sh)
set_tests_properties(kv8_primitive_tests PROPERTIES TIMEOUT 600)
add_test(NAME t12_rollback
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/toolchain/t12_rollback.py)
set_tests_properties(t12_rollback PROPERTIES TIMEOUT 600)
