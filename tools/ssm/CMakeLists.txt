# B60-era SYCL benchmark; skipped when AINFER_ENABLE_SYCL=OFF (e.g. 258v/clang++).
if(AINFER_ENABLE_SYCL)
add_executable(ssm_t37 ssm.cpp)
target_compile_options(ssm_t37 PRIVATE -fsycl)
target_link_options(ssm_t37 PRIVATE -fsycl)

enable_testing()
add_test(NAME ssm_t37
  COMMAND ssm_t37 ${CMAKE_CURRENT_BINARY_DIR}/report_t37ssm.json)
set_tests_properties(ssm_t37 PROPERTIES TIMEOUT 900)
endif()
