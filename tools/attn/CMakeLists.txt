# B60-era SYCL benchmark; skipped when AINFER_ENABLE_SYCL=OFF (e.g. 258v/clang++).
if(AINFER_ENABLE_SYCL)
add_executable(attn_t36 attn.cpp)
target_compile_options(attn_t36 PRIVATE -fsycl)
target_link_options(attn_t36 PRIVATE -fsycl)

enable_testing()
add_test(NAME attn_t36
  COMMAND attn_t36 ${CMAKE_CURRENT_BINARY_DIR}/report_t36.json)
set_tests_properties(attn_t36 PROPERTIES TIMEOUT 1800)
endif()
