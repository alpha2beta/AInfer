add_executable(gemv_t32 gemv.cpp)
target_compile_options(gemv_t32 PRIVATE -fsycl)
target_link_options(gemv_t32 PRIVATE -fsycl)

add_executable(dp4a_probe dp4a_probe.cpp)
target_compile_options(dp4a_probe PRIVATE -fsycl)
target_link_options(dp4a_probe PRIVATE -fsycl)
add_test(NAME dp4a_probe COMMAND dp4a_probe)
set_tests_properties(dp4a_probe PROPERTIES TIMEOUT 300)

enable_testing()
add_test(NAME gemv_t32
  COMMAND gemv_t32 ${CMAKE_CURRENT_BINARY_DIR}/report_t32.json)
set_tests_properties(gemv_t32 PROPERTIES TIMEOUT 1800)
