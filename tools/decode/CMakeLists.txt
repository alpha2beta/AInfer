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

# T5.3 loop adoption: full 64-layer decode as recorded raw-L0 lists.
# Manual parity runs (needs model + SPIR-V dir); the SYCL `decode` stays green.
add_executable(decode_l0 decode_l0.cpp)
target_include_directories(decode_l0 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(decode_l0 PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
add_dependencies(decode_l0 spv_mods)
