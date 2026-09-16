add_executable(l0load loader.cpp)
target_include_directories(l0load PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(l0load PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)

enable_testing()
add_test(NAME l0load
  COMMAND l0load ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                 ${CMAKE_CURRENT_BINARY_DIR}/report_l0load.json)
# Full 15 GB upload + 15 GB readback; generous timeout for USB-mounted source.
set_tests_properties(l0load PROPERTIES TIMEOUT 3600)

# T2.2 C++ negative suite: synthetic mutants through the real L0 binary.
find_program(VENV_PYTHON python3)
add_test(NAME l0negatives
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/negatives.py
                  ${CMAKE_CURRENT_BINARY_DIR}/l0load
                  ${CMAKE_CURRENT_BINARY_DIR}/report_l0neg.json)
set_tests_properties(l0negatives PROPERTIES TIMEOUT 1200 DEPENDS l0load)
