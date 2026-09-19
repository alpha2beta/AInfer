cmake_minimum_required(VERSION 3.16)
project(ainfer_l0load CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(ZE_LOADER_LIB NAMES ze_loader libze_loader.so.1
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../toolchain/sysroot/usr/lib
        /usr/lib /usr/lib/x86_64-linux-gnu NO_DEFAULT_PATH)
if(NOT ZE_LOADER_LIB)
  find_library(ZE_LOADER_LIB NAMES ze_loader libze_loader.so.1)
endif()

add_executable(l0load loader.cpp)
target_include_directories(l0load PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../toolchain/sysroot/usr/include)
target_link_libraries(l0load PRIVATE ${ZE_LOADER_LIB})

enable_testing()
add_test(NAME l0load
  COMMAND l0load ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer
                 ${CMAKE_CURRENT_BINARY_DIR}/report_l0load.json)
set_tests_properties(l0load PROPERTIES TIMEOUT 3600)

find_program(VENV_PYTHON python3)
add_test(NAME l0negatives
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/negatives.py
                  ${CMAKE_CURRENT_BINARY_DIR}/l0load
                  ${CMAKE_CURRENT_BINARY_DIR}/report_l0neg.json)
set_tests_properties(l0negatives PROPERTIES TIMEOUT 1200 DEPENDS l0load)
