cmake_minimum_required(VERSION 3.16)
project(ainfer_l0probe CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vendored Level Zero core headers (v1.28.2, matches loader 1.28.2).
# System provides only Intel extension headers + versioned libze_loader
# (no .so dev symlink without sudo), so link the versioned file directly.
add_executable(l0probe probe.cpp)
target_include_directories(l0probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(l0probe PRIVATE /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
