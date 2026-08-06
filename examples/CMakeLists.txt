cmake_minimum_required(VERSION 3.25)
project(MonteCarloExamples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Compiler flags ---
add_compile_options(-Wall -Wextra -Wpedantic)

# --- Find dependencies ---
# GoogleTest (optional — tests won't build if not found)
find_package(GTest CONFIG)
# Google Benchmark (optional — benchmarks won't build if not found)
find_package(benchmark CONFIG)

# --- Core library (header-only) ---
add_library(mc_framework INTERFACE)
target_include_directories(mc_framework INTERFACE ${CMAKE_SOURCE_DIR}/include)

# --- Algorithm executables ---
add_executable(metropolis src/metropolis.cpp)
target_link_libraries(metropolis PRIVATE mc_framework)

add_executable(swendsen_wang src/swendsen_wang.cpp)
target_link_libraries(swendsen_wang PRIVATE mc_framework)

add_executable(wolff src/wolff.cpp)
target_link_libraries(wolff PRIVATE mc_framework)

add_executable(heisenberg_metropolis src/heisenberg_metropolis.cpp)
target_link_libraries(heisenberg_metropolis PRIVATE mc_framework)

add_executable(heisenberg_wolff src/heisenberg_wolff.cpp)
target_link_libraries(heisenberg_wolff PRIVATE mc_framework)

# --- q-state Potts (self-contained; Ising binaries above left untouched) ---
add_executable(potts_metropolis src/potts_metropolis.cpp)
target_link_libraries(potts_metropolis PRIVATE mc_framework)

add_executable(potts_swendsen_wang src/potts_swendsen_wang.cpp)
target_link_libraries(potts_swendsen_wang PRIVATE mc_framework)

add_executable(potts_wolff src/potts_wolff.cpp)
target_link_libraries(potts_wolff PRIVATE mc_framework)

# --- Tests ---
if(GTest_FOUND)
    enable_testing()
    # add_executable(test_energy tests/test_energy.cpp)
    # target_link_libraries(test_energy PRIVATE mc_framework GTest::gtest_main)
    # add_test(NAME test_energy COMMAND test_energy)
endif()

# --- Benchmarks ---
if(benchmark_FOUND)
    # add_executable(scaling_bench benchmarks/scaling_bench.cpp)
    # target_link_libraries(scaling_bench PRIVATE mc_framework benchmark::benchmark)
endif()
