cmake_minimum_required(VERSION 3.10)
project(icp LANGUAGES CUDA CXX)

# -------------------- benchmark -----------------------

include(FetchContent)
FetchContent_Declare(
googlebenchmark
GIT_REPOSITORY https://github.com/google/benchmark.git
GIT_TAG origin/master
)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "FromGtest")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "FromGtest")
FetchContent_MakeAvailable(googlebenchmark)

# -------------------- Eigen -----------------------

add_subdirectory(lib)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# -------------------- Flags -----------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-Os")

# -------------------- Policies -----------------------

find_package(CUDA REQUIRED)
cmake_policy(SET CMP0104 OLD)

# -------------------- CPU -----------------------

set(SRCS
src/main.cc
src/load.cc
src/cpu.cc
)

set(HEADERS
src/cpu.hh
)


# -------------------- GPU -----------------------


set(SRCS_GPU
src/GPU/main.cc
src/GPU/gpu.cc
src/load.cc
)


set(HEADERS_GPU
src/load.hh
src/GPU/gpu.hh
)


# -------------------- GPU Compute -----------------------

set(SRCS_COMPUTE
src/GPU/compute.cu
)

set(HEADERS_COMPUTE
src/GPU/gpu.hh
)


add_library(compute)
target_compile_features(compute PUBLIC cxx_std_14)
target_sources(compute PRIVATE ${SRCS_GPU} ${HEADERS_GPU})
set_target_properties(compute PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(compute Eigen3::Eigen)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../)

add_executable(icp ${SRCS} ${HEADERS})
add_executable(icp-gpu ${SRCS_COMPUTE} ${HEADERS_COMPUTE})
add_executable(bench src/bench.cc src/load.cc src/cpu.cc src/GPU/compute.cu src/GPU/gpu.cc ${HEADERS_GPU} ${HEADERS})

target_link_libraries(icp Eigen3::Eigen)
target_link_libraries(icp-gpu compute)
target_link_libraries(bench Eigen3::Eigen benchmark::benchmark)
