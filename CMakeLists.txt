cmake_minimum_required(VERSION 3.10)

project(icp LANGUAGES CUDA CXX)

# Include Eigen from gihub
include(FetchContent)
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.tar.gz
    )
FetchContent_MakeAvailable(eigen)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-Os")

cmake_policy(SET CMP0104 OLD)

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
    src/load.hh
)


set(HEADERS_GPU
    src/load.cc
    src/GPU/compute.hh
    src/GPU/gpu.hh
)

set(SRCS_GPU_CUDA
    src/GPU/compute.cu
)


set(HEADERS_GPU_CUDA
    src/GPU/compute.hh
    src/GPU/gpu.hh
)


add_library(compute)
target_compile_features(compute PUBLIC cxx_std_14)
target_sources(compute PRIVATE ${SRCS_GPU} ${HEADERS_GPU})
set_target_properties(compute PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(compute Eigen3::Eigen)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../)

# CPU
#add_executable(icp ${SRCS} ${HEADERS})
#target_link_libraries(icp Eigen3::Eigen)

# GPU
add_executable(icp-gpu ${SRCS_GPU_CUDA} ${HEADERS_GPU_CUDA})
target_link_libraries(icp-gpu PRIVATE compute Eigen3::Eigen)
