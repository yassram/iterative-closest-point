cmake_minimum_required(VERSION 3.10)

project(icp)

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

set(SRCS
	src/main.cc
        src/load.cc
        src/cpu.cc
)

set(HEADERS
        src/cpu.hh
)

set(SRCS_GPU
	src/GPU/main.cc
)


set(HEADERS_GPU
	src/GPU/header.hh
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../)

add_executable(icp ${SRCS} ${HEADERS})
target_link_libraries (icp Eigen3::Eigen)
add_executable(icp-gpu ${SRCS_GPU} ${HEADERS_GPU})
