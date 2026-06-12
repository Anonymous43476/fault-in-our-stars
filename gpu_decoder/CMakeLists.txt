cmake_minimum_required(VERSION 3.18)
project(cuda_decoder LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 61)

# Release config
set(CMAKE_CXX_FLAGS_RELEASE "-g -DNDEBUG")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(GMP REQUIRED)
find_package(Threads REQUIRED)
find_package(NTL 11.5.1 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/external/include)

add_executable(ch_decoder main.cu ntl_utils.cpp array3d.cpp)


target_link_libraries(ch_decoder PRIVATE NTL GMP Threads::Threads )
