cmake_minimum_required(VERSION 3.10)
project(ntl_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Aggressive optimization flags
set(CMAKE_C_FLAGS "-march=native -O3 -flto -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")

# Release configuration with full optimizations
set(CMAKE_C_FLAGS_RELEASE "-g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-g -DNDEBUG")

# Path to custom cmake files for external libs
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(GMP REQUIRED)
find_package(Threads REQUIRED)
find_package(NTL 11.5.1 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/external/include)

add_executable(ch_decoder main.cpp)

# Link-time optimization and whole program optimization
set_target_properties(ch_decoder PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

target_link_libraries(ch_decoder PRIVATE NTL GMP Threads::Threads)
