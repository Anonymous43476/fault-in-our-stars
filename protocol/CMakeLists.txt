cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(secure_aggregation VERSION 0.1)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto -ffast-math -DNDEBUG")

# Path to custom cmake files for external libs
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(THREADS_PREFER_PTHREAD_FLAG ON)

# Find external dependencies
find_package(GMP REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost 1.83.0 COMPONENTS unit_test_framework REQUIRED)
find_package(NTL 11.5.1 REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(benchmark REQUIRED)

add_subdirectory(src)
add_subdirectory(bench)

enable_testing()
add_subdirectory(test)
