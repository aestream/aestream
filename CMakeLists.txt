cmake_minimum_required(VERSION 3.20)
project(aestream)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(GCC_COVERAGE_COMPILE_FLAGS "-fcoroutines")

set(default_build_type Release)
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

# Add includes
include_directories("include/")

# Add sources
add_subdirectory("src")

# Add tests
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory("test")
endif()