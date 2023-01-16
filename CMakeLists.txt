cmake_minimum_required(VERSION 3.20)
project(aestream)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# set coroutine flags for clang appropriately
# thanks to https://stackoverflow.com/questions/64703866/is-clang-11s-invocation-of-coroutine-headers-via-fcoroutines-flag
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(GCC_COVERAGE_COMPILE_FLAGS "-fcoroutines-ts")
    add_compile_definitions(USE_CLANG)
    message(STATUS "detected clang, set -ts flag")
else()
    set(GCC_COVERAGE_COMPILE_FLAGS "-fcoroutines")
endif()

set(default_build_type Release)
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

# Add includes
include_directories("include/")
include(include/CPM.cmake)

# Add sources
add_subdirectory("src")

# Add tests
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory("test")
else()
    set(GCC_COVERAGE_COMPILE_FLAGS ${GCC_COVERAGE_COMPILE_FLAGS} "-03")
endif()
