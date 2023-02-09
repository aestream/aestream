cmake_minimum_required(VERSION 3.20)

project(aestream)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

option(USE_TORCH "Build with Torch" OFF)
option(USE_PYTHON "Buid Python lib" OFF)

set(default_build_type Release)
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

# Add includes
include_directories("include/")
include(include/CPM.cmake)

# Add tests
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(USE_TORCH ON)
    add_subdirectory("test")
else()
    set(GCC_COVERAGE_COMPILE_FLAGS ${GCC_COVERAGE_COMPILE_FLAGS} "-03")
endif()


# Add sources
add_subdirectory("src")