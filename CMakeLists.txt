cmake_minimum_required(VERSION 3.20)
project(aestream)
message("@ CMakeLists.txt")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Build with python lib
option(USE_PYTHON "Buid Python lib" OFF)
# CUDA support
option(USE_CUDA "Build CUDA tensor export, if the library is found" ON)

# iNiVation camera support (And other?)
find_package(libcaer CONFIG QUIET)
option(USE_CAMERA "Build with camera drivers?" ${libcaer_FOUND})

# Compile flags
set(default_build_type Release)
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

# Includes
include_directories("include/")
include_directories("/opt/homebrew/opt/lz4/include") # Compression
link_directories("/opt/homebrew/opt/lz4/lib")
include_directories("/opt/local/include")
link_directories("/opt/local/lib")
include_directories("/opt/homebrew/opt/sdl2/include/") # Video rendering?
link_directories("/opt/homebrew/opt/sdl2/lib")
include(include/CPM.cmake)

# Add tests
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory("test")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
else()
    set(GCC_COVERAGE_COMPILE_FLAGS ${GCC_COVERAGE_COMPILE_FLAGS} "-03")
endif()


# Add sources
add_subdirectory("src")
