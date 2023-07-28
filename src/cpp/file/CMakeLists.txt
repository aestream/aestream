set(input_definitions "")
set(input_sources aedat.hpp aedat4.hpp evt3.hpp csv.hpp dat.hpp utils.hpp )
set(input_libraries aer)
set(input_include_directories "")

# Include flatbuffers and generate headers
find_package(flatbuffers CONFIG NAMES flatbuffers Flatbuffers FlatBuffers QUIET)
set(FLATBUFFERS_BUILD_TESTS OFF)
if (flatbuffers_FOUND AND FLATBUFFERS_SOURCE_DIR) # Add flatbuffers from source dir (set in flake.nix)
  add_subdirectory(${FLATBUFFERS_SOURCE_DIR} 
                  ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
                  EXCLUDE_FROM_ALL)
else()
  message(STATUS "flatbuffers not found, fetching from source")
  include(FetchContent)
  FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v23.1.21
  )
  FetchContent_MakeAvailable(flatbuffers)
endif()

set(input_libraries ${input_libraries} flatbuffers)
FILE(GLOB SCHEMA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/*.fbs")
flatbuffers_generate_headers(
    TARGET fb_headers
    SCHEMAS ${SCHEMA_FILES}
)

# LZ4 for AEDAT encoding
find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(lz4 NAMES lz4 LZ4)
if (lz4)
  set(input_libraries ${input_libraries} lz4)
else()
  set(BUILD_STATIC_LIBS ON)
  CPMFindPackage(NAME lz4 
                GITHUB_REPOSITORY lz4/lz4
                VERSION 1.9.4
                SOURCE_SUBDIR build/cmake
  )
  set(input_include_directories ${input_include_directories} "${LZ4_LIBRARY_BINARY_DIR}" "${LZ4_LIBRARY_SOURCE_DIR}/")
  set(input_libraries ${input_libraries} lz4_static)
endif()

# Create the file library
add_library(aestream_file STATIC ${input_sources})
target_link_libraries(aestream_file PUBLIC fb_headers ${input_libraries})
target_compile_definitions(aestream_file PUBLIC ${input_definitions})
target_include_directories(aestream_file PUBLIC ${input_include_directories})