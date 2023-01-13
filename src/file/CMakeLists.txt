set(input_definitions "")
set(input_sources aedat.hpp aedat4.hpp utils.hpp utils.cpp dat.hpp dat.cpp )
set(input_libraries aer)
set(input_include_directories "")

# Include flatbuffers and generate headers
find_package(flatbuffers CONFIG NAMES flatbuffers Flatbuffers FlatBuffers QUIET)
option(FLATBUFFERS_BUILD_TESTS OFF)
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
    GIT_TAG v2.0.6
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
find_library(LZ4_LIBRARY NAMES lz4 LZ4)
# if (NOT LZ4_LIBRARY)
# FetchContent_Declare(LZ4_LIBRARY
#   GIT_REPOSITORY "https://github.com/lz4/lz4.git"
#   GIT_TAG "v1.9.4"
#   SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lz4"
# )
# FetchContent_MakeAvailable(LZ4_LIBRARY)
# message(WARNING SRC ${LZ4_LIBRARY_SOURCE_DIR})
# endif()
set(input_libraries ${input_libraries} ${LZ4_LIBRARY})

# Create the file library
add_library(aestream_file SHARED ${input_sources})
target_link_libraries(aestream_file PUBLIC fb_headers ${input_libraries})
target_compile_definitions(aestream_file PUBLIC ${input_definitions})
target_include_directories(aestream_file PUBLIC ${input_include_directories})