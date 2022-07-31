# Include flatbuffers and generate headers
find_package(flatbuffers NAMES flatbuffers Flatbuffers FlatBuffers)

if (${flatbuffers_FOUND}) # Add flatbuffers from source dir (set in flake.nix)
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

FILE(GLOB SCHEMA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/*.fbs")
flatbuffers_generate_headers(
    TARGET fb_headers
    SCHEMAS ${SCHEMA_FILES}
)

# LZ4 for AEDAT encoding
find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)

# AEDAT4 processing
add_library(aedat4 SHARED aedat.hpp aedat4.hpp generator.hpp)
include_directories(aedat4 PRIVATE ${LZ4_INCLUDE_DIR})
target_include_directories(aedat4 PUBLIC ${FLATBUFFERS_SOURCE_DIR}/include/)
target_link_libraries(aedat4 PUBLIC fb_headers)
set_target_properties(aedat4 PROPERTIES LINKER_LANGUAGE CXX)

# Add subdirectories
add_subdirectory(input)
add_subdirectory(output)

# AEStream library
add_library(aestream_lib SHARED aestream.cpp)
target_include_directories(aestream_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(aestream_lib PUBLIC aedat4 aestream_input aestream_output)

# AEStream executable
add_executable(aestream aestream.cpp)
target_link_libraries(aestream PRIVATE aestream_lib)
install(IMPORTED_RUNTIME_ARTIFACTS aestream DESTINATION bin)