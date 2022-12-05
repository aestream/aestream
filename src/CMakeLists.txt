include(GNUInstallDirs)

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

FILE(GLOB SCHEMA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/*.fbs")
flatbuffers_generate_headers(
    TARGET fb_headers
    SCHEMAS ${SCHEMA_FILES}
)

# LZ4 for AEDAT encoding
find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(LZ4_LIBRARY NAMES lz4 LZ4 REQUIRED)

# Link in Torch outside of Python
if (NOT WITH_PYTHON)
  find_package(Torch)
endif()

# AEDAT4 processing
add_library(aedat4 SHARED aedat.hpp aedat4.hpp generator.hpp)
target_include_directories(aedat4 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(aedat4 PUBLIC fb_headers flatbuffers lz4)

# Add subdirectories
add_subdirectory(output)
add_subdirectory(input)
add_subdirectory(process)

# AEStream Python
if (WITH_PYTHON)
  add_subdirectory(pybind)
endif()

# AEStream library
add_library(aestream_lib SHARED aestream.cpp)
target_include_directories(aestream_lib PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries(aestream_lib "${TORCH_LIBRARIES}" aedat4 aestream_input aestream_output aestream_process)

# AEStream executable
add_executable(aestream aestream.cpp)
target_link_libraries(aestream PRIVATE aestream_lib)

# Install targets
install(TARGETS aedat4 aestream_input aestream_output aestream_process aestream_lib LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(IMPORTED_RUNTIME_ARTIFACTS aestream DESTINATION ${CMAKE_INSTALL_BINDIR})
