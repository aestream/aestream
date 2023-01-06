include(GNUInstallDirs)

# Include flatbuffers and generate headers
# find_package(flatbuffers CONFIG NAMES flatbuffers Flatbuffers FlatBuffers QUIET)
# option(FLATBUFFERS_BUILD_TESTS OFF)
# if (flatbuffers_FOUND AND FLATBUFFERS_SOURCE_DIR) # Add flatbuffers from source dir (set in flake.nix)
#   add_subdirectory(${FLATBUFFERS_SOURCE_DIR} 
#                   ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
#                   EXCLUDE_FROM_ALL)
# else()
#   message(STATUS "flatbuffers not found, fetching from source")
#   include(FetchContent)
#   FetchContent_Declare(
#     flatbuffers
#     GIT_REPOSITORY https://github.com/google/flatbuffers.git
#     GIT_TAG v2.0.6
#   )
#   FetchContent_MakeAvailable(flatbuffers)
# endif()

# FILE(GLOB SCHEMA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/*.fbs")
# flatbuffers_generate_headers(
#     TARGET fb_headers
#     SCHEMAS ${SCHEMA_FILES}
# )

# LZ4 for AEDAT encoding
# find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
# find_library(LZ4_LIBRARY NAMES lz4 LZ4 REQUIRED)

# set compile definition for clang
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(module_compile_definitions USE_CLANG)
endif()

# AEDAT4 processing
add_library(aedat4 SHARED aer.hpp aedat.hpp generator.hpp) #aedat4.hpp )
target_include_directories(aedat4 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(aedat4 PROPERTIES LINKER_LANGUAGE CXX)
# target_link_libraries(aedat4 PUBLIC fb_headers flatbuffers lz4)

# Ensure we have proper compile definitions when importing torch
if (CMAKE_CUDA_COMPILER) 
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  set(module_compile_definitions USE_CUDA)
endif()

# Attempt to import Torch
find_package(Torch)
if (Torch_FOUND) 
    message(STATUS "Torch dependencies (libtorch) not found, ignoring")
    set(module_compile_definitions USE_TORCH)
endif()

# Add subdirectories
add_subdirectory(output)
add_subdirectory(input)

if (WITH_PYTHON)
  # AEStream Python
  add_subdirectory(pybind)
else() 
  # AEStream C++
  # add_library(aestream_lib SHARED aestream.cpp)
  # target_include_directories(aestream_lib PRIVATE "${TORCH_INCLUDE_DIRS}")
  # target_link_libraries(aestream_lib aedat4 aestream_input aestream_output)

  # AEStream executable
  add_executable(aestream aestream.cpp)
  target_include_directories(aestream PRIVATE "${TORCH_INCLUDE_DIRS}" ${CMAKE_INSTALL_LIBDIR})
  target_link_libraries(aestream PRIVATE aedat4 aestream_input aestream_output)

  # Install targets
  # install(TARGETS aedat4 aestream_input aestream_output aestream_lib LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(IMPORTED_RUNTIME_ARTIFACTS aestream DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
