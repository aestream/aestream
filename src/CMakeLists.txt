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

# AEDAT4 processing
add_library(aedat4 SHARED aedat.hpp aedat4.hpp generator.hpp)
target_include_directories(aedat4 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(aedat4 PUBLIC fb_headers flatbuffers lz4)

# Add subdirectories
add_subdirectory(output)
add_subdirectory(input)

# AEStream Python
if (WITH_PYTHON)
  if (CMAKE_CUDA_COMPILER)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    set(module_source_files pybind/tensor_buffer_cuda.cpp pybind/tensor_buffer_kernel.cu)
  else()
    set(module_source_files pybind/tensor_buffer.cpp)
  endif()

  # Require input libraries
  find_library(libcaer NAMES caer libcaer REQUIRED)
  # find_library(MetavisionSDK NAMES metavision_sdk_core REQUIRED)

  find_package(PythonExtensions REQUIRED)
  find_package(Torch REQUIRED)
  
  # Import PyTorch's Python library to ensure correct linking
  # Thanks to https://github.com/pytorch/pytorch/issues/38122
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
  message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

  set(python_module aestream)
  add_library(${python_module} MODULE
        pybind/module.cpp 
        pybind/udp.cpp 
        pybind/usb.cpp 
        pybind/udp_client.cpp 
        pybind/udp_client.hpp 
        pybind/convert.cpp 
        pybind/convert.hpp 
        pybind/tensor_buffer.hpp 
        ${module_source_files})
  target_link_libraries(${python_module} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} aedat4 aestream_input aestream_output)
  target_include_directories(${python_module} PUBLIC ${TORCH_INCLUDE_DIRS} ${CMAKE_INSTALL_LIBDIR} ${PYTHON_INCLUDE_DIR})
  target_compile_features(${python_module} PUBLIC cxx_std_20)
  set_target_properties(${python_module} 
     PROPERTIES CXX_STANDARD 20
     INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}"
    )

  #python_extension_module(aedat4)
  python_extension_module(aestream_input)
  python_extension_module(aestream_output)
  python_extension_module(libcaer)
  python_extension_module(${python_module})

  install(TARGETS aedat4 aestream_input aestream_output DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(TARGETS ${python_module} DESTINATION .)
else()

  # AEStream library
  add_library(aestream_lib SHARED aestream.cpp)
  target_include_directories(aestream_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
  target_link_libraries(aestream_lib PUBLIC aedat4 aestream_input aestream_output)

  # AEStream executable
  add_executable(aestream aestream.cpp)
  target_link_libraries(aestream PRIVATE aestream_lib)

  # Install targets
  install(TARGETS aestream_lib LIBRARY)
  install(IMPORTED_RUNTIME_ARTIFACTS aestream DESTINATION bin)
endif()
