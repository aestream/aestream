include(GNUInstallDirs)
include(FetchContent)
include(FindCUDAToolkit)

# Configure CUDA
if (USE_CUDA)
  if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86)
  endif()
  find_package(CUDAToolkit REQUIRED)

  enable_language(CUDA)
  list(APPEND module_source_files "${module_source_files}" tensor_buffer_kernel.cu)
  list(APPEND module_link_libraries CUDA::cudart)
  list(APPEND module_compile_definitions USE_CUDA)
endif()

# Configure Inivation
if (WITH_CAER) 
  list(APPEND module_compile_definitions WITH_CAER)
  list(APPEND module_source_files "${module_source_files}" usb.cpp)
endif()

# Configure ZMQ
if (WITH_ZMQ)
  list(APPEND module_compile_definitions WITH_ZMQ)
  list(APPEND module_source_files "${module_source_files}" zmq.cpp)
endif()

# Configure Python extension
find_package(PythonExtensions REQUIRED)

# Thanks to https://github.com/wjakob/nanobind_example/blob/master/CMakeLists.txt
# Create CMake targets for all Python components needed by nanobind
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.26)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
else()
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
endif()

# Run `nanobind.cmake_dir()` from Python to detect where nanobind is installed
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  aestream_ext 
  NB_STATIC STABLE_ABI LTO 
  NOMINSIZE # Remove minimization to prevent -Os flags to propagate to nvcc

  "${module_source_files}" # Include e. g. camera vendors 
  module.cpp 
  udp.cpp 
  udp_client.cpp 
  udp_client.hpp
  # iterator.cpp
  file.hpp
  file.cpp
  tensor_buffer.hpp
  tensor_buffer.cpp
  tensor_iterator.hpp
  tensor_iterator.cpp
)
install(TARGETS aer aestream_file aestream_input aestream_output LIBRARY DESTINATION aestream)

if (USE_CUDA)
  set_target_properties(aestream_ext PROPERTIES
    INSTALL_RPATH "\$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # For Apple
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
else()
  set_target_properties(aestream_ext PROPERTIES
    INSTALL_RPATH "\$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )
endif()
target_compile_definitions(aestream_ext PRIVATE ${module_compile_definitions})
target_include_directories(aestream_ext PRIVATE "${CMAKE_INSTALL_LIBDIR}" "${CUDA_INCLUDE_DIRS}" "${CudaToolkitLibDir}" "${PYTHON_SITE_PACKAGES_DIR}")
target_link_libraries(aestream_ext PRIVATE ${module_link_libraries} aer aestream_file aestream_input aestream_output)

install(TARGETS aestream_ext LIBRARY DESTINATION aestream)
