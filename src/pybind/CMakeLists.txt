include(GNUInstallDirs)
include(FetchContent)
include(CheckLanguage)
include(FindCUDAToolkit)

# Attempt to configure CUDA
if (WITH_PYTHON)
  check_language(CUDA)
  find_package(CUDAToolkit)
  set(CMAKE_CUDA_COMPILER ${CUDAToolkit_BIN_DIR}/nvcc)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 35 50 52 60 61 70 75 80 86)
  endif()

  set(module_source_files "${module_source_files}" usb.cpp)
  set(module_compile_definitions ${module_compile_definitions} tensor_buffer_kernel.cu)
endif()

if (libcaer_FOUND) 
  set(module_compile_definitions WITH_CAER)
  set(module_source_files "${module_source_files}" usb.cpp)
endif()

# find_library(MetavisionSDK NAMES metavision_sdk_core REQUIRED)

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

# add_library(aestream_cu STATIC
#   tensor_buffer_kernel.h
#   tensor_buffer_kernel.cu
# )
# target_link_libraries(aestream_cu PRIVATE CUDA::cudart CUDA::cuda_driver)
# target_include_directories(aestream_cu PUBLIC "${CUDAToolkit_INCLUDE_DIRS}")
# set_target_properties(aestream_cu PROPERTIES 
#   CUDA_SEPARABLE_COMPILATION ON
#   POSITION_INDEPENDENT_CODE ON
#   CUDA_VISIBILITY_PRESET "hidden"
# )

nanobind_add_module(aestream_ext 
      STABLE_ABI

      "${module_source_files}"
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
install(TARGETS aer aestream_file aestream_input aestream_output nanobind LIBRARY DESTINATION aestream)

set_target_properties(aestream_ext PROPERTIES 
  # BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} # For Apple
  INSTALL_RPATH "\$ORIGIN"
  # INSTALL_RPATH_USE_LINK_PATH TRUE
#   # POSITION_INDEPENDENT_CODE ON
#   # CUDA_SEPARABLE_COMPILATION ON
#   INTERPROCEDURAL_OPTIMIZATION TRUE
#   CXX_VISIBILITY_PRESET "hidden"
#   # COMPILE_FLAGS_RELEASE "-O3" # Overwrite / remove coroutine flag
)
#target_compile_definitions(aestream_ext PRIVATE ${module_compile_definitions})
#target_compile_features(aestream_ext PUBLIC cxx_std_17)
target_include_directories(aestream_ext PRIVATE "${CMAKE_INSTALL_LIBDIR}" "${PYTHON_SITE_PACKAGES_DIR}/aestream")
target_link_libraries(aestream_ext PRIVATE aer aestream_file aestream_input aestream_output nanobind)

install(TARGETS aestream_ext LIBRARY DESTINATION aestream)