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

  set(module_compile_definitions USE_CUDA)
endif()

if (libcaer_FOUND) 
  set(module_compile_definitions WITH_CAER)
  set(module_source_files "${module_source_files}" usb.cpp)
endif()

# find_library(MetavisionSDK NAMES metavision_sdk_core REQUIRED)

# Thanks to https://github.com/scikit-build/scikit-build-sample-projects/blob/master/projects/hello-pybind11/CMakeLists.txt
find_package(PythonExtensions REQUIRED)
find_package(pybind11 CONFIG)
if (NOT pybind11_FOUND)
  FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.tar.gz
    URL_HASH SHA256=6bd528c4dbe2276635dc787b6b1f2e5316cf6b49ee3e150264e455a0d68d19c1
  )
  FetchContent_MakeAvailable(pybind11)
endif()

add_library(aestream_cu STATIC
  tensor_buffer_kernel.h
  tensor_buffer_kernel.cu
)
target_link_libraries(aestream_cu PRIVATE CUDA::cudart CUDA::cuda_driver)
target_include_directories(aestream_cu PUBLIC "${CUDAToolkit_INCLUDE_DIRS}")
set_target_properties(aestream_cu PROPERTIES 
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_VISIBILITY_PRESET "hidden"
)

set(PYTHON_MODULE_NAME aestream_ext)
pybind11_add_module(${PYTHON_MODULE_NAME} MODULE
      "${module_source_files}"
      module.cpp 
      udp.cpp 
      udp_client.cpp 
      udp_client.hpp 
      iterator.cpp
      file.hpp
      file.cpp
      tensor_buffer.hpp
      tensor_buffer.cpp
      tensor_iterator.hpp
      tensor_iterator.cpp
)
install(TARGETS aer aestream_cu aestream_file aestream_input aestream_output DESTINATION aestream)

set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES 
  # BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} # For Apple
  INSTALL_RPATH "\$ORIGIN"
  # INSTALL_RPATH_USE_LINK_PATH TRUE
  # POSITION_INDEPENDENT_CODE ON
  # CUDA_SEPARABLE_COMPILATION ON
  INTERPROCEDURAL_OPTIMIZATION TRUE
  CXX_VISIBILITY_PRESET "hidden"
  # COMPILE_FLAGS_RELEASE "-O3" # Overwrite / remove coroutine flag
)
target_compile_options(${PYTHON_MODULE_NAME} PRIVATE "-ffat-lto-objects")
target_compile_definitions(${PYTHON_MODULE_NAME} PRIVATE ${module_compile_definitions})
#target_compile_features(${PYTHON_MODULE_NAME} PUBLIC cxx_std_17)
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE "${CMAKE_INSTALL_LIBDIR}" "${PYTHON_SITE_PACKAGES_DIR}/aestream")
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE aer aestream_cu aestream_file aestream_input aestream_output)

install(TARGETS ${PYTHON_MODULE_NAME} DESTINATION aestream)