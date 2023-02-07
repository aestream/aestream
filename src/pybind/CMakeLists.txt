include(GNUInstallDirs)
include(FetchContent)

if (USE_CUDA) # Include cuda buffers
  set(module_source_files tensor_buffer_kernel.cu)
endif()
# Import PyTorch's Python library to ensure correct linking
# Thanks to https://github.com/pytorch/pytorch/issues/38122
# find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
# message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY} - ${PYTHON_INCLUDE_DIRS}")
# message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES} - ${TORCH_INCLUDE_DIRS}")

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

set(PYTHON_MODULE_NAME aestream_ext)
pybind11_add_module(${PYTHON_MODULE_NAME} MODULE
      #${module_source_files}
      module.cpp 
      udp.cpp 
      udp_client.cpp 
      udp_client.hpp 
      iterator.cpp
      file.hpp
      file.cpp
      # convert.cpp 
      # convert.hpp 
      tensor_buffer.hpp
      tensor_buffer.cpp
      tensor_buffer_kernel.cu
      tensor_buffer_kernel.h
      tensor_iterator.hpp
      tensor_iterator.cpp
      )

install(IMPORTED_RUNTIME_ARTIFACTS aer aestream_file aestream_input aestream_output DESTINATION aestream)

set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES 
  BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} # For Apple
  INSTALL_RPATH "\$ORIGIN"
  INSTALL_RPATH_USE_LINK_PATH TRUE
  CUDA_SEPARABLE_COMPILATION ON
)
target_compile_definitions(${PYTHON_MODULE_NAME} PRIVATE USE_CUDA=1)
target_compile_features(${PYTHON_MODULE_NAME} PUBLIC cxx_std_20)
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE "${CUDAToolkit_INCLUDE_DIRS}" "${CUDA_LIBRARIES}" "${CMAKE_INSTALL_LIBDIR}" "${PYTHON_SITE_PACKAGES_DIR}/aestream")
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver aer aestream_file aestream_input aestream_output)

install(TARGETS ${PYTHON_MODULE_NAME} DESTINATION aestream)