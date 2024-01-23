message("@ src/pybind/CMakeLists.txt")

include(GNUInstallDirs)
include(FetchContent)
include(FindCUDAToolkit)

# Configure CUDA
find_package(CUDAToolkit QUIET)
if (USE_CUDA AND CUDAToolkit_FOUND)
  if(NOT DEFINED ${CMAKE_CUDA_COMPILER})
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
  endif()
  message("CUDA toolkit found at " ${CMAKE_CUDA_COMPILER})
  if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86)
  endif()
  find_package(CUDAToolkit REQUIRED)

  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  list(APPEND module_source_files "${module_source_files}" tensor_buffer_kernel.cu)
  list(APPEND module_link_libraries CUDA::cudart)
  list(APPEND module_compile_definitions USE_CUDA)
elseif(NOT USE_CUDA)
  message("CUDA support disabled; CUDA extensions won't be built")
else()
  message("CUDA toolkit could not be found; CUDA extensions won't be built")
endif()

# Configure Inivation cameras
if (WITH_CAER OR WITH_METAVISION) 
  if (WITH_CAER) 
    list(APPEND module_compile_definitions WITH_CAER)
  endif()
  if (WITH_METAVISION) 
    list(APPEND module_compile_definitions WITH_METAVISION)
  endif()
  list(APPEND module_source_files "${module_source_files}" usb.cpp)
endif()

# Configure ZMQ for Speck
if (WITH_ZMQ)
  list(APPEND module_compile_definitions WITH_ZMQ)
  list(APPEND module_source_files "${module_source_files}" zmq.cpp)
endif()

# Import Python dependencies
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)


find_package(nanobind CONFIG REQUIRED)

# Add nanobind module
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
install(TARGETS aer aestream_file aestream_input aestream_output RUNTIME_DEPENDENCIES DESTINATION aestream)

# Setup install paths
if (USE_CUDA)
  set_target_properties(aestream_ext PROPERTIES
    INSTALL_RPATH "\$ORIGIN"
    BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # For Apple
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
else()
  set_target_properties(aestream_ext PROPERTIES
    INSTALL_RPATH "\$ORIGIN"
  )
endif()
target_compile_definitions(aestream_ext PRIVATE ${module_compile_definitions})
target_include_directories(aestream_ext PRIVATE "${CMAKE_INSTALL_LIBDIR}" "${CUDA_INCLUDE_DIRS}" "${CudaToolkitLibDir}" "${Python_SITEDIR}")
target_link_libraries(aestream_ext PRIVATE ${module_link_libraries} aer aestream_file aestream_input aestream_output)

message("Installing aestream_ext to aestream")
install(TARGETS aestream_ext LIBRARY DESTINATION aestream)
