project(eventbuffer LANGUAGE CXX)

set(EVENTBUFFER_SOURCES "")

# Enable CUDA, if needed
if (CMAKE_CUDA_COMPILER)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
  enable_language(CUDA)

  set(EVENTBUFFER_SOURCES tensor_buffer_cuda.cpp tensor_buffer_kernel.cu)
  # find_package(CUDAToolkit REQUIRED)
else()
  set(EVENTBUFFER_SOURCES tensor_buffer_cuda.cpp tensor_buffer_kernel.cu)
endif()

find_package(PythonExtensions REQUIRED)
find_package(Torch REQUIRED)

add_library(aestream_eventbuffer MODULE ../aedat.h tensor_buffer.hpp ${EVENTBUFFER_SOURCES})
target_link_libraries(aestream ${TORCH_LIBRARIES} aestream_lib)
target_include_directories(aestream PRIVATE ${TORCH_INCLUDE_DIRS})

# add_library(aestream MODULE 
# module.cpp udp.cpp usb.cpp udp_client.cpp udp_client.hpp convert.cpp convert.hpp tensor_buffer.hpp 
# ${module_source_files})
# python_extension_module(aestream)
# target_link_libraries(aestream ${TORCH_LIBRARIES} aestream_lib)
# target_include_directories(aestream PRIVATE ${TORCH_INCLUDE_DIRS})