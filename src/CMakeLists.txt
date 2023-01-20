include(GNUInstallDirs)
include(FetchContent)

# AER processing
add_library(aer SHARED aer.hpp generator.hpp)
target_include_directories(aer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(aer PROPERTIES LINKER_LANGUAGE CXX)

# Ensure we have proper compile definitions when importing torch
if (CMAKE_CUDA_COMPILER) 
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  set(module_compile_definitions USE_CUDA)
endif()

# Attempt to import Torch
if (USE_TORCH)
  find_package(Torch REQUIRED)
  set(module_compile_definitions USE_TORCH)
endif()

# Add subdirectories
add_subdirectory(file)
add_subdirectory(output)
add_subdirectory(input)

if (WITH_PYTHON)
  # AEStream Python
  add_subdirectory(pybind)
else() 
  # AEStream executable
  add_executable(aestream aestream.cpp)
  target_include_directories(aestream PRIVATE "${TORCH_INCLUDE_DIRS}" ${CMAKE_INSTALL_LIBDIR})
  target_link_libraries(aestream PRIVATE aer aestream_file aestream_input aestream_output)

  # Install targets
  # install(TARGETS aer aestream_input aestream_output aestream_lib LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(IMPORTED_RUNTIME_ARTIFACTS aestream DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
