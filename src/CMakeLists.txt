include(GNUInstallDirs)
include(FetchContent)

# Add subdirectories
add_subdirectory(output)
add_subdirectory(input)

if (USE_PYTHON)
  # AEStream Python
  add_subdirectory(pybind)
endif() 

# AEStream executable
add_executable(aestream aestream.cpp)
target_include_directories(aestream PRIVATE ${CMAKE_INSTALL_LIBDIR})
target_link_libraries(aestream PRIVATE aer aestream_file aestream_input aestream_output)

# Install targets
install(IMPORTED_RUNTIME_ARTIFACTS aestream DESTINATION ${CMAKE_INSTALL_BINDIR})
