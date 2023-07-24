message("@ src/CMakeLists.txt")

# Define install path
if (DEFINED SKBUILD)
  find_package(Python REQUIRED)
  set(AESTREAM_BINDIR ${SKBUILD_SCRIPTS_DIR})
  set(AESTREAM_LIBDIR ${Python_SITELIB}/aestream)
else()
  set(AESTREAM_BINDIR ${CMAKE_INSTALL_BINDIR})
  set(AESTREAM_LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()

# Add CPP files
add_subdirectory("cpp")

# Add Python files
if (USE_PYTHON)
  message("Setting up python binds")
  add_subdirectory("python")
endif() 