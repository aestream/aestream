message("@ src/CMakeLists.txt")

include(GNUInstallDirs)

# Define install path
if (DEFINED SKBUILD)
  find_package(Python REQUIRED)
  set(AESTREAM_BINDIR ${SKBUILD_SCRIPTS_DIR})
else()
  set(AESTREAM_BINDIR ${CMAKE_INSTALL_BINDIR})
endif()
set(AESTREAM_LIBDIR ${CMAKE_INSTALL_LIBDIR})

# Add CPP files
add_subdirectory("cpp")

# Add Python files
if (USE_PYTHON)
  message("Setting up python binds")
  add_subdirectory("python")
endif() 