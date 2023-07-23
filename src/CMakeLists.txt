message("@ src/CMakeLists.txt")


add_subdirectory("cpp")

if (USE_PYTHON)
  message("Setting up python binds")
  # AEStream Python
  add_subdirectory("python")
endif() 