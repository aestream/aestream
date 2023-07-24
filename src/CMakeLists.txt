message("@ src/CMakeLists.txt")

# AEStream lib
add_subdirectory("cpp")

# AEStream python bindings
if (USE_PYTHON)
  message("Setting up python binds")
  add_subdirectory("python")
endif() 