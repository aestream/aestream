# Setup googletest
find_library(googletest NAMES GoogleTest googletest gtest REQUIRED)
if (NOT ${googletest_FOUND}) # Manually download if not found
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
  )
endif()

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

# Enable testing
enable_testing()

add_executable(
  aestream_test
  main_test.cpp
  file_test.cpp
)
target_link_libraries(
  aestream_test
  aer
  aestream_input
  aestream_output
  gtest
  ${LZ4_LIBRARY}
)
