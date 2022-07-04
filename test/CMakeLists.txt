include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

find_package(Torch)

# Enable testing
enable_testing()

add_executable(
  aestream_test
  main_test.cpp
  aedat_test.cpp
  torch_test.cpp
)
target_link_libraries(
  aestream_test
  gtest_main 
  aedat4
  aestream_input
  aestream_output
  ${LZ4_LIBRARY}
  ${TORCH_LIBRARIES}
)


include(GoogleTest)
gtest_discover_tests(aestream_test)