# GoogleTest requires at least C++20
set(CMAKE_CXX_STANDARD 20)

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

# Enable testing
# enable_testing()

# add_executable(
#   aestream_test
#   main_test.cpp
#   aedat_test.cpp
#   torch_test.cpp
# )
# target_include_directories(aestream_test PRIVATE ${TORCH_INCLUDE_DIRS})
# target_link_libraries(
#   aestream_test
#   aedat4
#   dvs2tensor
#   gtest_main ${LZ4_LIBRARY} ${CAER_LIBRARY} ${TORCH_PYTHON_LIBRARY}
# )
# #include_directories("../src")

# include(GoogleTest)
# gtest_discover_tests(aestream_test)