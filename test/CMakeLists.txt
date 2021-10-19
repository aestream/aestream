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
enable_testing()

add_executable(
  aedat_test
  aedat_test.cpp
)
target_link_libraries(
  aedat_test
  aedat4
  gtest_main ${LZ4_LIBRARY} ${CAER_LIBRARY}
  
)
#include_directories("../src")

include(GoogleTest)
gtest_discover_tests(aedat_test)