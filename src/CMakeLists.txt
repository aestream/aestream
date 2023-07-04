include(GNUInstallDirs)
include(FetchContent)

# AER processing
add_library(aer SHARED aer.hpp generator.hpp)
target_include_directories(aer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(aer PROPERTIES LINKER_LANGUAGE CXX)
# set coroutine flags for clang appropriately
# thanks to https://stackoverflow.com/questions/64703866/is-clang-11s-invocation-of-coroutine-headers-via-fcoroutines-flag
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(aer PUBLIC "-fcoroutines-ts")
  add_compile_definitions(USE_CLANG)
  message(STATUS "detected clang, set -ts flag")
else()
  target_compile_options(aer PUBLIC "-fcoroutines")
endif()

# Add subdirectories
add_subdirectory(file)
add_subdirectory(output)
add_subdirectory(input)

if (USE_PYTHON)
  # AEStream Python
  add_subdirectory(pybind)
endif() 

# AEStream executable
find_package(SDL2 REQUIRED CONFIG COMPONENTS SDL2)
add_executable(aestream aestream.cpp viewer/viewer.hpp viewer/viewer.cpp)
target_include_directories(aestream PRIVATE ${CMAKE_INSTALL_LIBDIR})
target_link_libraries(aestream PRIVATE aer aestream_file aestream_input aestream_output SDL2::SDL2)

# Install targets
if (${USE_CAMERA})
  set(aestream_drivers "")
  if (${WITH_CAER})
    list(APPEND aestream_drivers "libcaer")
  endif()
  if (${WITH_METAVISION})
    list(APPEND aestream_drivers "MetavisionSDK::core MetavisionSDK::driver")
  endif()
  # install(TARGETS "${aestream_drivers}" DESTINATION ${CMAKE_INSTALL_BINDIR})
  # install(TARGETS MetavisionSDK::core MetavisionSDK::driver DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(TARGETS aer aestream_file aestream_input aestream_output DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS aestream DESTINATION ${CMAKE_INSTALL_BINDIR})
