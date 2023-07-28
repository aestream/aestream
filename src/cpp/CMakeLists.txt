message("@ src/cpp/CMakeLists.txt")
include(FetchContent)

# AER processing
add_library(aer STATIC aer.hpp generator.hpp)
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

# AEStream executable
set(aestream_sources "")
set(aestream_link_libraries "")
find_package(SDL2 CONFIG COMPONENTS SDL2 QUIET)
if (SDL2_FOUND)
  message(STATUS "SDL2 found")
  set(WITH_SDL ON)
  list(APPEND aestream_sources viewer/viewer.cpp viewer/viewer.hpp)
  list(APPEND aestream_link_libraries SDL2)
  add_compile_definitions(WITH_SDL)
else()
  message(STATUS "SDL2 not found, cannot compile viewer")
endif()
add_executable(aestream aestream.cpp ${aestream_sources})
target_include_directories(aestream PRIVATE ${AESTREAM_LIBDIR} ${CMAKE_INSTALL_LIBDIR})
target_link_libraries(aestream PRIVATE aer aestream_file aestream_input aestream_output ${aestream_link_libraries})
if (USE_PYTHON)
  # Thanks to https://github.com/scikit-build/scikit-build-sample-projects/
  set_target_properties(aestream PROPERTIES INSTALL_RPATH ${AESTREAM_RUNPATH})
endif()

# Install targets
if (${USE_CAMERA})
  set(aestream_drivers "")
  if (${WITH_CAER})
    list(APPEND aestream_drivers "libcaer")
  endif()
  if (${WITH_METAVISION})
    list(APPEND aestream_drivers "MetavisionSDK::core MetavisionSDK::driver")
  endif()
endif()
install(TARGETS aer aestream_file aestream_input aestream_output LIBRARY DESTINATION ${AESTREAM_LIBDIR})
install(TARGETS aestream DESTINATION ${AESTREAM_BINDIR})
