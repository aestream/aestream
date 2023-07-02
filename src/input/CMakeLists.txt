set(input_definitions "")
set(input_sources file.hpp file.cpp zmq.hpp zmq.cpp)
set(input_libraries aer aestream_file)
set(input_include_directories "")

include(FetchContent)
include(GNUInstallDirs)

find_package(cppzmq)

if (${USE_CAMERA})
  message(DEBUG "Camera option set to ON, building...")

  # Include libcaer
  find_package(libcaer CONFIG QUIET)
  if (${libcaer_FOUND})
    set(WITH_CAER ON PARENT_SCOPE)
    set(input_definitions ${input_definitions} WITH_CAER)
    set(input_sources ${input_sources} inivation.hpp inivation.cpp )
    set(input_libraries ${input_libraries} libcaer::caer)
    message(STATUS "Inivation dependencies (libcaer) found at ${libcaer_DIR}")
  endif()

  # Include MetavisionSDK
  find_package(MetavisionSDK COMPONENTS core driver QUIET)
  if (${MetavisionSDK_FOUND})
    set(input_definitions ${input_definitions} WITH_METAVISION)
    set(input_sources ${input_sources} prophesee.hpp prophesee.cpp)
    set(input_libraries ${input_libraries} MetavisionSDK::core MetavisionSDK::driver)
    set(input_include_directories ${input_include_directories})
  endif()
else()
  message(STATUS "Camera option set to OFF")
endif()

# Create the input library
add_library(aestream_input SHARED ${input_sources})
target_link_libraries(aestream_input ${input_libraries} cppzmq)
target_compile_definitions(aestream_input PUBLIC ${input_definitions})
target_include_directories(aestream_input PUBLIC ${input_include_directories})