set(input_definitions "")
set(input_sources file.hpp file.cpp )
set(input_libraries aer aestream_file)
set(input_include_directories "")

include(CMakePackageConfigHelpers)
find_package(PkgConfig REQUIRED)
pkg_check_modules(libusb REQUIRED libusb-1.0)

include(FetchContent)
include(GNUInstallDirs)

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
    
    # message(DEBUG "Libcaer not found, fetching...")
    # CPMFindPackage(NAME libcaer
    #   GIT_REPOSITORY https://gitlab.com/inivation/dv/libcaer
    #   GIT_TAG 8710e73ceb21d9fb4f74ead62ca3fae42d828997
    #   SOURCE_SUBDIR build/cmake
    #   DOWNLOAD_ONLY True
    # )
    # set(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_DIR}:${libcaer_SOURCE_DIR}/cmakemod")
    # # list(APPEND CMAKE_MODULE_PATH ${libcaer_SOURCE_DIR}/cmakemod)
    # add_subdirectory(${libcaer_SOURCE_DIR} ${libcaer_BINARY_DIR})
    
    # FetchContent_MakeAvailable(libcaer)

    # message(DEBUG " Successfully downloaded and initialized libcaer")
  endif()

  # Include MetavisionSDK
  # find_package(MetavisionSDK COMPONENTS core driver QUIET)
  # if (NOT ${MetavisionSDK_FOUND})
  #   message(DEBUG "MetavisionSDK not found, fetching...")
  #   CPMFindPackage(NAME MetavisionSDK
  #     GITHUB_REPOSITORY prophesee-ai/openeb
  #     GIT_TAG 09c57ca
  #     DOWNLOAD_ONLY True
  #     OPTIONS 
  #       "COMPILE_PYTHON3_BINDINGS OFF"
  #       "COMPILE_3DVIEW OFF"
  #       "UDEV_RULES_SYSTEM_INSTALL OFF"
  #   )
  #   include(${MetavisionSDK_SOURCE_DIR}/cmake/custom_functions/add_library_version_header.cmake)
  #   # include_directories(${MetavisionSDK_SOURCE_DIR}/)
  #   add_subdirectory(${MetavisionSDK_SOURCE_DIR}/sdk/ ${MetavisionSDK_BINARY_DIR})
  #   # FetchContent_MakeAvailable(MetavisionSDK)
    
  #   find_package(MetavisionSDK COMPONENTS core driver REQUIRED)

  #   # include_directories(${MetavisionSDK_SOURCE_DIR}/cmake)
  #   # set(input_include_directories ${input_include_directories} "${MetavisionSDK_SOURCE_DIR}/sdk/modules")
  #   # set(input_libraries ${input_libraries} MetavisionSDK)
  # endif()

  # set(input_definitions ${input_definitions} WITH_METAVISION)
  # set(input_sources ${input_sources} prophesee.hpp prophesee.cpp)
  # set(input_libraries ${input_libraries} MetavisionSDK::core MetavisionSDK::driver)
  # set(input_include_directories ${input_include_directories})
else()
  message(STATUS "Camera option set to OFF")
endif()

# Create the input library
add_library(aestream_input SHARED ${input_sources})
target_link_libraries(aestream_input ${input_libraries})
target_compile_definitions(aestream_input PUBLIC ${input_definitions})
target_include_directories(aestream_input PUBLIC ${input_include_directories})