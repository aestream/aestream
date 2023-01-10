set(input_definitions "")
set(input_sources file_aedat.hpp file_aedat4.hpp file_utils.hpp file_utils.cpp file_dat.hpp file_dat.cpp file.hpp file.cpp )
set(input_libraries aer ${LZ4_LIBRARY} flatbuffers)
set(input_include_directories "")

# Include flatbuffers and generate headers
find_package(flatbuffers CONFIG NAMES flatbuffers Flatbuffers FlatBuffers QUIET)
option(FLATBUFFERS_BUILD_TESTS OFF)
if (flatbuffers_FOUND AND FLATBUFFERS_SOURCE_DIR) # Add flatbuffers from source dir (set in flake.nix)
  add_subdirectory(${FLATBUFFERS_SOURCE_DIR} 
                  ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
                  EXCLUDE_FROM_ALL)
else()
  message(STATUS "flatbuffers not found, fetching from source")
  include(FetchContent)
  FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v2.0.6
  )
  FetchContent_MakeAvailable(flatbuffers)
endif()

FILE(GLOB SCHEMA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/*.fbs")
flatbuffers_generate_headers(
    TARGET fb_headers
    SCHEMAS ${SCHEMA_FILES}
)

# LZ4 for AEDAT encoding
find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(LZ4_LIBRARY NAMES lz4 LZ4)
if (NOT LZ4_LIBRARY)
  message(WARNING "LZ4 NOT FOUND")
  FetchContent_Declare(LZ4
    GIT_REPOSITORY "https://github.com/lz4/lz4.git"
    GIT_TAG "v1.9.4"
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lz4"
  )
  FetchContent_MakeAvailable(LZ4)
endif()

# find_package(PkgConfig REQUIRED)
#     pkg_check_modules(libusb REQUIRED libusb-1.0)
#     # pkg_check_modules(lz4 REQUIRED)
#   include(FetchContent)
#   include(GNUInstallDirs)
  
#   find_package(libcaer QUIET)
#   if (NOT ${libcaer_FOUND})
#     message(DEBUG "Libcaer not found, fetching...")
#     # Include libcaer
#     FetchContent_Declare(
#       libcaer
#       GIT_REPOSITORY https://gitlab.com/inivation/dv/libcaer.git
#       GIT_TAG        417052fa9a0c070a067e747c163bf99135536028 # 3.14
#     )
#     FetchContent_Populate(libcaer)
#     FetchContent_GetProperties(libcaer)
#     SET(CMAKE_MODULE_PATH ${libcaer_SOURCE_DIR}/cmakemod/ ${CMAKE_MODULE_PATH})
#     #set(CMAKE_INSTALL_INCLUDEDIR ${libcaer_SOURCE_DIR}/include)
#     #set(libcaer_INCLUDE_DIRS ${libcaer_SOURCE_DIR}/include)
#     # include_directories(${libcaer_SOURCE_DIR}/cmakemod/)
#     add_subdirectory(${libcaer_SOURCE_DIR} ${libcaer_BINARY_DIR})
#     SET(CMAKE_PREFIX_PATH ${libcaer_BINARY_DIR} ${CMAKE_PREFIX_PATH})
#     include_directories(${libcaer_BINARY_DIR})
#     FetchContent_MakeAvailable(libcaer)
#     message(WARNING ${libcaer_BINARY_DIR})
#     find_package(libcaer REQUIRED)
#     #include(${libcaer_BINARY_DIR})
#     message(DEBUG " Successfully downloaded and initialized libcaer")
#   endif()
  
#   set(input_definitions ${input_definitions} WITH_CAER )
#   set(input_sources ${input_sources} inivation.hpp inivation.cpp )
#   set(input_libraries ${input_libraries} libcaer)

# Include libcaer
find_library(libcaer NAMES caer libcaer)
if (libcaer_FOUND)
  set(input_definitions ${input_definitions} WITH_CAER )
  set(input_sources ${input_sources} inivation.hpp inivation.cpp )
  set(input_libraries ${input_libraries} ${libcaer})
  message(STATUS "Inivation dependencies (libcaer) found at ${libcaer}")
else()
  message(STATUS "Inivation dependencies (libcaer) not found, ignoring")
endif()

# Include MetavisionSDK
find_package(MetavisionSDK COMPONENTS core driver)
if (MetavisionSDK_FOUND)
    set(input_definitions ${input_definitions} WITH_METAVISION )
    set(input_sources ${input_sources} prophesee.hpp prophesee.cpp)
    set(input_libraries ${input_libraries} MetavisionSDK::core MetavisionSDK::driver)
    set(input_include_directories ${input_include_directories})
else()
    message(STATUS "Prophesee dependencies (MetavisionSDK) not found, ignoring")
endif()

# Create the input library
add_library(aestream_input SHARED ${input_sources})
target_link_libraries(aestream_input ${input_libraries})
target_compile_definitions(aestream_input PUBLIC ${input_definitions})
target_include_directories(aestream_input PUBLIC ${input_include_directories})