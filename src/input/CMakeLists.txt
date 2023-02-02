set(input_definitions "")
set(input_sources file.hpp file.cpp )
set(input_libraries aer aestream_file)
set(input_include_directories "")

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
find_package(libcaer CONFIG)
if (${libcaer_FOUND})
  set(input_definitions ${input_definitions} WITH_CAER)
  set(input_sources ${input_sources} inivation.hpp inivation.cpp )
  set(input_libraries ${input_libraries} libcaer::caer)
  message(STATUS "Inivation dependencies (libcaer) found at ${libcaer}")
else()
  message(STATUS "Inivation dependencies (libcaer) not found, ignoring")
endif()

# Include MetavisionSDK
# find_package(MetavisionSDK COMPONENTS core driver)
# if (${MetavisionSDK_FOUND})
#     set(input_definitions ${input_definitions} WITH_METAVISION )
#     set(input_sources ${input_sources} prophesee.hpp prophesee.cpp)
#     set(input_libraries ${input_libraries} MetavisionSDK::core MetavisionSDK::driver)
#     set(input_include_directories ${input_include_directories})
# else()
#     message(STATUS "Prophesee dependencies (MetavisionSDK) not found, ignoring")
# endif()

# Create the input library
add_library(aestream_input SHARED ${input_sources})
target_link_libraries(aestream_input ${input_libraries})
target_compile_definitions(aestream_input PUBLIC ${input_definitions})
target_include_directories(aestream_input PUBLIC ${input_include_directories})