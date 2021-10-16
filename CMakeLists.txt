cmake_minimum_required(VERSION 3.7)
project(aedat)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP ON)
set(GCC_COVERAGE_COMPILE_FLAGS "-fcoroutines")

include_directories("include/")

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

# Make sure libcaer can be found
find_library(CAER_LIBRARY NAMES caer)

# Include MetavisionSDK
find_package(MetavisionSDK COMPONENTS core driver REQUIRED)
find_package(OpenCV COMPONENTS highgui REQUIRED)

# Include OpenCV
find_package (OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

# LZ4 for AEDAT encoding
find_path(LZ4_INCLUDE_DIR NAMES	lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)
find_package (Python3 REQUIRED COMPONENTS Development)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add subdirectories
add_subdirectory("src")
add_subdirectory("test")