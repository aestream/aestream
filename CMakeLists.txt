cmake_minimum_required(VERSION 3.20)
project(aestream)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(GCC_COVERAGE_COMPILE_FLAGS "-fcoroutines")

include_directories("include/")

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

# Require Torch 
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

# For linking against python
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
list(APPEND TORCH_LIBRARIES ${TORCH_PYTHON_LIBRARY})

# Include libcaer and opencv
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