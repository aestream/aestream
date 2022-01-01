# LZ4 for AEDAT encoding
find_path(LZ4_INCLUDE_DIR NAMES	lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)
if (NOT LZ4_LIBRARY)
    message(FATAL_ERROR "LZ4 required but not installed")
endif()

# AEDAT4 processing
add_library(aedat4 SHARED aedat4.hpp aedat4.cpp generator.hpp)
target_include_directories(aedat4 PUBLIC .)
include_directories(aedat4 PRIVATE ${LZ4_INCLUDE_DIR})
target_link_libraries(aedat4 PRIVATE ${LZ4_LIBRARY})

# Add subdirectories
add_subdirectory("input")
add_subdirectory("output")

# Search for Torch
find_package(Torch REQUIRED)

# For linking against python
# find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
# list(APPEND TORCH_LIBRARIES ${TORCH_PYTHON_LIBRARY})

# Add compilation definitions
# add_definitions( -DVARNAME=FOUND_TORCH )
# message(STATUS "Found Torch, building torch outputs")
# # endif()


# # find_package(OpenCV COMPONENTS highgui)

# # Include OpenCV
# # find_package (OpenCV REQUIRED)
# # include_directories(${OpenCV_DIRS})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# # Send DVS data to Ethernet
# add_library(dvs2udp SHARED dvsToUDP.cpp)

# # Send DVS data to File
# add_library(dvs2file SHARED dvsToFile.cpp)

# # Convert DVS data to Sparse Tensor
# add_library(dvs2tensor SHARED dvsToTensor.cpp convert.cpp)
# target_include_directories(dvs2tensor PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
# target_link_libraries(dvs2tensor PRIVATE ${TORCH_LIBRARIES} dvsusb)

# AEStream executable
add_executable(aestream aestream.cpp)
target_include_directories(aestream PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(aestream PRIVATE aedat4 aestream_input)
# MetavisionSDK::core MetavisionSDK::driver )
