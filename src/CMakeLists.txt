# List iniVation devices connected to USB
add_library(list_devices SHARED device_discovery.cpp)

# Reading DVS data from USB
add_library(dvsusb SHARED usb.hpp usb.cpp)
target_link_libraries(dvsusb PRIVATE ${CAER_LIBRARY} ${OpenCV_LIBS} MetavisionSDK::core MetavisionSDK::driver opencv_highgui list_devices)

# Send DVS data to Ethernet
add_library(dvs2udp SHARED dvsToUDP.cpp)

# Send DVS data to File
add_library(dvs2file SHARED dvsToFile.cpp)

# AEDAT4 processing
add_library(aedat4 SHARED aedat4.hpp aedat4.cpp generator.hpp)
target_include_directories(aedat4 PUBLIC .)
include_directories(aedat4 PRIVATE ${LZ4_INCLUDE_DIR})
target_link_libraries(aedat4 PRIVATE ${LZ4_LIBRARY} ${CAER_LIBRARY})

# Convert DVS data to Sparse Tensor
add_library(dvs2tensor SHARED dvsToTensor.cpp convert.cpp)
target_include_directories(dvs2tensor PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_link_libraries(dvs2tensor PRIVATE ${TORCH_LIBRARIES} dvsusb)

# Stream from camera
# add_executable(stream stream.cpp)
# target_include_directories(stream PRIVATE ${TORCH_INCLUDE_DIRS})
# target_link_libraries(stream PRIVATE dvsusb aedat4 dvs2udp dvs2file ${TORCH_LIBRARIES} ${OpenCV_LIBS} MetavisionSDK::core MetavisionSDK::driver )