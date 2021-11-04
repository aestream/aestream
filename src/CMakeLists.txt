
# List iniVation devices connected to USB
add_library(list_devices SHARED device_discovery.cpp)

# Reading DVS data from USB
add_library(dvsusb SHARED usb.hpp usb.cpp)
target_link_libraries(dvsusb PRIVATE ${CAER_LIBRARY} ${OpenCV_LIBS} MetavisionSDK::core MetavisionSDK::driver opencv_highgui list_devices)

# Send DVS data to Ethernet
add_library(dvs2udp SHARED dvs2udp.cpp)

# AEDAT4 processing
add_library(aedat4 SHARED aedat4.hpp aedat4.cpp)
target_include_directories(aedat4 PUBLIC .)
include_directories(aedat4 PRIVATE ${LZ4_INCLUDE_DIR})
target_link_libraries(aedat4 PRIVATE ${LZ4_LIBRARY} ${CAER_LIBRARY})

add_executable(stream stream.cpp)
target_include_directories(stream PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(stream PRIVATE dvsusb aedat4 dvs2udp ${TORCH_LIBRARIES} ${OpenCV_LIBS})


# Stream DVS data and transform it into sparse tensors 
# add_library(dvs2tensor SHARED dvs2tensor.hpp dvs2tensor.cpp)
# target_compile_features(dvs2tensor PRIVATE cxx_std_14)
# target_include_directories(dvs2tensor PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
# target_link_directories(dvs2tensor PRIVATE ${LIBCAER_LINK_DIRECTORIES} ${TORCH_LINK_DIRECTORIES} ${CMAKE_BINARY_DIR}/res)
# target_link_libraries(dvs2tensor PRIVATE ${CAER_LIBRARY} ${TORCH_LIBRARIES} ${Python3_LIBRARIES})

# add_executable(stream2tensor stream2tensor.cpp)
# target_include_directories(stream2tensor PRIVATE ${TORCH_INCLUDE_DIRS})
# target_link_libraries(stream2tensor PRIVATE dvs2tensor ${TORCH_LIBRARIES} ${CAER_LIBRARY})
