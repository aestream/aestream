# Stream DVS data via Ethernet using UDP
add_library(usbdvs SHARED usb.hpp usb.cpp)
# target_compile_features(dvs2udp PRIVATE cxx_std_14)
# target_include_directories(dvs2udp PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
# target_link_directories(dvs2udp PRIVATE ${LIBCAER_LINK_DIRECTORIES} ${TORCH_LINK_DIRECTORIES} ${CMAKE_BINARY_DIR}/res)
# target_link_libraries(dvs2udp PRIVATE ${CAER_LIBRARY} ${TORCH_LIBRARIES} ${Python3_LIBRARIES})

add_executable(stream stream.cpp)
target_include_directories(stream PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(stream PRIVATE usbdvs ${TORCH_LIBRARIES} ${CAER_LIBRARY})


# Stream DVS data and transform it into sparse tensors 
# add_library(dvs2tensor SHARED dvs2tensor.hpp dvs2tensor.cpp)
# target_compile_features(dvs2tensor PRIVATE cxx_std_14)
# target_include_directories(dvs2tensor PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
# target_link_directories(dvs2tensor PRIVATE ${LIBCAER_LINK_DIRECTORIES} ${TORCH_LINK_DIRECTORIES} ${CMAKE_BINARY_DIR}/res)
# target_link_libraries(dvs2tensor PRIVATE ${CAER_LIBRARY} ${TORCH_LIBRARIES} ${Python3_LIBRARIES})

# add_executable(stream2tensor stream2tensor.cpp)
# target_include_directories(stream2tensor PRIVATE ${TORCH_INCLUDE_DIRS})
# target_link_libraries(stream2tensor PRIVATE dvs2tensor ${TORCH_LIBRARIES} ${CAER_LIBRARY})