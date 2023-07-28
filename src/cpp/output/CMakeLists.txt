set(output_definitions "")
set(output_sources dvs_to_udp.hpp dvs_to_udp.cpp dvs_to_file.hpp dvs_to_file.cpp)
set(output_libraries aer aestream_file)

# Create the output library
add_library(aestream_output STATIC ${output_sources})
target_compile_definitions(aestream_output PUBLIC ${output_definitions})
target_compile_features(aestream_output PUBLIC cxx_std_20)
target_include_directories(aestream_output PRIVATE "${TORCH_INCLUDE_DIRS}" ${CMAKE_INSTALL_LIBDIR})
target_link_libraries(aestream_output ${output_libraries})
