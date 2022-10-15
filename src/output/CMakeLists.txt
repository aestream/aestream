set(output_definitions "")
set(output_sources dvs_to_file.hpp dvs_to_file.cpp dvs_to_udp.hpp dvs_to_udp.cpp)
set(output_libraries aedat4)

# Include Torch
if (CMAKE_CUDA_COMPILER) # Ensure we have proper compile definitions
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif()
find_package(Torch)
if (TORCH_FOUND)
    set(output_definitions ${output_definitions} WITH_TORCH)
    set(output_sources ${output_sources} dvs_to_tensor.hpp dvs_to_tensor.cpp)
    set(output_libraries ${output_libraries} "${TORCH_LIBRARIES}")
else()
    message(STATUS "Torch dependencies (libtorch) not found, ignoring")
endif()

# Create the output library
add_library(aestream_output STATIC ${output_sources})
target_link_libraries(aestream_output ${output_libraries})
target_compile_definitions(aestream_output PUBLIC ${output_definitions})
target_include_directories(aestream_output PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(aestream_output PUBLIC cxx_std_20)
set_target_properties(aestream_output PROPERTIES POSITION_INDEPENDENT_CODE ON)