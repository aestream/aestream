set(output_definitions "")
set(output_sources dvs_to_file.hpp dvs_to_file.cpp dvs_to_udp.hpp dvs_to_udp.cpp)
set(output_libraries aedat4)

# Include Torch
find_package(Torch)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
if (TORCH_PYTHON_LIBRARY)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    set(output_definitions ${output_definitions} WITH_TORCH)
    set(output_sources ${output_sources} dvs_to_tensor.hpp dvs_to_tensor.cpp)
    set(output_libraries ${output_libraries} ${TORCH_LIBRARIES})
else()
    message(STATUS "Torch dependencies (libtorch) not found, ignoring")
endif()

# Create the output library
add_library(aestream_output SHARED ${output_sources})
target_link_libraries(aestream_output PRIVATE ${output_libraries})
target_compile_definitions(aestream_output PUBLIC ${output_definitions})