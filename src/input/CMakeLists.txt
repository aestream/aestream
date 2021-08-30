set(input_definitions "")
set(input_sources file.hpp file.cpp)
set(input_libraries aedat4 ${LZ4_LIBRARY})

# Include libcaer
find_library(CAER_LIBRARY NAMES caer)
if (CAER_LIBRARY)
    set(input_definitions ${input_definitions} WITH_CAER )
    set(input_sources ${input_sources} inivation_device.hpp inivation_device.cpp inivation.hpp inivation.cpp )
    set(input_libraries ${input_libraries} ${CAER_LIBRARY})
else()
    message(STATUS "Inivation dependencies (libcaer) not found, ignoring")
endif()
    
# Include MetavisionSDK
find_library(MetavisionSDK COMPONENTS core driver)
if (MetavisionSDK)
    set(input_definitions ${input_definitions} WITH_METAVISION )
    set(input_sources ${input_sources} prophesee.hpp prophesee.cpp)
    set(input_libraries ${input_libraries} MetavisionSDK::core MetavisionSDK::driver)
else()
    message(STATUS "Prophesee dependencies (MetavisionSDK) not found, ignoring")
endif()

# Create the input library
add_library(aestream_input SHARED ${input_sources})
target_link_libraries(aestream_input PRIVATE ${input_libraries})
target_compile_definitions(aestream_input PUBLIC ${input_definitions})