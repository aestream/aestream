set(process_definitions "")
set(process_sources transformation.hpp transformation.cpp)

# Create the input library
add_library(aestream_process SHARED ${process_sources})
target_compile_definitions(aestream_process PUBLIC ${process_definitions})