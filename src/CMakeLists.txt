# LZ4 for AEDAT encoding
find_path(LZ4_INCLUDE_DIR NAMES	lz4.h)
find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)

# AEDAT4 processing
add_library(aedat4 SHARED aedat.hpp aedat4.hpp generator.hpp)
include_directories(aedat4 PRIVATE ${LZ4_INCLUDE_DIR})
target_link_libraries(aedat4 PRIVATE ${LZ4_LIBRARY})
set_target_properties(aedat4 PROPERTIES LINKER_LANGUAGE CXX)

# Add subdirectories
add_subdirectory(input)
add_subdirectory(output)

# AEStream library
add_library(aestream_lib SHARED aestream.cpp)
target_include_directories(aestream_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(aestream_lib PUBLIC aedat4 aestream_input aestream_output)

# AEStream executable
add_executable(aestream aestream.cpp)
target_link_libraries(aestream PRIVATE aestream_lib)
