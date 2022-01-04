# LZ4 for AEDAT encoding
find_path(LZ4_INCLUDE_DIR NAMES	lz4.h)
find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)

# AEDAT4 processing
add_library(aedat4 SHARED aedat4.hpp aedat4.cpp generator.hpp)
target_include_directories(aedat4 PUBLIC .)
include_directories(aedat4 PRIVATE ${LZ4_INCLUDE_DIR})
target_link_libraries(aedat4 PRIVATE ${LZ4_LIBRARY})

# Add subdirectories
add_subdirectory("input")
add_subdirectory("output")

# # find_package(OpenCV COMPONENTS highgui)

# # Include OpenCV
# # find_package (OpenCV REQUIRED)
# # include_directories(${OpenCV_DIRS})

# AEStream executable
add_executable(aestream aestream.cpp)
target_include_directories(aestream PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(aestream PRIVATE aedat4 aestream_input aestream_output)