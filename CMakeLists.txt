cmake_minimum_required(VERSION 3.7)
project(aedat)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL2 REQUIRED)
find_package(Torch REQUIRED)

find_path(LZ4_INCLUDE_DIR NAMES	lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(viewer viewer.cpp aedat.hpp )
include_directories(viewer ${SDL2_INCLUDE_DIRS} ${LZ4_INCLUDE_DIR})
target_link_libraries(viewer ${SDL2_LIBRARIES} ${LZ4_LIBRARY})


add_library(convert SHARED convert.cpp)
target_link_libraries(convert PRIVATE "${TORCH_LIBRARIES}")


add_executable(converter converter.cpp)
target_link_libraries(converter convert ${TORCH_LIBRARIES})
