cmake_minimum_required(VERSION 3.7)
project(aedat)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL2 REQUIRED)
find_package(Torch REQUIRED)

# for linking against python
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
list(APPEND TORCH_LIBRARIES ${TORCH_PYTHON_LIBRARY})

find_path(LZ4_INCLUDE_DIR NAMES	lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)
find_package (Python3 REQUIRED COMPONENTS Development)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(viewer viewer.cpp aedat.hpp )

include_directories(viewer ${SDL2_INCLUDE_DIRS} ${LZ4_INCLUDE_DIR} PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(viewer ${SDL2_LIBRARIES} ${LZ4_LIBRARY} ${Python3_LIBRARIES})


add_library(convert SHARED convert.cpp)
target_compile_features(convert PRIVATE cxx_std_14)
target_include_directories(convert PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_link_directories(convert PRIVATE ${TORCH_LINK_DIRECTORIES})
target_link_libraries(convert PRIVATE ${TORCH_LIBRARIES} ${Python3_LIBRARIES} ${LZ4_LIBRARY})


add_executable(converter converter.cpp)
target_include_directories(converter PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(converter convert ${TORCH_LIBRARIES})
