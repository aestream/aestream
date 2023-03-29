# AER processing
add_library(aer INTERFACE  aer.hpp generator.hpp)
target_include_directories(aer INTERFACE .)
set_target_properties(aer PROPERTIES LINKER_LANGUAGE CXX)


add_subdirectory(file)