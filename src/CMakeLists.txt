cmake_minimum_required (VERSION 3.4)
project (YoloDroneDetection)


find_package(OpenCV 3.4.10 REQUIRED PATHS "/home/daitao/libs/opencv3.4.10/")
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(darknet STATIC IMPORTED)
set_property(TARGET darknet PROPERTY IMPORTED_LOCATION "../../libdarknetcpu.so")

set(the_target "YoloDroneDetection")
add_executable(${the_target} main.cpp) # add other .cpp and .h files here
target_link_libraries(${the_target} ${OpenCV_LIBS})
target_link_libraries(${the_target} darknet)