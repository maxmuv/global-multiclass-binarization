cmake_minimum_required(VERSION 3.0)

project(binarization)

include_directories(thirdparty)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(binarization binarization_main.cpp)

target_link_libraries(binarization ${OpenCV_LIBS} IMAGE_PROCESSING)

add_subdirectory(image_processing)
