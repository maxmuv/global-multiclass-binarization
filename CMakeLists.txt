cmake_minimum_required(VERSION 3.0)

project(binarization)

include_directories(thirdparty)

set(CMAKE_CXX_STANDARD 11)

option(DRAWADD "key to draw hists and plots" ON)
if (DRAWADD)
  add_definitions(-Ddraw=1)
else (DRAWADD)
  add_definitions(-Ddraw=0)
endif (DRAWADD)

find_package(OpenCV 4.0 REQUIRED)
if (OpenCV_FOUND)
  message("OpenCV is found")
else (OpenCV_FOUND)
  message("OpenCV need to be installed to run the program")
endif (OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(binarization binarization_main.cpp)

target_link_libraries(binarization ${OpenCV_LIBS} IMAGE_PROCESSING)

find_package(Doxygen)
if (DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen_config.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxygen)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    add_custom_target( doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

enable_testing()
add_subdirectory(image_processing)
add_subdirectory(tools)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/demo/ DESTINATION ${CMAKE_BINARY_DIR}/demo/)
install(TARGETS binarization DESTINATION ${CMAKE_BINARY_DIR})
