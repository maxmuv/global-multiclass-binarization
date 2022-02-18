add_library(IMAGE_PROCESSING image_processing_unit.h image_processing_unit.cpp)

add_executable(IMAGE_PROCESSING_TEST image_processing_test.cpp)

target_link_libraries(IMAGE_PROCESSING_TEST ${OpenCV_LIBS} IMAGE_PROCESSING)

target_link_libraries(IMAGE_PROCESSING ${OpenCV_LIBS} )

add_test(NAME TEST COMMAND IMAGE_PROCESSING_TEST)

install(TARGETS IMAGE_PROCESSING_TEST DESTINATION ${CMAKE_BINARY_DIR})
