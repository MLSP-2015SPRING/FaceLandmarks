cmake_minimum_required(VERSION 2.8)
project(mlsp_project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)

find_package( OpenCV REQUIRED )
add_executable(mlsp_project ${SOURCE_FILES})
add_subdirectory(asmlib-opencv)

include_directories(${mlsp-opencv_SOURCE_DIR}/asmlib-opencv)
link_directories(${mlsp-opencv_SOURCE_DIR}/asmlib-opencv)

target_link_libraries( mlsp_project ${OpenCV_LIBS} )
target_link_libraries( mlsp_project asm )