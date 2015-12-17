cmake_minimum_required(VERSION 3.0.0)
project(EECS442Project)
find_package(OpenCV REQUIRED CONFIG)
set(cpp_files
  DetectorTest.cpp
  DetectionTracker.h
  DetectionTracker.cpp
  )

add_executable(DetectorTest ${cpp_files})
target_link_libraries(DetectorTest ${OpenCV_LIBS})
target_include_directories(DetectorTest PRIVATE ${OpenCV_INCLUDE_DIRS})

set_property(TARGET DetectorTest PROPERTY CXX_STANDARD 11)
