cmake_minimum_required(VERSION 3.10)

# Project Name
project(ImageEnhancement)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Optimization and OpenMP Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")

# Find OpenCV Package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable target
add_executable(image_enhancement image_enhancement.cpp)

# Link the OpenCV libraries
target_link_libraries(image_enhancement ${OpenCV_LIBS})

