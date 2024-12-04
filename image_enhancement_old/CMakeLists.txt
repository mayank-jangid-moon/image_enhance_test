# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project Name
project(OpenCV_OpenCL_Project)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Check for OpenCL support
if(OpenCV_VERSION VERSION_LESS "4.0")
    message(FATAL_ERROR "OpenCL support requires OpenCV 4.0 or higher")
endif()

# Print OpenCV build information for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Add the source code
add_executable(${PROJECT_NAME} image_enhancement.cpp)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Enable OpenCL support
add_definitions(-DWITH_OPENCL)
