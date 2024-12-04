cmake_minimum_required(VERSION 3.10)

# Project Name
project(CudaImageEnhancement)

# Set CUDA as the language
enable_language(CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Optimization and OpenMP Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# Find OpenCV Package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add CUDA source files
set(SOURCES image_enhancement.cpp image_enhancement_kernel.cu)

# Add executable for CUDA
add_executable(image_enhancement ${SOURCES})

# Link OpenCV libraries
target_link_libraries(image_enhancement ${OpenCV_LIBS})

# Ensure CUDA code is compiled with the proper flags
target_compile_options(image_enhancement PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)

