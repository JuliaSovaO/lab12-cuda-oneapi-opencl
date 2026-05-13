cmake_minimum_required(VERSION 3.15)

#! Check every comment after the "#!"

#! CHANGE YOUR PROJECT NAME
#  It is used as your project's main executable name.
set(PROJECT_NAME integrate_cuda)
project(${PROJECT_NAME} C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

##########################################################
# Project files, packages, libraries and so on
##########################################################

find_package(CUDA REQUIRED)
find_package(OpenCL REQUIRED)

# CUDA
cuda_add_executable(integrate_cuda
		src/cuda/main_cuda.cpp
		src/cuda/integrate_cuda.cu
)

target_include_directories(integrate_cuda PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
		${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
		${CMAKE_CURRENT_SOURCE_DIR}/src/config_parser
)

set_target_properties(integrate_cuda PROPERTIES
		CUDA_ARCHITECTURES "89"
)

# OpenCL
add_executable(integrate_opencl
		src/opencl/integrate_opencl.cpp
)

target_include_directories(integrate_opencl PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
		${CMAKE_CURRENT_SOURCE_DIR}/src/config_parser
		${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(integrate_opencl
		${OpenCL_LIBRARIES}
)

# Copy configs
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/configs DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/configs)

##########################################################
# Fixed CMakeLists.txt part
##########################################################

# Define ALL_TARGETS variable to use in PVS and Sanitizers
set(ALL_TARGETS integrate_cuda)

# Include CMake setup
include(cmake/main-config.cmake)