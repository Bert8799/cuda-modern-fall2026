cmake_minimum_required(VERSION 3.20.0)

# Compile for the local GPU by default when the CMake version supports it.
# Must be set before project() to avoid CMake's compiler default. Override
# with -DCMAKE_CUDA_ARCHITECTURES=<arch> when needed.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_VERSION VERSION_LESS 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Enable the C, C++, and CUDA languages for the project.
project(cuda-modern VERSION 0.1.0 LANGUAGES C CXX CUDA)

# Default to a Release build so profiling measures optimized code; configure
# with -DCMAKE_BUILD_TYPE=Debug to build with device debug information instead.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Locate the CPU BLAS libraries and the CUDA Toolkit imported targets.
find_package(BLAS REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Load targets defined by the reduction example.
add_subdirectory(reduce)