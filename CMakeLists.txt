cmake_minimum_required(VERSION 3.20.0)
project(cuda-modern VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(BLAS REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(reduce)