# Build the reduction CUDA executable from reduce.cu.
add_executable(reduce reduce.cu)

# Require CUDA C++17 for this target.
target_compile_features(reduce PRIVATE cuda_std_17)

# Link the CUDA runtime, cuBLAS, and CPU BLAS libraries.
target_link_libraries(
    reduce
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        BLAS::BLAS
)

# Generate device debug information only for CUDA sources in Debug builds.
target_compile_options(
    reduce
    PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)