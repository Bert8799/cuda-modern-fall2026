add_executable(reduce reduce.cu)

target_compile_features(reduce PRIVATE cuda_std_17)

target_link_libraries(
    reduce
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        BLAS::BLAS
)

target_compile_options(
    reduce
    PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)