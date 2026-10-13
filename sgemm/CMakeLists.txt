# Build the SGEMM CUDA executable from sgemm.cu.
add_executable(sgemm sgemm.cu)

# Require CUDA C++17 for this target.
target_compile_features(sgemm PRIVATE cuda_std_17)

# Link the CUDA runtime library.
target_link_libraries(
    sgemm
    PRIVATE
        CUDA::cudart
)

# Generate device debug information only for CUDA sources in Debug builds.
target_compile_options(
    sgemm
    PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)
