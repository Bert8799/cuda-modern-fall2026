#pragma once

namespace sgemm_v2 {

// A one-dimensional block maps consecutive threads to consecutive output
// elements, so each warp reads adjacent columns of B during every K step.
constexpr int THREADS_PER_BLOCK = 256;

__global__ void sgemm(const float* A, const float* B, float* C, int m, int k, int n) {
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_count = m * n;
    if (output_index >= output_count) {
        return;
    }

    const int row = output_index / n;
    const int column = output_index % n;

    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += A[row * k + i] * B[i * n + column];
    }
    C[output_index] = sum;
}

}  // namespace sgemm_v2
