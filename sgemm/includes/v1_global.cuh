#pragma once

namespace sgemm_v1 {

// One thread computes one output element and reads both matrices from global
// memory without reusing loaded values across output elements.
constexpr int THREADS_PER_BLOCK = 16;

__global__ void sgemm(const float* A, const float* B, float* C, int m, int k, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || column >= n) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += A[row * k + i] * B[i * n + column];
    }
    C[row * n + column] = sum;
}

}  // namespace sgemm_v1
