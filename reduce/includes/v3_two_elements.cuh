#pragma once

namespace reduce_v3 {

constexpr int THREADS_PER_BLOCK = 256;

static_assert((THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0,
              "THREADS_PER_BLOCK must be a power of two");

__global__ void reduce(float* input, float* output, int N) {
    __shared__ float shared_data[THREADS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * blockDim.x * 2;
    const int index_a = block_start + tid;
    const int index_b = index_a + blockDim.x;

    // load two elements per thread from global memory to shared memory
    const float a = index_a < N ? input[index_a] : 0.0f;
    const float b = index_b < N ? input[index_b] : 0.0f;

    // Pad an incomplete final block with zero so every shared-memory access
    // during the reduction stays within the fixed-size shared array.
    shared_data[tid] = a + b;
    __syncthreads();

    // Reduce the distance by half after every round: 128, 64, ..., 1.
    // The reverse order prevents a value written in the current round from
    // being read again by another pair in that same round.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

}  // namespace reduce_v3
