#pragma once

namespace reduce_v1 {

__global__ void reduce(float* input, float* output, int N) {
    // This version intentionally uses global memory instead of shared memory.
    // The input values are reduced in place within each block's range.
    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * blockDim.x;
    const int index = block_start + tid;

    // Reduce the distance by half after every round: 128, 64, ..., 1.
    // The reverse order prevents a value written in the current round from
    // being read again by another pair in that same round.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && index + stride < N) {
            input[index] += input[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = input[block_start];
    }
}

}  // namespace reduce_v1
