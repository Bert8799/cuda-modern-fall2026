#pragma once

namespace reduce_v5 {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARP_COUNT = THREADS_PER_BLOCK / WARP_SIZE;

static_assert((THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0,
              "THREADS_PER_BLOCK must be a power of two");

// Reduce a partial sum across all 32 lanes of a warp with shuffle instructions.
// The value converges to lane 0; every lane must call it with the same mask.
__device__ __forceinline__ float warp_reduce_sum(float value) {
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);
    return value;
}

__global__ void reduce(float* input, float* output, int N) {
    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * blockDim.x * 2;
    const int index_a = block_start + tid;
    const int index_b = index_a + blockDim.x;

    // Load two elements per thread and add them in registers.
    const float a = index_a < N ? input[index_a] : 0.0f;
    const float b = index_b < N ? input[index_b] : 0.0f;

    // Reduce each warp to a single partial sum in registers, then hand only
    // the warp-level sums to shared memory for the cross-warp reduction.
    __shared__ float warp_level_sums[WARP_COUNT];
    const float warp_sum = warp_reduce_sum(a + b);
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    if (lane_id == 0) {
        warp_level_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    // Let the first warp reduce the WARP_COUNT warp-level sums.
    if (warp_id == 0) {
        const float warp_level_sum = lane_id < WARP_COUNT ? warp_level_sums[lane_id] : 0.0f;
        const float block_sum = warp_reduce_sum(warp_level_sum);
        if (lane_id == 0) {
            output[blockIdx.x] = block_sum;
        }
    }
}

}  // namespace reduce_v5
