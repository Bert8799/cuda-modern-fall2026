#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>
#include <cuda_runtime.h>

constexpr int THREADS_PER_BLOCK = 256;
constexpr float RELATIVE_TOLERANCE = 1e-5f;

static_assert((THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0,
              "THREADS_PER_BLOCK must be a power of two");

struct CudaDeleter {
    void operator()(float* pointer) const noexcept {
        if (pointer != nullptr) {
            cudaFree(pointer);
        }
    }
};

#define CUDA_CHECK(call)                             \
    do {                                             \
        const cudaError_t error = (call);            \
        if (error != cudaSuccess) {                  \
            std::fprintf(stderr,                     \
                         "%s failed at %s:%d: %s\n", \
                         #call,                      \
                         __FILE__,                   \
                         __LINE__,                   \
                         cudaGetErrorString(error)); \
            return EXIT_FAILURE;                     \
        }                                            \
    } while (false)

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

    // Unroll the last warp to avoid __syncthreads() overhead.
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Unroll the last warp without __syncthreads().
    if (tid < 32) {
        volatile float* volatile_shared_data = shared_data;
        volatile_shared_data[tid] += volatile_shared_data[tid + 32];
        volatile_shared_data[tid] += volatile_shared_data[tid + 16];
        volatile_shared_data[tid] += volatile_shared_data[tid + 8];
        volatile_shared_data[tid] += volatile_shared_data[tid + 4];
        volatile_shared_data[tid] += volatile_shared_data[tid + 2];
        volatile_shared_data[tid] += volatile_shared_data[tid + 1];
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

bool check(const std::vector<float>& expected, const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        return false;
    }

    for (std::size_t i = 0; i < expected.size(); i++) {
        const float difference = std::fabs(expected[i] - actual[i]);
        const float tolerance = RELATIVE_TOLERANCE * std::max(1.0f, std::fabs(expected[i]));
        if (difference > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024;
    const int block_num = (N + THREADS_PER_BLOCK - 1) / (2 * THREADS_PER_BLOCK);

    std::vector<float> input(N);
    std::vector<float> output(block_num);
    std::vector<float> expected(block_num);

    std::mt19937 generator(0);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (float& value : input) {
        value = distribution(generator);
    }

    for (int i = 0; i < block_num; i++) {
        float sum = 0.0f;
        const int block_start = i * 2 * THREADS_PER_BLOCK;
        const int block_end = std::min(block_start + 2 * THREADS_PER_BLOCK, N);
        for (int j = block_start; j < block_end; j++) {
            sum += input[j];
        }
        expected[i] = sum;
    }

    float* device_input = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(float)));

    std::unique_ptr<float, CudaDeleter> input_guard(device_input);

    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), output.size() * sizeof(float)));

    std::unique_ptr<float, CudaDeleter> output_guard(device_output);

    CUDA_CHECK(cudaMemcpy(
        input_guard.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    reduce<<<static_cast<unsigned int>(block_num), THREADS_PER_BLOCK>>>(
        input_guard.get(), output_guard.get(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        output.data(), output_guard.get(), output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    if (check(expected, output)) {
        std::printf("Results match!\n");
    } else {
        std::printf("Results don't match!\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}