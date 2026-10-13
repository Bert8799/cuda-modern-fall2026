#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "includes/v1_global.cuh"
#include "includes/v2_shared.cuh"
#include "includes/v3_two_elements.cuh"
#include "includes/v4_unroll_warp.cuh"
#include "includes/v5_shuffle.cuh"

// Threads per block used to launch every reduction version.
constexpr int THREADS_PER_BLOCK = 256;

// Running without arguments executes the newest kernel version.
constexpr int LATEST_VERSION = 5;
constexpr float RELATIVE_TOLERANCE = 1e-5f;

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

int main(int argc, char* argv[]) {
    const int version = argc > 1 ? std::atoi(argv[1]) : LATEST_VERSION;
    if (version < 1 || version > LATEST_VERSION) {
        std::fprintf(stderr, "usage: %s [kernel version: 1..%d]\n", argv[0], LATEST_VERSION);
        return EXIT_FAILURE;
    }

    const int N = 32 * 1024 * 1024;

    // reduce_v1/v2 reduce THREADS_PER_BLOCK input elements per block, while
    // reduce_v3 and later read two elements per thread. This decides the grid
    // size and how the host-side expected sums are split.
    const int elements_per_block = version <= 2 ? THREADS_PER_BLOCK : 2 * THREADS_PER_BLOCK;
    const int block_num = (N + elements_per_block - 1) / elements_per_block;

    std::vector<float> input(N);
    std::vector<float> output(block_num);
    std::vector<float> expected(block_num);

    std::mt19937 generator(0);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (float& value : input) {
        value = distribution(generator);
    }

    // Host-side expected sums, split into chunks with the same block size as
    // the selected kernel.
    for (int i = 0; i < block_num; i++) {
        float sum = 0.0f;
        const int block_start = i * elements_per_block;
        const int block_end = std::min(block_start + elements_per_block, N);
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

    switch (version) {
        case 1:
            reduce_v1::reduce<<<block_num, THREADS_PER_BLOCK>>>(
                input_guard.get(), output_guard.get(), N);
            break;
        case 2:
            reduce_v2::reduce<<<block_num, THREADS_PER_BLOCK>>>(
                input_guard.get(), output_guard.get(), N);
            break;
        case 3:
            reduce_v3::reduce<<<block_num, THREADS_PER_BLOCK>>>(
                input_guard.get(), output_guard.get(), N);
            break;
        case 4:
            reduce_v4::reduce<<<block_num, THREADS_PER_BLOCK>>>(
                input_guard.get(), output_guard.get(), N);
            break;
        case 5:
            reduce_v5::reduce<<<block_num, THREADS_PER_BLOCK>>>(
                input_guard.get(), output_guard.get(), N);
            break;
    }
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
