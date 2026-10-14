#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "includes/v1_global.cuh"
#include "includes/v2_global_1d.cuh"
#include "includes/v3_shared.cuh"
#include "includes/v4_shared_float4.cuh"

constexpr int M = 2048;
constexpr int K = 2048;
constexpr int N = 2048;
// v4 uses unchecked float4 accesses, so its fixed matrix must contain full tiles and aligned rows.
static_assert(M % sgemm_v4::BM == 0 && N % sgemm_v4::BN == 0 && K % sgemm_v4::BK == 0,
              "v4 requires complete BM x BN x BK tiles");
static_assert(K % sgemm_v4::VECTOR_WIDTH == 0 && N % sgemm_v4::VECTOR_WIDTH == 0,
              "v4 requires float4-aligned A, B, and C row strides");
// Running without arguments executes the newest kernel version.
constexpr int LATEST_VERSION = 4;
// The reference and the kernel sum the same K products in the same order,
// but the kernel may fuse some operations into FMAs, so the results differ by
// a few ULPs. 1e-4 covers that while still catching real indexing bugs, which
// produce differences of order 1.
constexpr float RELATIVE_TOLERANCE = 1e-4f;

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

    const std::size_t a_count = static_cast<std::size_t>(M) * K;
    const std::size_t b_count = static_cast<std::size_t>(K) * N;
    const std::size_t c_count = static_cast<std::size_t>(M) * N;

    std::vector<float> host_a(a_count);
    std::vector<float> host_b(b_count);
    std::vector<float> host_c(c_count);
    std::vector<float> expected(c_count);

    std::mt19937 generator(0);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (std::size_t i = 0; i < a_count; i++) {
        host_a[i] = distribution(generator);
    }
    for (std::size_t i = 0; i < b_count; i++) {
        host_b[i] = distribution(generator);
    }

    // Reference computed on the host with the same summation order as the
    // kernel (sequential dot product over k), with B transposed so both
    // operands are walked with unit stride.
    std::vector<float> transposed_b(b_count);
    for (int i = 0; i < K; i++) {
        for (int column = 0; column < N; column++) {
            transposed_b[column * K + i] = host_b[i * N + column];
        }
    }

    for (int row = 0; row < M; row++) {
        for (int column = 0; column < N; column++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += host_a[row * K + i] * transposed_b[column * K + i];
            }
            expected[row * N + column] = sum;
        }
    }

    float* device_a = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a), a_count * sizeof(float)));

    std::unique_ptr<float, CudaDeleter> a_guard(device_a);

    float* device_b = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b), b_count * sizeof(float)));

    std::unique_ptr<float, CudaDeleter> b_guard(device_b);

    float* device_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c), c_count * sizeof(float)));

    std::unique_ptr<float, CudaDeleter> c_guard(device_c);

    CUDA_CHECK(
        cudaMemcpy(a_guard.get(), host_a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(b_guard.get(), host_b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block;
    dim3 grid;
    if (version == 1) {
        block = dim3(sgemm_v1::THREADS_PER_BLOCK, sgemm_v1::THREADS_PER_BLOCK);
        grid = dim3((N + sgemm_v1::THREADS_PER_BLOCK - 1) / sgemm_v1::THREADS_PER_BLOCK,
                    (M + sgemm_v1::THREADS_PER_BLOCK - 1) / sgemm_v1::THREADS_PER_BLOCK);
    } else if (version == 2) {
        block = dim3(sgemm_v2::THREADS_PER_BLOCK);
        const std::size_t block_count =
            (c_count + sgemm_v2::THREADS_PER_BLOCK - 1) / sgemm_v2::THREADS_PER_BLOCK;
        grid = dim3(static_cast<unsigned int>(block_count));
    } else if (version == 3) {
        block = dim3(sgemm_v3::BLOCK_X, sgemm_v3::BLOCK_Y);
        grid = dim3((N + sgemm_v3::BN - 1) / sgemm_v3::BN,
                    (M + sgemm_v3::BM - 1) / sgemm_v3::BM);
    } else {
        block = dim3(sgemm_v4::NUM_THREADS);
        grid = dim3((N + sgemm_v4::BN - 1) / sgemm_v4::BN,
                    (M + sgemm_v4::BM - 1) / sgemm_v4::BM);
    }

    switch (version) {
        case 1:
            sgemm_v1::sgemm<<<grid, block>>>(
                a_guard.get(), b_guard.get(), c_guard.get(), M, K, N);
            break;
        case 2:
            sgemm_v2::sgemm<<<grid, block>>>(
                a_guard.get(), b_guard.get(), c_guard.get(), M, K, N);
            break;
        case 3:
            sgemm_v3::sgemm<<<grid, block>>>(
                a_guard.get(), b_guard.get(), c_guard.get(), M, K, N);
            break;
        case 4:
            sgemm_v4::sgemm<<<grid, block>>>(
                a_guard.get(), b_guard.get(), c_guard.get(), K, N);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(host_c.data(), c_guard.get(), c_count * sizeof(float), cudaMemcpyDeviceToHost));

    if (check(expected, host_c)) {
        std::printf("Results match!\n");
    } else {
        std::printf("Results don't match!\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
