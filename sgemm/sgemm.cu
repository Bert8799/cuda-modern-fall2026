#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>
#include <cuda_runtime.h>

constexpr int M = 2048;
constexpr int K = 2048;
constexpr int N = 2048;
// Sizes of the tiles staged in shared memory: bm x bk for A, bk x bn for B,
// and bm x bn for the output tile a block owns.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
// Outputs a single thread accumulates: THREAD_TILE_Y rows x THREAD_TILE_X
// columns. The thread block size follows from the tile sizes: BLOCK_X x
// BLOCK_Y threads cover the BM x BN output tile exactly.
constexpr int THREAD_TILE_X = 8;
constexpr int THREAD_TILE_Y = 8;
constexpr int BLOCK_X = BN / THREAD_TILE_X;
constexpr int BLOCK_Y = BM / THREAD_TILE_Y;

static_assert(BLOCK_Y * THREAD_TILE_Y == BM && BLOCK_X * THREAD_TILE_X == BN,
              "the thread tile must tile the output tile exactly");
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

// SGEMM with shared memory tiles: each block owns a BM x BN tile of C and
// walks the K dimension BK at a time. The matching BM x BK tile of A and
// BK x BN tile of B are loaded into shared memory, and every thread
// accumulates the elements of its output tile from there.
__global__ void sgemm(const float* A, const float* B, float* C, int m, int k, int n) {
    __shared__ float a_tile[BM * BK];
    __shared__ float b_tile[BK * BN];

    const int tile_row = blockIdx.y * BM;
    const int tile_column = blockIdx.x * BN;
    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;

    // Each thread owns a THREAD_TILE_Y x THREAD_TILE_X block of outputs at
    // rows ty, ty + BLOCK_Y, ... and columns tx, tx + BLOCK_X, ... of the
    // output tile, so every thread accumulates the same number of partial
    // sums across the whole K loop.
    float sums[THREAD_TILE_Y][THREAD_TILE_X];
    for (int row = 0; row < THREAD_TILE_Y; row++) {
        for (int column = 0; column < THREAD_TILE_X; column++) {
            sums[row][column] = 0.0f;
        }
    }

    for (int tile = 0; tile < k; tile += BK) {
        // Load the A and B tiles cooperatively with unit stride.
        for (int index = tid; index < BM * BK; index += BLOCK_X * BLOCK_Y) {
            a_tile[index] = A[(tile_row + index / BK) * k + tile + index % BK];
        }
        for (int index = tid; index < BK * BN; index += BLOCK_X * BLOCK_Y) {
            b_tile[index] = B[(tile + index / BN) * n + tile_column + index % BN];
        }
        __syncthreads();

        // Accumulate this K slice into the partial sums of every owned output.
        for (int i = 0; i < BK; i++) {
            for (int row = 0; row < THREAD_TILE_Y; row++) {
                for (int column = 0; column < THREAD_TILE_X; column++) {
                    sums[row][column] +=
                        a_tile[(threadIdx.y + row * BLOCK_Y) * BK + i] *
                        b_tile[i * BN + threadIdx.x + column * BLOCK_X];
                }
            }
        }
        __syncthreads();
    }

    for (int row = 0; row < THREAD_TILE_Y; row++) {
        for (int column = 0; column < THREAD_TILE_X; column++) {
            C[(tile_row + threadIdx.y + row * BLOCK_Y) * n + tile_column + threadIdx.x +
              column * BLOCK_X] = sums[row][column];
        }
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

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm<<<grid, block>>>(a_guard.get(), b_guard.get(), c_guard.get(), M, K, N);
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
