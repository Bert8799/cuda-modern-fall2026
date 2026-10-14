#pragma once

#include <cstddef>

namespace sgemm_v4 {

// Tile dimensions staged in shared memory for A and B.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

// Each thread accumulates a THREAD_TILE_Y x THREAD_TILE_X output tile.
constexpr int THREAD_TILE_X = 8;
constexpr int THREAD_TILE_Y = 8;
constexpr int BLOCK_X = BN / THREAD_TILE_X;
constexpr int BLOCK_Y = BM / THREAD_TILE_Y;

// A float4 moves four adjacent FP32 values in one 128-bit vector operation.
constexpr int VECTOR_WIDTH = 4;
constexpr int NUM_THREADS = BLOCK_X * BLOCK_Y;
constexpr int A_VECTORS_PER_ROW = BK / VECTOR_WIDTH;
constexpr int B_VECTORS_PER_ROW = BN / VECTOR_WIDTH;
constexpr int A_VECTOR_COUNT = BM * A_VECTORS_PER_ROW;
constexpr int B_VECTOR_COUNT = BK * B_VECTORS_PER_ROW;

static_assert(BLOCK_Y * THREAD_TILE_Y == BM && BLOCK_X * THREAD_TILE_X == BN,
              "the thread tile must cover the output tile exactly");
static_assert(BK % VECTOR_WIDTH == 0 && BN % VECTOR_WIDTH == 0 && THREAD_TILE_X % VECTOR_WIDTH == 0,
              "K, N, and each thread's output columns must contain complete float4 vectors");
static_assert(A_VECTOR_COUNT == NUM_THREADS && B_VECTOR_COUNT == NUM_THREADS,
              "each thread must load one float4 from each input tile");
static_assert(alignof(float4) == 16, "float4 accesses require 16-byte alignment");

// This fast path assumes complete tiles and aligned float4 accesses.
__global__ void sgemm(const float* A, const float* B, float* C, int k, int n) {
    // A is transposed to [K, M] while it is staged; B stays in [K, N] order.
    __shared__ __align__(16) float a_tile[BM * BK];
    __shared__ __align__(16) float b_tile[BK * BN];

    const int tile_row = blockIdx.y * BM;     // First matrix row covered by this block.
    const int tile_column = blockIdx.x * BN;  // First matrix column covered by this block.

    // Threads are arranged as 16 output-column groups by 16 output-row groups.
    // Each thread owns one contiguous THREAD_TILE_Y x THREAD_TILE_X output region.
    const int tid = threadIdx.x;
    const int thread_col = tid % BLOCK_X;
    const int thread_row = tid / BLOCK_X;

    // Per-thread accumulators hold the complete output tile owned by this thread.
    float sums[THREAD_TILE_Y][THREAD_TILE_X] = {0.0f};
    // These register fragments are reused across the outer product for each K value.
    float reg_m[THREAD_TILE_Y];
    float reg_n[THREAD_TILE_X];

    // One thread loads one float4 from each input tile.
    const int inner_row_a = tid / A_VECTORS_PER_ROW;  // A row selected by this thread.
    const int inner_col_a = tid % A_VECTORS_PER_ROW;  // A float4 position within BK.
    const int inner_row_b = tid / B_VECTORS_PER_ROW;  // B K row selected by this thread.
    const int inner_col_b = tid % B_VECTORS_PER_ROW;  // B float4 position within BN.

    for (int tile = 0; tile < k; tile += BK) {
        // Load four adjacent A values, then transpose them into K-major shared memory.
        // This makes each dot-product slice address A as a contiguous row fragment.
        const int a_global_row = tile_row + inner_row_a;  // A element row in global memory.
        const int a_global_column =
            tile + inner_col_a * VECTOR_WIDTH;  // First K value in A float4.
        const std::size_t a_shared_offset =
            static_cast<std::size_t>(a_global_column - tile) * BM + inner_row_a;

        const std::size_t a_global_offset =
            static_cast<std::size_t>(a_global_row) * k + a_global_column;
        const float4 a_values = *reinterpret_cast<const float4*>(A + a_global_offset);
        a_tile[a_shared_offset + 0 * BM] = a_values.x;
        a_tile[a_shared_offset + 1 * BM] = a_values.y;
        a_tile[a_shared_offset + 2 * BM] = a_values.z;
        a_tile[a_shared_offset + 3 * BM] = a_values.w;

        // B is already K-major, so its aligned float4 can be copied directly to shared memory.
        const int b_global_row = tile + inner_row_b;  // B K row in global memory.
        const int b_global_column =
            tile_column + inner_col_b * VECTOR_WIDTH;  // First column in B float4.
        const std::size_t b_shared_offset =
            static_cast<std::size_t>(inner_row_b) * BN + inner_col_b * VECTOR_WIDTH;

        const std::size_t b_global_offset =
            static_cast<std::size_t>(b_global_row) * n + b_global_column;
        *reinterpret_cast<float4*>(b_tile + b_shared_offset) =
            *reinterpret_cast<const float4*>(B + b_global_offset);
        __syncthreads();

        // Reuse each loaded A and B fragment across the thread's output tile.
        for (int i = 0; i < BK; ++i) {
            for (int row = 0; row < THREAD_TILE_Y; ++row) {
                reg_m[row] = a_tile[i * BM + thread_row * THREAD_TILE_Y + row];
            }
            for (int column = 0; column < THREAD_TILE_X; ++column) {
                reg_n[column] = b_tile[i * BN + thread_col * THREAD_TILE_X + column];
            }
            for (int row = 0; row < THREAD_TILE_Y; ++row) {
                for (int column = 0; column < THREAD_TILE_X; ++column) {
                    sums[row][column] += reg_m[row] * reg_n[column];
                }
            }
        }
        __syncthreads();
    }

    // Store each thread's contiguous output row in float4 chunks.
    for (int row = 0; row < THREAD_TILE_Y; ++row) {
        const int output_row = tile_row + thread_row * THREAD_TILE_Y + row;
        for (int column = 0; column < THREAD_TILE_X; column += VECTOR_WIDTH) {
            const int output_column = tile_column + thread_col * THREAD_TILE_X + column;
            float4 output_values;
            output_values.x = sums[row][column + 0];
            output_values.y = sums[row][column + 1];
            output_values.z = sums[row][column + 2];
            output_values.w = sums[row][column + 3];
            *reinterpret_cast<float4*>(C + static_cast<std::size_t>(output_row) * n +
                                       output_column) = output_values;
        }
    }
}

}  // namespace sgemm_v4
