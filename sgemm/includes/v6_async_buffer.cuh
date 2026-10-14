#pragma once

#include <cstddef>
#include <cuda/pipeline>

namespace sgemm_v6 {

// Async double buffering overlaps the next K tile's global-to-shared copies
// with the current tile's warp-tiled computation. Each buffer holds A and B.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int NUM_BUFFERS = 2;

constexpr int NUM_THREADS = 128;
constexpr int WARP_SIZE = 32;
constexpr int WM = 64;
constexpr int WN = 64;

constexpr int WNITER = 4;
constexpr int THREAD_TILE_X = 4;
constexpr int THREAD_TILE_Y = 8;
constexpr int WMITER = (WM * WN) / (WARP_SIZE * THREAD_TILE_Y * THREAD_TILE_X * WNITER);
constexpr int WSUBM = WM / WMITER;
constexpr int WSUBN = WN / WNITER;

constexpr int VECTOR_WIDTH = 4;
constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
constexpr int A_VECTORS_PER_ROW = BK / VECTOR_WIDTH;
constexpr int B_VECTORS_PER_ROW = BN / VECTOR_WIDTH;
constexpr int A_ROW_STRIDE = (NUM_THREADS * VECTOR_WIDTH) / BK;
constexpr int B_ROW_STRIDE = NUM_THREADS / B_VECTORS_PER_ROW;

static_assert(NUM_THREADS % WARP_SIZE == 0, "a block must contain complete warps");
static_assert((BM % WM == 0) && (BN % WN == 0), "the warp tiles must evenly divide the block tile");
static_assert((BM / WM) * (BN / WN) == NUM_WARPS,
              "the warp tiles must cover the block tile exactly");
static_assert((WM * WN) % (WARP_SIZE * THREAD_TILE_Y * THREAD_TILE_X * WNITER) == 0,
              "the thread tiles must evenly divide each warp tile");
static_assert(WM % WMITER == 0 && WN % WNITER == 0,
              "the warp subtiles must cover each warp tile exactly");
static_assert(BK % VECTOR_WIDTH == 0 && BN % VECTOR_WIDTH == 0 && THREAD_TILE_X % VECTOR_WIDTH == 0,
              "float4 accesses require complete vectors");
static_assert((NUM_THREADS * VECTOR_WIDTH) % BK == 0 && BM % A_ROW_STRIDE == 0 &&
                  NUM_THREADS == A_ROW_STRIDE * A_VECTORS_PER_ROW,
              "threads must cover the A tile with complete float4 rows");
static_assert(BK % B_ROW_STRIDE == 0 && NUM_THREADS == B_ROW_STRIDE * B_VECTORS_PER_ROW,
              "threads must cover the B tile with complete float4 rows");
static_assert(alignof(float4) == 16, "float4 accesses require 16-byte alignment");

__device__ __forceinline__ void load_tiles_from_global_memory(const float* A,
                                                              const float* B,
                                                              float* a_tile,
                                                              float* b_tile,
                                                              int inner_row_a,
                                                              int inner_col_a,
                                                              int inner_row_b,
                                                              int inner_col_b,
                                                              int k,
                                                              int n,
                                                              cuda::pipeline<cuda::thread_scope_thread>& pipeline) {
    // A is transposed to [K, M]. Each component needs its own 4-byte async copy
    // because adjacent global K values go to different shared-memory K planes.
    for (int offset = 0; offset + A_ROW_STRIDE <= BM; offset += A_ROW_STRIDE) {
        for (int column = 0; column < VECTOR_WIDTH; ++column) {
            const int a_shared_offset =
                (inner_col_a * VECTOR_WIDTH + column) * BM + inner_row_a + offset;
            const std::size_t a_global_offset =
                static_cast<std::size_t>(inner_row_a + offset) * k +
                inner_col_a * VECTOR_WIDTH + column;
            cuda::memcpy_async(a_tile + a_shared_offset, A + a_global_offset,
                               cuda::aligned_size_t<sizeof(float)>(sizeof(float)), pipeline);
        }
    }

    for (int offset = 0; offset + B_ROW_STRIDE <= BK; offset += B_ROW_STRIDE) {
        const int b_shared_offset =
            (inner_row_b + offset) * BN + inner_col_b * VECTOR_WIDTH;
        const std::size_t b_global_offset =
            static_cast<std::size_t>(inner_row_b + offset) * n + inner_col_b * VECTOR_WIDTH;
        // B is contiguous in both layouts: one aligned 16-byte async copy
        // bypasses the intermediate float4 register used by a synchronous load.
        cuda::memcpy_async(b_tile + b_shared_offset, B + b_global_offset,
                           cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)), pipeline);
    }
}

__device__ __forceinline__ void compute_warp_tile(float* sums,
                                                  float* reg_m,
                                                  float* reg_n,
                                                  const float* a_tile,
                                                  const float* b_tile,
                                                  int warp_row,
                                                  int warp_col,
                                                  int thread_row,
                                                  int thread_col) {
    for (int i = 0; i < BK; ++i) {
        for (int warp_subrow = 0; warp_subrow < WMITER; ++warp_subrow) {
            for (int row = 0; row < THREAD_TILE_Y; ++row) {
                reg_m[warp_subrow * THREAD_TILE_Y + row] =
                    a_tile[i * BM + warp_row * WM + warp_subrow * WSUBM +
                           thread_row * THREAD_TILE_Y + row];
            }
        }
        for (int warp_subcolumn = 0; warp_subcolumn < WNITER; ++warp_subcolumn) {
            for (int column = 0; column < THREAD_TILE_X; ++column) {
                reg_n[warp_subcolumn * THREAD_TILE_X + column] =
                    b_tile[i * BN + warp_col * WN + warp_subcolumn * WSUBN +
                           thread_col * THREAD_TILE_X + column];
            }
        }

        for (int warp_subrow = 0; warp_subrow < WMITER; ++warp_subrow) {
            for (int warp_subcolumn = 0; warp_subcolumn < WNITER; ++warp_subcolumn) {
                for (int row = 0; row < THREAD_TILE_Y; ++row) {
                    for (int column = 0; column < THREAD_TILE_X; ++column) {
                        const int sum_index =
                            (warp_subrow * THREAD_TILE_Y + row) * (WNITER * THREAD_TILE_X) +
                            warp_subcolumn * THREAD_TILE_X + column;
                        sums[sum_index] += reg_m[warp_subrow * THREAD_TILE_Y + row] *
                                           reg_n[warp_subcolumn * THREAD_TILE_X + column];
                    }
                }
            }
        }
    }
}

// Hardware global-to-shared async copies require compute capability 8.0 or newer.
__global__ __launch_bounds__(NUM_THREADS) void sgemm(
    const float* A, const float* B, float* C, int k, int n) {
    __shared__ __align__(16) float a_tile[NUM_BUFFERS][BM * BK];
    __shared__ __align__(16) float b_tile[NUM_BUFFERS][BK * BN];

    const int tile_row = blockIdx.y * BM;
    const int tile_column = blockIdx.x * BN;

    const int warp_index = threadIdx.x / WARP_SIZE;
    const int warp_col = warp_index % (BN / WN);
    const int warp_row = warp_index / (BN / WN);

    const int lane_index = threadIdx.x % WARP_SIZE;
    const int thread_col = lane_index % (WSUBN / THREAD_TILE_X);
    const int thread_row = lane_index / (WSUBN / THREAD_TILE_X);

    const int inner_row_a = threadIdx.x / A_VECTORS_PER_ROW;
    const int inner_col_a = threadIdx.x % A_VECTORS_PER_ROW;
    const int inner_row_b = threadIdx.x / B_VECTORS_PER_ROW;
    const int inner_col_b = threadIdx.x % B_VECTORS_PER_ROW;

    float sums[WMITER * THREAD_TILE_Y * WNITER * THREAD_TILE_X] = {0.0f};
    float reg_m[WMITER * THREAD_TILE_Y];
    float reg_n[WNITER * THREAD_TILE_X];

    const float* tile_a = A + static_cast<std::size_t>(tile_row) * k;
    const float* tile_b = B + tile_column;
    auto pipeline = cuda::make_pipeline();

    // Prime buffer 0. Every thread commits the same copy stage while converged.
    pipeline.producer_acquire();
    load_tiles_from_global_memory(tile_a, tile_b, a_tile[0], b_tile[0],
                                  inner_row_a, inner_col_a, inner_row_b, inner_col_b,
                                  k, n, pipeline);
    pipeline.producer_commit();

    int buffer = 0;
    for (int tile = 0; tile < k; tile += BK) {
        // XOR swaps buffer 0/1: compute from buffer while filling next_buffer.
        const int next_buffer = buffer ^ 1;
        const bool has_next = tile + BK < k;
        if (has_next) {
            // Prefetch the next K slice into the other buffer before computing
            // this one: A advances BK columns and B advances BK matrix rows.
            pipeline.producer_acquire();
            load_tiles_from_global_memory(
                tile_a + tile + BK, tile_b + static_cast<std::size_t>(tile + BK) * n,
                a_tile[next_buffer], b_tile[next_buffer], inner_row_a, inner_col_a,
                inner_row_b, inner_col_b, k, n, pipeline);
            pipeline.producer_commit();

            // Wait for the current stage, leaving the newest one in flight.
            cuda::pipeline_consumer_wait_prior<1>(pipeline);
        } else {
            // Drain the last stage; no out-of-range prefetch is submitted.
            cuda::pipeline_consumer_wait_prior<0>(pipeline);
        }

        // Thread-scoped waits cover only each thread's own copies. The block
        // barrier makes every thread's A/B values available to all four warps.
        __syncthreads();

        compute_warp_tile(
            sums, reg_m, reg_n, a_tile[buffer], b_tile[buffer], warp_row, warp_col,
            thread_row, thread_col);

        // All warps must finish reading before this buffer can be overwritten
        // by the iteration after next. Then swap the read/prefetch roles.
        __syncthreads();
        pipeline.consumer_release();
        buffer = next_buffer;
    }

    float* warp_output =
        C + static_cast<std::size_t>(tile_row + warp_row * WM) * n + tile_column + warp_col * WN;
    for (int warp_subrow = 0; warp_subrow < WMITER; ++warp_subrow) {
        for (int warp_subcolumn = 0; warp_subcolumn < WNITER; ++warp_subcolumn) {
            float* subtile_output = warp_output +
                                    static_cast<std::size_t>(warp_subrow * WSUBM) * n +
                                    warp_subcolumn * WSUBN;
            for (int row = 0; row < THREAD_TILE_Y; ++row) {
                for (int column = 0; column < THREAD_TILE_X; column += VECTOR_WIDTH) {
                    const int sum_index =
                        (warp_subrow * THREAD_TILE_Y + row) * (WNITER * THREAD_TILE_X) +
                        warp_subcolumn * THREAD_TILE_X + column;
                    const float4 output_values = {sums[sum_index + 0],
                                                  sums[sum_index + 1],
                                                  sums[sum_index + 2],
                                                  sums[sum_index + 3]};
                    *reinterpret_cast<float4*>(
                        subtile_output +
                        static_cast<std::size_t>(thread_row * THREAD_TILE_Y + row) * n +
                        thread_col * THREAD_TILE_X + column) = output_values;
                }
            }
        }
    }
}

}  // namespace sgemm_v6
