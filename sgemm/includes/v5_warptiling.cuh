#pragma once

#include <cstddef>

namespace sgemm_v5 {

// Warp tiling assigns each 64 x 64 output tile to one warp.
// The block uses 128 threads instead of 256; each lane reuses 8 A and 16 B
// register values to accumulate 128 outputs across its warp subtiles.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

// Four 64 x 64 warp tiles cover the 128 x 128 block tile in a 2 x 2 arrangement.
constexpr int NUM_THREADS = 128;
constexpr int WARP_SIZE = 32;
constexpr int WM = 64;
constexpr int WN = 64;

// Split each warp tile into four 64 x 16 column subtiles; a lane owns 8 x 4
// outputs in each subtile and reuses its A fragment across all four.
constexpr int WNITER = 4;
constexpr int THREAD_TILE_X = 4;
constexpr int THREAD_TILE_Y = 8;
// Row subtile count: warp area / (32 lanes * thread tile area * column subtiles).
constexpr int WMITER = (WM * WN) / (WARP_SIZE * THREAD_TILE_Y * THREAD_TILE_X * WNITER);
// Divide the warp's rows/columns by their subtile counts: WSUBM=64, WSUBN=16.
constexpr int WSUBM = WM / WMITER;
constexpr int WSUBN = WN / WNITER;

constexpr int VECTOR_WIDTH = 4;
constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
constexpr int A_VECTORS_PER_ROW = BK / VECTOR_WIDTH;
constexpr int B_VECTORS_PER_ROW = BN / VECTOR_WIDTH;
// One load round covers NUM_THREADS*VECTOR_WIDTH floats. Divide by the tile's
// row width to get the next round's row offset: 64 M rows for A, 4 K rows for B.
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
                                                              int n) {
    // Each thread loads two float4 vectors from each input tile.
    for (int offset = 0; offset + A_ROW_STRIDE <= BM; offset += A_ROW_STRIDE) {
        // A already points to the current block/K slice. Select the local M row
        // with inner_row_a+offset, then the K-vector start with inner_col_a*4.
        const float4 a_values = *reinterpret_cast<const float4*>(A + (inner_row_a + offset) * k +
                                                                 inner_col_a * VECTOR_WIDTH);
        // Transpose A from [M, K] to [K, M]: shared index = local K * BM + local M.
        a_tile[(inner_col_a * VECTOR_WIDTH + 0) * BM + inner_row_a + offset] = a_values.x;
        a_tile[(inner_col_a * VECTOR_WIDTH + 1) * BM + inner_row_a + offset] = a_values.y;
        a_tile[(inner_col_a * VECTOR_WIDTH + 2) * BM + inner_row_a + offset] = a_values.z;
        a_tile[(inner_col_a * VECTOR_WIDTH + 3) * BM + inner_row_a + offset] = a_values.w;
    }

    // B stays [K, N]; inner_col_b counts float4 vectors, so multiply it by 4.
    for (int offset = 0; offset + B_ROW_STRIDE <= BK; offset += B_ROW_STRIDE) {
        // The selected K row has stride BN in shared memory and n in global memory.
        *reinterpret_cast<float4*>(b_tile + (inner_row_b + offset) * BN +
                                   inner_col_b * VECTOR_WIDTH) =
            *reinterpret_cast<const float4*>(B + (inner_row_b + offset) * n +
                                             inner_col_b * VECTOR_WIDTH);
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
        // A index: K plane + warp row + row subtile + lane's row tile.
        for (int warp_subrow = 0; warp_subrow < WMITER; ++warp_subrow) {
            for (int row = 0; row < THREAD_TILE_Y; ++row) {
                // Pack each row subtile's 8 A values consecutively in reg_m.
                reg_m[warp_subrow * THREAD_TILE_Y + row] =
                    a_tile[i * BM + warp_row * WM + warp_subrow * WSUBM +
                           thread_row * THREAD_TILE_Y + row];
            }
        }
        // B index: K row + warp column + column subtile + lane's column tile.
        for (int warp_subcolumn = 0; warp_subcolumn < WNITER; ++warp_subcolumn) {
            for (int column = 0; column < THREAD_TILE_X; ++column) {
                // Pack each column subtile's 4 B values consecutively in reg_n.
                reg_n[warp_subcolumn * THREAD_TILE_X + column] =
                    b_tile[i * BN + warp_col * WN + warp_subcolumn * WSUBN +
                           thread_col * THREAD_TILE_X + column];
            }
        }

        // Reuse the cached fragments in register outer products across subtiles.
        for (int warp_subrow = 0; warp_subrow < WMITER; ++warp_subrow) {
            for (int warp_subcolumn = 0; warp_subcolumn < WNITER; ++warp_subcolumn) {
                for (int row = 0; row < THREAD_TILE_Y; ++row) {
                    for (int column = 0; column < THREAD_TILE_X; ++column) {
                        // Flatten the lane's output: row within all row subtiles,
                        // then column within all column subtiles.
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

// Requires complete tiles and float4-aligned matrix row strides.
__global__ __launch_bounds__(NUM_THREADS) void sgemm(
    const float* A, const float* B, float* C, int k, int n) {
    __shared__ __align__(16) float a_tile[BM * BK];
    __shared__ __align__(16) float b_tile[BK * BN];

    const int tile_row = blockIdx.y * BM;
    const int tile_column = blockIdx.x * BN;

    // threadIdx.x/32 identifies the warp. Divide/modulo by BN/WN=2 warps per
    // output row maps that linear warp ID to the 2 x 2 warp-tile grid.
    const int warp_index = threadIdx.x / WARP_SIZE;
    const int warp_col = warp_index % (BN / WN);
    const int warp_row = warp_index / (BN / WN);

    // Within a 64 x 16 subtile, lane%4 selects a 4-column group and lane/4
    // selects an 8-row group; repeat this mapping for each warp subtile.
    const int lane_index = threadIdx.x % WARP_SIZE;
    const int thread_col = lane_index % (WSUBN / THREAD_TILE_X);
    const int thread_row = lane_index / (WSUBN / THREAD_TILE_X);

    // Divide/modulo by vectors per row gives (row, float4 column) for loads.
    // A maps 128 threads to 64 rows x 2 vectors; B maps them to 4 rows x 32 vectors.
    const int inner_row_a = threadIdx.x / A_VECTORS_PER_ROW;
    const int inner_col_a = threadIdx.x % A_VECTORS_PER_ROW;
    const int inner_row_b = threadIdx.x / B_VECTORS_PER_ROW;
    const int inner_col_b = threadIdx.x % B_VECTORS_PER_ROW;

    // Cache all subtile fragments: 128 output sums, 8 A values, and 16 B values.
    float sums[WMITER * THREAD_TILE_Y * WNITER * THREAD_TILE_X] = {0.0f};
    float reg_m[WMITER * THREAD_TILE_Y];
    float reg_n[WNITER * THREAD_TILE_X];

    // A starts at the block's first M row; B starts at its first N column.
    const float* tile_a = A + static_cast<std::size_t>(tile_row) * k;
    const float* tile_b = B + tile_column;
    for (int tile = 0; tile < k; tile += BK) {
        load_tiles_from_global_memory(tile_a,
                                      tile_b,
                                      a_tile,
                                      b_tile,
                                      inner_row_a,
                                      inner_col_a,
                                      inner_row_b,
                                      inner_col_b,
                                      k,
                                      n);
        __syncthreads();

        compute_warp_tile(
            sums, reg_m, reg_n, a_tile, b_tile, warp_row, warp_col, thread_row, thread_col);

        // Advance A by BK columns and B by BK full matrix rows to the next K slice.
        tile_a += BK;
        tile_b += static_cast<std::size_t>(BK) * n;
        __syncthreads();
    }

    // Warp origin in C: row=tile_row+warp_row*WM, column=tile_column+warp_col*WN.
    // Multiply only the row coordinate by the full matrix stride n.
    float* warp_output =
        C + static_cast<std::size_t>(tile_row + warp_row * WM) * n + tile_column + warp_col * WN;
    for (int warp_subrow = 0; warp_subrow < WMITER; ++warp_subrow) {
        for (int warp_subcolumn = 0; warp_subcolumn < WNITER; ++warp_subcolumn) {
            // Add the subtile's row offset (warp_subrow*WSUBM)*n and column offset.
            float* subtile_output = warp_output +
                                    static_cast<std::size_t>(warp_subrow * WSUBM) * n +
                                    warp_subcolumn * WSUBN;
            for (int row = 0; row < THREAD_TILE_Y; ++row) {
                for (int column = 0; column < THREAD_TILE_X; column += VECTOR_WIDTH) {
                    // Same flattened lane-local layout as accumulation; locate
                    // this row's four adjacent output columns for one float4.
                    const int sum_index =
                        (warp_subrow * THREAD_TILE_Y + row) * (WNITER * THREAD_TILE_X) +
                        warp_subcolumn * THREAD_TILE_X + column;
                    const float4 output_values = {sums[sum_index + 0],
                                                  sums[sum_index + 1],
                                                  sums[sum_index + 2],
                                                  sums[sum_index + 3]};
                    // Within the subtile, row=thread_row*8+row and column=thread_col*4+column.
                    *reinterpret_cast<float4*>(
                        subtile_output +
                        static_cast<std::size_t>(thread_row * THREAD_TILE_Y + row) * n +
                        thread_col * THREAD_TILE_X + column) = output_values;
                }
            }
        }
    }
}

}  // namespace sgemm_v5
