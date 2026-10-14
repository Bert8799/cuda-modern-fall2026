#pragma once

namespace sgemm_v3 {

// Sizes of the tiles staged in shared memory: BM x BK for A, BK x BN for B,
// and BM x BN for the output tile a block owns.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

// Each thread owns a 2D output tile with THREAD_TILE_Y rows and THREAD_TILE_X
// columns, and accumulates all values in that tile.
constexpr int THREAD_TILE_X = 8;
constexpr int THREAD_TILE_Y = 8;
constexpr int BLOCK_X = BN / THREAD_TILE_X;
constexpr int BLOCK_Y = BM / THREAD_TILE_Y;

static_assert(BLOCK_Y * THREAD_TILE_Y == BM && BLOCK_X * THREAD_TILE_X == BN,
              "the thread tile must tile the output tile exactly");

// Each block stages A and B tiles in shared memory, then reuses those values
// while its threads accumulate the corresponding BM x BN output tile.
__global__ void sgemm(const float* A, const float* B, float* C, int m, int k, int n) {
    __shared__ float a_tile[BM * BK];
    __shared__ float b_tile[BK * BN];

    const int tile_row = blockIdx.y * BM;
    const int tile_column = blockIdx.x * BN;
    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;

    // Threads own outputs at rows ty, ty + BLOCK_Y, ... and columns tx,
    // tx + BLOCK_X, ... of the block's output tile.
    float sums[THREAD_TILE_Y][THREAD_TILE_X];
    for (int row = 0; row < THREAD_TILE_Y; row++) {
        for (int column = 0; column < THREAD_TILE_X; column++) {
            sums[row][column] = 0.0f;
        }
    }

    for (int tile = 0; tile < k; tile += BK) {
        // Load the A and B tiles cooperatively, padding incomplete edge tiles
        // with zero so the same kernel supports dimensions not divisible by
        // the tile sizes.
        // Each warp writes 32 consecutive floats in each shared tile, so the
        // cooperative stores touch distinct banks.
        for (int index = tid; index < BM * BK; index += BLOCK_X * BLOCK_Y) {
            const int row = tile_row + index / BK;
            const int column = tile + index % BK;
            a_tile[index] = row < m && column < k ? A[row * k + column] : 0.0f;
        }
        for (int index = tid; index < BK * BN; index += BLOCK_X * BLOCK_Y) {
            const int row = tile + index / BN;
            const int column = tile_column + index % BN;
            b_tile[index] = row < k && column < n ? B[row * n + column] : 0.0f;
        }
        __syncthreads();

        // Accumulate this K slice into every output owned by the thread.
        // With BLOCK_X=16 and BK=8, each warp spans two adjacent threadIdx.y
        // values. A is broadcast within each 16-lane group; the two addresses
        // differ by 8 words and therefore map to different banks.
        // B varies across 16 consecutive threadIdx.x values (distinct banks),
        // while the other 16 lanes repeat those addresses and use broadcast.
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
            const int output_row = tile_row + threadIdx.y + row * BLOCK_Y;
            const int output_column = tile_column + threadIdx.x + column * BLOCK_X;
            if (output_row < m && output_column < n) {
                C[output_row * n + output_column] = sums[row][column];
            }
        }
    }
}

}  // namespace sgemm_v3
