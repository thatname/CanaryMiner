// Pearl NoisyGEMM for Ampere and Ada (SM86, SM89): a multistage mma.sync int8
// kernel shaped after CUTLASS's Ampere GEMM rather than after the SM120 kernels.
//
// Same contract as noisyGemmPtx: A row-major (m x k), B n-major (n x k), one
// transcript of sixteen words per 16x16 tile, folded every 128 of k.
//
// What differs from noisyGemmPtx<4,4,4,2,64,3,8>, which ran at 86% of the
// tensor pipe on an RTX 3080 against 90% here (Nsight, clocks locked):
//
//   * 8 warps of 64x64 instead of 16 warps of 64x32: 32 mma per 8 ldmatrix.
//   * k128 stages XOR-swizzled instead of padded, so two fit in 96 KB and a
//     barrier comes once per 128 of k, which is also exactly one fold.
//   * Fragments double-buffered in registers: the ldmatrix for the next k32
//     step is issued before the mma of the current one.
//   * K is a compile-time constant and every copy row and fragment row of a
//     thread shares one swizzle phase, so each operand needs one base address
//     and the rest are immediates. Per-row pointers had pushed the kernel to
//     255 registers and a stack spill.
//
// Persistent (one CTA per SM). With one 98 KB CTA per SM, a CTA per tile left
// the tensor pipe idle while each new CTA waited for its first stage (4% of the
// samples) and while the old one stored its transcripts (1.4%). Each CTA now
// walks its tiles as one stream of k-blocks, so the next tile's first stage is
// in flight while the current one folds and stores: 91.6% -> 95.8% of the
// tensor pipe on an RTX 4090. The tiles are ordered so that the CTAs running
// together cover every m-block of a few B columns, which reads B from DRAM once
// per GEMM instead of once per kGroupM rows (kGroupM is no longer used).
#pragma once

#include <stdint.h>
#include "cuda_common_dim.h"
#include "noisy_gemm.cuh"

namespace om {
namespace pearl {
namespace ampere {

constexpr int kRank = 128;  // fold interval this kernel is specialised for

__device__ __forceinline__ void cpAsync16(uint32_t smem, const void *gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem));
}
// L2 policies: B (128 MB at n = 65536) streams through a 72 MB L2 once per GEMM and would evict the transcripts the
// scan reads right after it; marking B evict-first and the transcripts evict-last keeps the scan's 64 MB on chip.
__device__ __forceinline__ uint64_t l2EvictFirst() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}
__device__ __forceinline__ uint64_t l2EvictLast() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}
__device__ __forceinline__ void cpAsync16(uint32_t smem, const void *gmem, uint64_t policy) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(smem), "l"(gmem), "l"(policy));
}
__device__ __forceinline__ void storeKeep(uint32_t *p, uint32_t v, uint64_t policy) {
    asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n" ::"l"(p), "r"(v), "l"(policy) : "memory");
}
__device__ __forceinline__ void cpCommit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cpWait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void ldsm4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void imma(uint32_t d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/**
 * Swizzle phase of a stage row. Rows are kBK bytes, unpadded; 16-byte chunk c
 * of row r is stored at chunk c ^ phase(r), which puts the eight row addresses
 * of one ldmatrix on eight distinct bank quads. The phase repeats every eight
 * rows, which is what lets a thread's rows (16 or more apart) share one.
 */
__device__ __forceinline__ int swizzlePhase(int r) { return r & 7; }

template <int kWarpsM, int kWarpsN, int kTilesM, int kTilesN, int kBK, int kStages, int kGroupM, int kK>
__global__ __launch_bounds__(kWarpsM *kWarpsN * 32, 1) void noisyGemmAmpere(
    const int8_t *__restrict__ A, const int8_t *__restrict__ Bn, uint32_t *__restrict__ transcripts,
    int m, int n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int kThreads = kWarpsM * kWarpsN * 32;
    constexpr int kBlockM = kWarpsM * kTilesM * 16;
    constexpr int kBlockN = kWarpsN * kTilesN * 16;
    constexpr int kChunks = kBK / 16;                   // 16-byte chunks per row per stage
    constexpr int kSub = kBK / 32;                      // mma k-steps per stage
    constexpr int kStageA = kBlockM * kBK;
    constexpr int kStageBytes = (kBlockM + kBlockN) * kBK;
    constexpr int kRowsPerPass = kThreads / kChunks;    // rows one pass of the CTA's copies covers
    constexpr int kCopiesA = kBlockM / kRowsPerPass;
    constexpr int kCopiesB = kBlockN / kRowsPerPass;
    constexpr int kBlocksPerTile = kK / kBK;
    static_assert(kThreads % kChunks == 0 && kBlockM % kRowsPerPass == 0 &&
                  kBlockN % kRowsPerPass == 0, "copy work must split evenly");
    static_assert(kRowsPerPass % 8 == 0, "every copy row of a thread must share one swizzle phase");
    // One k-block per fold, and two stages so that a block's stage index is its parity within the tile.
    static_assert(kBK == kRank && kStages == 2 && kBlocksPerTile % 2 == 0, "k128 stages, two of them");

    extern __shared__ __align__(128) int8_t smem[];
    const uint32_t smemBase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpM = warp / kWarpsN;
    const int warpN = warp % kWarpsN;

    // CTA c takes tiles c, c + grid, ...; tile L covers m-block L % blocksM of column L / blocksM, so the CTAs
    // running together share a few B columns between all their m-blocks.
    const int blocksM = m / kBlockM;
    const int tiles = blocksM * (n / kBlockN);
    const int myTiles = (tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
    if (myTiles <= 0) return;
    auto origin = [&](int t, int *row, int *col) {
        const int linear = (int)blockIdx.x + t * (int)gridDim.x;
        *row = (linear % blocksM) * kBlockM;
        *col = (linear / blocksM) * kBlockN;
    };

    // Copy role: rows r0 + i * kRowsPerPass, logical chunk c0.
    const int r0 = threadIdx.x / kChunks, c0 = threadIdx.x % kChunks;
    const uint32_t sCopy = smemBase + r0 * kBK + ((c0 ^ swizzlePhase(r0)) << 4);
    const uint64_t streamB = l2EvictFirst();
    auto issue = [&](int row, int col, int blk, int stage) {
        const uint32_t st = sCopy + stage * kStageBytes;
        const int8_t *const gA = A + (size_t)(row + r0) * kK + blk * kBK + c0 * 16;
        const int8_t *const gB = Bn + (size_t)(col + r0) * kK + blk * kBK + c0 * 16;
#pragma unroll
        for (int i = 0; i < kCopiesA; i++)
            cpAsync16(st + i * kRowsPerPass * kBK, gA + (size_t)i * kRowsPerPass * kK);
#pragma unroll
        for (int i = 0; i < kCopiesB; i++)
            cpAsync16(st + kStageA + i * kRowsPerPass * kBK, gB + (size_t)i * kRowsPerPass * kK, streamB);
    };

    // ldmatrix lane roles. A: (rows 0-7, k lo), (rows 8-15, k lo), (rows 0-7, k hi), (rows 8-15, k hi).
    // B, n-major: (cols 0-7, k lo), (cols 0-7, k hi), (cols 8-15, k lo), (cols 8-15, k hi), so each
    // column half's two registers come back adjacent.
    const int g = lane >> 3;
    const int aRow = warpM * kTilesM * 16 + (lane & 7) + (g & 1) * 8;
    const int bRow = warpN * kTilesN * 16 + (lane & 7) + (g >> 1) * 8;
    const uint32_t aBase = smemBase + aRow * kBK;
    const uint32_t bBase = smemBase + kStageA + bRow * kBK;
    // Logical chunk is sub * 2 + half; bit 0 is the half, so the swizzled chunk is (sub * 2) ^ q.
    const int aQ = (g >> 1) ^ swizzlePhase(aRow);
    const int bQ = (g & 1) ^ swizzlePhase(bRow);
    auto loadFrags = [&](uint32_t fa[kTilesM][4], uint32_t fb[kTilesN][4], int stage, int sub) {
        const uint32_t aAddr = aBase + stage * kStageBytes + (((sub * 2) ^ aQ) << 4);
        const uint32_t bAddr = bBase + stage * kStageBytes + (((sub * 2) ^ bQ) << 4);
#pragma unroll
        for (int i = 0; i < kTilesM; i++) ldsm4(fa[i], aAddr + i * 16 * kBK);
#pragma unroll
        for (int j = 0; j < kTilesN; j++) ldsm4(fb[j], bAddr + j * 16 * kBK);
    };

    uint32_t acc[kTilesM][kTilesN][2][4];
    uint32_t slot[kTilesM][kTilesN];
    uint32_t fa[2][kTilesM][4], fb[2][kTilesN][4];

    int row, col;
    origin(0, &row, &col);
    issue(row, col, 0, 0);
    cpCommit();
    cpWait0();
    __syncthreads();
    loadFrags(fa[0], fb[0], 0, 0);

#pragma unroll 1
    for (int t = 0; t < myTiles; t++) {
        const bool more = t + 1 < myTiles;
        int nextRow = row, nextCol = col;
        if (more) origin(t + 1, &nextRow, &nextCol);
#pragma unroll
        for (int i = 0; i < kTilesM; i++)
#pragma unroll
            for (int j = 0; j < kTilesN; j++) {
                slot[i][j] = 0;
#pragma unroll
                for (int h = 0; h < 2; h++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[i][j][h][e] = 0;
            }

#pragma unroll 1
        for (int pair = 0; pair < kBlocksPerTile; pair += 2) {
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int blk = pair + u;  // also the fold, and its parity is its stage
                const bool lastBlock = blk + 1 == kBlocksPerTile;
                // The stage written here was last read before the barrier that ended the previous block. The
                // block after a tile's last one is the next tile's first, so its loads overlap this tile's end.
                if (!lastBlock) issue(row, col, blk + 1, u ^ 1);
                else if (more) issue(nextRow, nextCol, 0, u ^ 1);
                cpCommit();
#pragma unroll
                for (int sub = 0; sub < kSub; sub++) {
                    const int cur = (u * kSub + sub) & 1;
                    if (sub == kSub - 1) {
                        // The next block's stage must have landed and every warp must be done with this one.
                        cpWait0();
                        __syncthreads();
                        if (!lastBlock || more) loadFrags(fa[cur ^ 1], fb[cur ^ 1], u ^ 1, 0);
                    } else {
                        loadFrags(fa[cur ^ 1], fb[cur ^ 1], u, sub + 1);
                    }
#pragma unroll
                    for (int i = 0; i < kTilesM; i++)
#pragma unroll
                        for (int j = 0; j < kTilesN; j++) {
                            imma(acc[i][j][0], fa[cur][i], fb[cur][j][0], fb[cur][j][1]);
                            imma(acc[i][j][1], fa[cur][i], fb[cur][j][2], fb[cur][j][3]);
                        }
                }
                // XOR the tile's 256 accumulators into transcript word `blk % 16`, owned by that lane.
#pragma unroll
                for (int i = 0; i < kTilesM; i++)
#pragma unroll
                    for (int j = 0; j < kTilesN; j++) {
                        const uint32_t v = warpXor(
                            xor3(xor3(acc[i][j][0][0], acc[i][j][0][1], acc[i][j][0][2]),
                                 xor3(acc[i][j][0][3], acc[i][j][1][0], acc[i][j][1][1]),
                                 acc[i][j][1][2] ^ acc[i][j][1][3]));
                        if (lane == (blk & (kTranscriptU32 - 1)))
                            slot[i][j] = rotl32(slot[i][j], kRotation) ^ v;
                    }
            }
        }

        if (lane < kTranscriptU32) {
            const uint64_t keep = l2EvictLast();
            constexpr int tps = kRank / kHashTile;
            const int blocksPerRow = n / kRank;
#pragma unroll
            for (int i = 0; i < kTilesM; i++)
#pragma unroll
                for (int j = 0; j < kTilesN; j++) {
                    const int r = row + (warpM * kTilesM + i) * kHashTile;
                    const int c = col + (warpN * kTilesN + j) * kHashTile;
                    const size_t flat = ((size_t)((r / kRank) * blocksPerRow + c / kRank) * tps +
                                         (r % kRank) / kHashTile) * tps + (c % kRank) / kHashTile;
                    storeKeep(transcripts + flat * kTranscriptU32 + lane, slot[i][j], keep);
                }
        }
        row = nextRow;
        col = nextCol;
    }
    cpWait0();
#endif
}

/**
 * Catalog launcher (TileConfig signature). The caller's grid is (n / blockN, m / blockM), one entry per tile;
 * the kernel runs persistent, one CTA per SM, and orders the tiles itself. Refuses any k but the build's common
 * dimension, any rank but 128 and inexact tiling, so a caller with another geometry gets an error instead of a
 * wrong transcript.
 */
template <int WM, int WN, int TM, int TN, int BK, int ST, int GM>
cudaError_t launch(dim3 grid, int threads, cudaStream_t s, const int8_t *a, const int8_t *b,
                   uint32_t *t, int m, int n, int k, int rank) {
    constexpr int kK = CANARY_PEARL_EXPERIMENT_COMMON_DIM;
    constexpr int bm = WM * TM * 16, bn = WN * TN * 16;
    if (k != kK || rank != kRank || m % bm || n % bn || m % kRank || n % kRank ||
        threads != WM * WN * 32 || grid.x * grid.y != unsigned((m / bm) * (n / bn)))
        return cudaErrorInvalidValue;
    constexpr int smem = ST * (bm + bn) * BK;
    auto fn = noisyGemmAmpere<WM, WN, TM, TN, BK, ST, GM, kK>;
    int device = 0, sms = 0;
    cudaError_t e = cudaGetDevice(&device);
    if (e == cudaSuccess) e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return e;
    const unsigned tiles = grid.x * grid.y;
    fn<<<sms > 0 && unsigned(sms) < tiles ? unsigned(sms) : tiles, threads, smem, s>>>(a, b, t, m, n);
    return cudaGetLastError();
}

}  // namespace ampere
}  // namespace pearl
}  // namespace om
