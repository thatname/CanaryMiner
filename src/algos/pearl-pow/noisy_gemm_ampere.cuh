// Pearl NoisyGEMM for Ampere (SM86): a multistage mma.sync int8 kernel shaped
// after CUTLASS's Ampere GEMM rather than after the SM120 kernels.
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
__device__ __forceinline__ void cpCommit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cpWait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

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

__host__ __device__ constexpr int gcd(int a, int b) { return b ? gcd(b, a % b) : a; }

/**
 * Swizzle phase of a stage row. Rows are kBK bytes, unpadded; 16-byte chunk c
 * of row r is stored at chunk c ^ phase(r), which puts the eight row addresses
 * of one ldmatrix on eight distinct bank quads. Both phases repeat every eight
 * rows, which is what lets a thread's rows (16 or more apart) share one.
 */
template <int kBK>
__device__ __forceinline__ int swizzlePhase(int r) {
    static_assert(kBK == 64 || kBK == 128, "BK must be 64 or 128");
    return kBK == 128 ? (r & 7) : ((r >> 1) & 3);
}

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
    static_assert(kThreads % kChunks == 0 && kBlockM % kRowsPerPass == 0 &&
                  kBlockN % kRowsPerPass == 0, "copy work must split evenly");
    static_assert(kRowsPerPass % 8 == 0, "every copy row of a thread must share one swizzle phase");
    static_assert(kRank % kBK == 0 && kK % kRank == 0, "a stage must not straddle a fold");
    constexpr int kBlocksPerFold = kRank / kBK;
    constexpr int kTotalBlocks = kK / kBK;
    // Unrolling by lcm(stages, blocks per fold) makes both the stage index and the fold point compile-time.
    constexpr int kUnroll = kStages * kBlocksPerFold / gcd(kStages, kBlocksPerFold);
    static_assert(kTotalBlocks % kUnroll == 0 && kTotalBlocks >= kStages, "unrolled groups must tile k");

    extern __shared__ __align__(128) int8_t smem[];
    const uint32_t smemBase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpM = warp / kWarpsN;
    const int warpN = warp % kWarpsN;

    // Grouped rasterisation: kGroupM consecutive blocks walk down m sharing one B tile.
    const int blocksN = n / kBlockN, blocksM = m / kBlockM;
    const int linear = blockIdx.x + blockIdx.y * gridDim.x;  // 1D or the catalog's 2D grid
    const int perGroup = kGroupM * blocksN;
    const int mFirst = (linear / perGroup) * kGroupM;
    const int inGroup = linear % perGroup;
    const int groupRows = min(kGroupM, blocksM - mFirst);
    const int rowBase = (mFirst + inGroup % groupRows) * kBlockM;
    const int colBase = (inGroup / groupRows) * kBlockN;

    // Copy role: rows r0 + i * kRowsPerPass, logical chunk c0.
    const int r0 = threadIdx.x / kChunks, c0 = threadIdx.x % kChunks;
    const uint32_t sCopy = smemBase + r0 * kBK + ((c0 ^ swizzlePhase<kBK>(r0)) << 4);
    const int8_t *const gA = A + (size_t)(rowBase + r0) * kK + c0 * 16;
    const int8_t *const gB = Bn + (size_t)(colBase + r0) * kK + c0 * 16;
    auto issue = [&](int blk, int stage) {
        const uint32_t st = sCopy + stage * kStageBytes;
        const int kOff = blk * kBK;
#pragma unroll
        for (int i = 0; i < kCopiesA; i++)
            cpAsync16(st + i * kRowsPerPass * kBK, gA + (size_t)i * kRowsPerPass * kK + kOff);
#pragma unroll
        for (int i = 0; i < kCopiesB; i++)
            cpAsync16(st + kStageA + i * kRowsPerPass * kBK, gB + (size_t)i * kRowsPerPass * kK + kOff);
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
    const int aQ = (g >> 1) ^ swizzlePhase<kBK>(aRow);
    const int bQ = (g & 1) ^ swizzlePhase<kBK>(bRow);
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

#pragma unroll
    for (int s = 0; s < kStages - 1; s++) {
        issue(s, s);
        cpCommit();
    }
    cpWait<kStages - 2>();
    __syncthreads();

    uint32_t fa[2][kTilesM][4], fb[2][kTilesN][4];
    loadFrags(fa[0], fb[0], 0, 0);

    int fold = 0;
#pragma unroll 1
    for (int group = 0; group < kTotalBlocks; group += kUnroll) {
#pragma unroll
        for (int u = 0; u < kUnroll; u++) {
            const int blk = group + u;
            // The stage written here was last read before the barrier that ended the previous block. Every
            // block commits a group, empty at the tail, so one constant wait depth stays right to the end.
            if (blk + kStages - 1 < kTotalBlocks) issue(blk + kStages - 1, (u + kStages - 1) % kStages);
            cpCommit();
#pragma unroll
            for (int sub = 0; sub < kSub; sub++) {
                const int cur = (u * kSub + sub) & 1;
                if (sub == kSub - 1) {
                    // The next block's stage must have landed and every warp must be done with this one.
                    cpWait<kStages - 2>();
                    __syncthreads();
                    if (blk + 1 < kTotalBlocks) loadFrags(fa[cur ^ 1], fb[cur ^ 1], (u + 1) % kStages, 0);
                } else {
                    loadFrags(fa[cur ^ 1], fb[cur ^ 1], u % kStages, sub + 1);
                }
#pragma unroll
                for (int i = 0; i < kTilesM; i++)
#pragma unroll
                    for (int j = 0; j < kTilesN; j++) {
                        imma(acc[i][j][0], fa[cur][i], fb[cur][j][0], fb[cur][j][1]);
                        imma(acc[i][j][1], fa[cur][i], fb[cur][j][2], fb[cur][j][3]);
                    }
            }
            if ((u + 1) % kBlocksPerFold == 0) {
                // XOR the tile's 256 accumulators into transcript word `fold % 16`, owned by that lane.
#pragma unroll
                for (int i = 0; i < kTilesM; i++)
#pragma unroll
                    for (int j = 0; j < kTilesN; j++) {
                        const uint32_t v = warpXor(
                            xor3(xor3(acc[i][j][0][0], acc[i][j][0][1], acc[i][j][0][2]),
                                 xor3(acc[i][j][0][3], acc[i][j][1][0], acc[i][j][1][1]),
                                 acc[i][j][1][2] ^ acc[i][j][1][3]));
                        if (lane == (fold & (kTranscriptU32 - 1)))
                            slot[i][j] = rotl32(slot[i][j], kRotation) ^ v;
                    }
                ++fold;
            }
        }
    }
    cpWait<0>();

    if (lane < kTranscriptU32) {
        constexpr int tps = kRank / kHashTile;
        const int blocksPerRow = n / kRank;
#pragma unroll
        for (int i = 0; i < kTilesM; i++)
#pragma unroll
            for (int j = 0; j < kTilesN; j++) {
                const int row = rowBase + (warpM * kTilesM + i) * kHashTile;
                const int col = colBase + (warpN * kTilesN + j) * kHashTile;
                const size_t flat = ((size_t)((row / kRank) * blocksPerRow + col / kRank) * tps +
                                     (row % kRank) / kHashTile) * tps + (col % kRank) / kHashTile;
                transcripts[flat * kTranscriptU32 + lane] = slot[i][j];
            }
    }
#endif
}

/**
 * Catalog launcher (TileConfig signature). The grid is the caller's
 * (n / blockN, m / blockM); the kernel linearises it and applies its own
 * grouped rasterisation. Refuses any k but the build's common dimension, any
 * rank but 128 and inexact tiling, so a caller with another geometry gets an
 * error instead of a wrong transcript.
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
    cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return e;
    fn<<<grid, threads, smem, s>>>(a, b, t, m, n);
    return cudaGetLastError();
}

}  // namespace ampere
}  // namespace pearl
}  // namespace om
