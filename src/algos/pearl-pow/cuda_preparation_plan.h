// Single-source, host-only metadata for the CUDA Pearl launch catalog.
// No CUDA types or runtime calls belong here. The CUDA factory consumes these
// descriptors directly; plan tokens are permanent names, never array indexes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string_view>

#include "job_publication.h"
#include "cuda_common_dim.h"
#include "tile_arch.h"

namespace om::pearl {

enum TileFamily : uint8_t { kSingle, kDoubleBuffered, kAsync, kPtx };

struct CudaPreparationShapeDescriptor {
    const char *id;
    uint32_t m, n;
};

// Exact legacy tuner shape order. Sharing these records with algo.cu keeps
// request admission tied to the shapes its existing kernels actually use.
// Larger shapes remain opt-in until GPU correctness and performance qualify.
#if defined(CANARY_PEARL_EXPERIMENT_LARGE_SHAPES)
static_assert(kCudaCommonDim == 8192, "large-shape experiment requires K8192");
#endif
inline constexpr CudaPreparationShapeDescriptor kCudaPreparationShapes[] = {
#if defined(CANARY_PEARL_EXPERIMENT_LARGE_SHAPES)
    {"m4096-n131072", 4096, 131072},
    {"m16384-n131072", 16384, 131072},
    {"m131072-n131072", 131072, 131072},
#endif
#if defined(CANARY_PEARL_AMPERE_BUILD)
    // SM86 at K2048: wider n amortises the per-attempt A preparation further.
    {"m4096-n131072", 4096, 131072},
    {"m4096-n262144", 4096, 262144},
#endif
    {"m4096-n65536", 4096, 65536},
    {"m8192-n16384", 8192, 16384},
    {"m4096-n32768", 4096, 32768},
    {"m4096-n16384", 4096, 16384},
    {"m2048-n65536", 2048, 65536},
    {"m2048-n32768", 2048, 32768},
    {"m2048-n16384", 2048, 16384},
    {"m2048-n8192", 2048, 8192},
    {"m1024-n32768", 1024, 32768},
    {"m1024-n16384", 1024, 16384},
    {"m512-n8192", 512, 8192},
    {"m256-n2048", 256, 2048},
};

inline constexpr size_t kCudaPreparationShapeCount =
    sizeof(kCudaPreparationShapes) / sizeof(kCudaPreparationShapes[0]);

constexpr const CudaPreparationShapeDescriptor *findCudaPreparationShape(
    uint32_t m, uint32_t n) noexcept {
    for (const auto &shape : kCudaPreparationShapes)
        if (shape.m == m && shape.n == n) return &shape;
    return nullptr;
}

// The upper word names CUDA; the next 16 bits version the device fixture ABI.
// Bump the ABI for changed preparation bytes/layout/derivation semantics.
// Never renumber or reuse a token when catalog rows move or disappear.
inline constexpr uint64_t kCudaPreparationPlanNamespace =
    UINT64_C(0x4355444100000000);
inline constexpr uint16_t kCudaPreparationFixtureAbi = 1;

enum class CudaPreparationPlanToken : uint16_t {
    SingleWm2Wn4Tm2Tn2 = 0x0001,
    SingleWm2Wn4Tm4Tn2 = 0x0002,
    SingleWm2Wn4Tm2Tn4 = 0x0003,
    SingleWm2Wn4Tm4Tn4 = 0x0004,
    SingleWm4Wn2Tm2Tn2 = 0x0005,
    SingleWm4Wn4Tm2Tn2 = 0x0006,
    DoubleBufferedWm2Wn4Tm2Tn2 = 0x0007,
    DoubleBufferedWm2Wn4Tm4Tn2 = 0x0008,
    DoubleBufferedWm2Wn4Tm2Tn4 = 0x0009,
    DoubleBufferedWm2Wn4Tm4Tn4 = 0x000a,
    DoubleBufferedWm4Wn2Tm2Tn2 = 0x000b,
    DoubleBufferedWm4Wn4Tm2Tn2 = 0x000c,
    AsyncWm2Wn4Tm2Tn2 = 0x000d,
    AsyncWm2Wn4Tm4Tn2 = 0x000e,
    AsyncWm2Wn4Tm2Tn4 = 0x000f,
    AsyncWm2Wn4Tm4Tn4 = 0x0010,
    AsyncWm4Wn2Tm2Tn2 = 0x0011,
    AsyncWm4Wn4Tm2Tn2 = 0x0012,
    PtxBk64Wm2Wn4Tm2Tn2 = 0x0013,
    PtxBk64Wm2Wn4Tm4Tn2 = 0x0014,
    PtxBk64Wm2Wn4Tm2Tn4 = 0x0015,
    PtxBk64Wm2Wn4Tm4Tn4 = 0x0016,
    PtxBk64Wm4Wn2Tm2Tn2 = 0x0017,
    PtxBk64Wm4Wn4Tm2Tn2 = 0x0018,
    PtxBk64Wm4Wn4Tm4Tn2 = 0x0019,
    PtxBk64Wm4Wn4Tm2Tn4 = 0x001a,
    PtxBk64Wm8Wn2Tm2Tn2 = 0x001b,
    PtxBk64Wm4Wn8Tm2Tn2 = 0x001c,
    PtxBk32Wm2Wn4Tm2Tn2 = 0x001d,
    PtxBk32Wm2Wn4Tm4Tn2 = 0x001e,
    PtxBk32Wm2Wn4Tm2Tn4 = 0x001f,
    PtxBk32Wm2Wn4Tm4Tn4 = 0x0020,
    PtxBk32Wm4Wn2Tm2Tn2 = 0x0021,
    PtxBk32Wm4Wn4Tm2Tn2 = 0x0022,
    PtxBk32Wm4Wn4Tm4Tn2 = 0x0023,
    PtxBk32Wm4Wn4Tm2Tn4 = 0x0024,
    PtxBk32Wm8Wn2Tm2Tn2 = 0x0025,
    PtxBk32Wm4Wn8Tm2Tn2 = 0x0026,
    ExperimentalSm120FixedK2048R128 = 0x0027,
    ExperimentalSm120Wm8Runtime = 0x0028,
    ExperimentalSm120Wm8InlineCap112 = 0x0029,
    ExperimentalSm120Wm8FixedK2048R128 = 0x002a,
    ExperimentalSm120Bk32FixedK2048R128 = 0x002c,
    ExperimentalSm120Wm4ReuseFixedK2048R128 = 0x002d,
    ExperimentalSm120TmaWm8FixedK2048R128 = 0x002e,
    ExperimentalSm120TmaTraversal16x32 = 0x002f,
    ExperimentalSm120TmaTraversal16x32Stage4 = 0x0031,
    ExperimentalSm120TmaDeliveryK8192 = 0x0032,
    AmpereBk128Wm2Wn4Tm4Tn4 = 0x0033,
};

constexpr uint64_t cudaPreparationPlanId(
    CudaPreparationPlanToken token) noexcept {
    return kCudaPreparationPlanNamespace |
           (uint64_t{kCudaPreparationFixtureAbi} << 16) |
           static_cast<uint16_t>(token);
}

struct CudaPreparationPlanDescriptor {
    CudaPreparationPlanToken token;
    uint64_t preparationPlanId;
    const char *name;
    const char *kernelId;
    const char *tileId;
    TileFamily family;
    int wm, wn, tm, tn;
    int blockM, blockN, threads;
    int blockK;
    int stages, groupM;
    bool nMajorB;
    TileArchEligibility architecture;
};

constexpr CudaPreparationPlanDescriptor makeCudaPreparationPlan(
    CudaPreparationPlanToken token, const char *name, const char *kernelId,
    const char *tileId, TileFamily family, int wm, int wn, int tm, int tn,
    int blockK, int stages, int groupM,
    TileArchEligibility architecture) noexcept {
    return {token, cudaPreparationPlanId(token), name, kernelId, tileId, family,
            wm, wn, tm, tn, wm * tm * 16, wn * tn * 16, wm * wn * 32,
            blockK, stages, groupM, family == kPtx, architecture};
}

// Order deliberately preserves the legacy CUDA tuner order. IDs do not depend
// on this order. Metadata and template geometry have one source of truth.
inline constexpr CudaPreparationPlanDescriptor kCudaPreparationPlans[] = {
    makeCudaPreparationPlan(CudaPreparationPlanToken::SingleWm2Wn4Tm2Tn2,
        "single 2x4/2x2", "cuda-wmma-single-v1",
        "wm2-wn4-tm2-tn2-bk32", kSingle, 2, 4, 2, 2,
        32, 1, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::SingleWm2Wn4Tm4Tn2,
        "single 2x4/4x2", "cuda-wmma-single-v1",
        "wm2-wn4-tm4-tn2-bk32", kSingle, 2, 4, 4, 2,
        32, 1, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::SingleWm2Wn4Tm2Tn4,
        "single 2x4/2x4", "cuda-wmma-single-v1",
        "wm2-wn4-tm2-tn4-bk32", kSingle, 2, 4, 2, 4,
        32, 1, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::SingleWm2Wn4Tm4Tn4,
        "single 2x4/4x4", "cuda-wmma-single-v1",
        "wm2-wn4-tm4-tn4-bk32", kSingle, 2, 4, 4, 4,
        32, 1, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::SingleWm4Wn2Tm2Tn2,
        "single 4x2/2x2", "cuda-wmma-single-v1",
        "wm4-wn2-tm2-tn2-bk32", kSingle, 4, 2, 2, 2,
        32, 1, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::SingleWm4Wn4Tm2Tn2,
        "single 4x4/2x2", "cuda-wmma-single-v1",
        "wm4-wn4-tm2-tn2-bk32", kSingle, 4, 4, 2, 2,
        32, 1, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::DoubleBufferedWm2Wn4Tm2Tn2,
        "dbuf 2x4/2x2", "cuda-wmma-dbuf-v1",
        "wm2-wn4-tm2-tn2-bk32", kDoubleBuffered, 2, 4, 2, 2,
        32, 2, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::DoubleBufferedWm2Wn4Tm4Tn2,
        "dbuf 2x4/4x2", "cuda-wmma-dbuf-v1",
        "wm2-wn4-tm4-tn2-bk32", kDoubleBuffered, 2, 4, 4, 2,
        32, 2, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::DoubleBufferedWm2Wn4Tm2Tn4,
        "dbuf 2x4/2x4", "cuda-wmma-dbuf-v1",
        "wm2-wn4-tm2-tn4-bk32", kDoubleBuffered, 2, 4, 2, 4,
        32, 2, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::DoubleBufferedWm2Wn4Tm4Tn4,
        "dbuf 2x4/4x4", "cuda-wmma-dbuf-v1",
        "wm2-wn4-tm4-tn4-bk32", kDoubleBuffered, 2, 4, 4, 4,
        32, 2, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::DoubleBufferedWm4Wn2Tm2Tn2,
        "dbuf 4x2/2x2", "cuda-wmma-dbuf-v1",
        "wm4-wn2-tm2-tn2-bk32", kDoubleBuffered, 4, 2, 2, 2,
        32, 2, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::DoubleBufferedWm4Wn4Tm2Tn2,
        "dbuf 4x4/2x2", "cuda-wmma-dbuf-v1",
        "wm4-wn4-tm2-tn2-bk32", kDoubleBuffered, 4, 4, 2, 2,
        32, 2, 0, minimumTileArchitecture(70)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::AsyncWm2Wn4Tm2Tn2,
        "async 2x4/2x2", "cuda-wmma-async-v1",
        "wm2-wn4-tm2-tn2-bk32", kAsync, 2, 4, 2, 2,
        32, 4, 0, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::AsyncWm2Wn4Tm4Tn2,
        "async 2x4/4x2", "cuda-wmma-async-v1",
        "wm2-wn4-tm4-tn2-bk32", kAsync, 2, 4, 4, 2,
        32, 4, 0, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::AsyncWm2Wn4Tm2Tn4,
        "async 2x4/2x4", "cuda-wmma-async-v1",
        "wm2-wn4-tm2-tn4-bk32", kAsync, 2, 4, 2, 4,
        32, 4, 0, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::AsyncWm2Wn4Tm4Tn4,
        "async 2x4/4x4", "cuda-wmma-async-v1",
        "wm2-wn4-tm4-tn4-bk32", kAsync, 2, 4, 4, 4,
        32, 4, 0, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::AsyncWm4Wn2Tm2Tn2,
        "async 4x2/2x2", "cuda-wmma-async-v1",
        "wm4-wn2-tm2-tn2-bk32", kAsync, 4, 2, 2, 2,
        32, 4, 0, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::AsyncWm4Wn4Tm2Tn2,
        "async 4x4/2x2", "cuda-wmma-async-v1",
        "wm4-wn4-tm2-tn2-bk32", kAsync, 4, 4, 2, 2,
        32, 4, 0, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk64Wm2Wn4Tm2Tn2,
        "ptx k64 2x4/2x2", "cuda-ptx-mma-s8-v1",
        "wm2-wn4-tm2-tn2-bk64-st3-gm8", kPtx, 2, 4, 2, 2,
        64, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk64Wm2Wn4Tm4Tn2,
        "ptx k64 2x4/4x2", "cuda-ptx-mma-s8-v1",
        "wm2-wn4-tm4-tn2-bk64-st3-gm8", kPtx, 2, 4, 4, 2,
        64, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk64Wm2Wn4Tm2Tn4,
        "ptx k64 2x4/2x4", "cuda-ptx-mma-s8-v1",
        "wm2-wn4-tm2-tn4-bk64-st3-gm8", kPtx, 2, 4, 2, 4,
        64, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk64Wm2Wn4Tm4Tn4,
        "ptx k64 2x4/4x4", "cuda-ptx-mma-s8-v1",
        "wm2-wn4-tm4-tn4-bk64-st3-gm8", kPtx, 2, 4, 4, 4,
        64, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk64Wm4Wn2Tm2Tn2,
        "ptx k64 4x2/2x2", "cuda-ptx-mma-s8-v1",
        "wm4-wn2-tm2-tn2-bk64-st3-gm8", kPtx, 4, 2, 2, 2,
        64, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk64Wm4Wn4Tm2Tn2,
        "ptx k64 4x4/2x2", "cuda-ptx-mma-s8-v1",
        "wm4-wn4-tm2-tn2-bk64-st3-gm8", kPtx, 4, 4, 2, 2,
        64, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk64Wm4Wn4Tm4Tn2,
        "ptx k64 4x4/4x2", "cuda-ptx-mma-s8-v1",
        "wm4-wn4-tm4-tn2-bk64-st3-gm8", kPtx, 4, 4, 4, 2,
        64, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk64Wm4Wn4Tm2Tn4,
        "ptx k64 4x4/2x4", "cuda-ptx-mma-s8-v1",
        "wm4-wn4-tm2-tn4-bk64-st3-gm8", kPtx, 4, 4, 2, 4,
        64, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk64Wm8Wn2Tm2Tn2,
        "ptx k64 8x2/2x2", "cuda-ptx-mma-s8-v1",
        "wm8-wn2-tm2-tn2-bk64-st3-gm8", kPtx, 8, 2, 2, 2,
        64, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk64Wm4Wn8Tm2Tn2,
        "ptx k64 4x8/2x2", "cuda-ptx-mma-s8-v1",
        "wm4-wn8-tm2-tn2-bk64-st3-gm8", kPtx, 4, 8, 2, 2,
        64, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk32Wm2Wn4Tm2Tn2,
        "ptx k32 2x4/2x2", "cuda-ptx-mma-s8-v1",
        "wm2-wn4-tm2-tn2-bk32-st3-gm8", kPtx, 2, 4, 2, 2,
        32, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk32Wm2Wn4Tm4Tn2,
        "ptx k32 2x4/4x2", "cuda-ptx-mma-s8-v1",
        "wm2-wn4-tm4-tn2-bk32-st3-gm8", kPtx, 2, 4, 4, 2,
        32, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk32Wm2Wn4Tm2Tn4,
        "ptx k32 2x4/2x4", "cuda-ptx-mma-s8-v1",
        "wm2-wn4-tm2-tn4-bk32-st3-gm8", kPtx, 2, 4, 2, 4,
        32, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk32Wm2Wn4Tm4Tn4,
        "ptx k32 2x4/4x4", "cuda-ptx-mma-s8-v1",
        "wm2-wn4-tm4-tn4-bk32-st3-gm8", kPtx, 2, 4, 4, 4,
        32, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk32Wm4Wn2Tm2Tn2,
        "ptx k32 4x2/2x2", "cuda-ptx-mma-s8-v1",
        "wm4-wn2-tm2-tn2-bk32-st3-gm8", kPtx, 4, 2, 2, 2,
        32, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk32Wm4Wn4Tm2Tn2,
        "ptx k32 4x4/2x2", "cuda-ptx-mma-s8-v1",
        "wm4-wn4-tm2-tn2-bk32-st3-gm8", kPtx, 4, 4, 2, 2,
        32, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk32Wm4Wn4Tm4Tn2,
        "ptx k32 4x4/4x2", "cuda-ptx-mma-s8-v1",
        "wm4-wn4-tm4-tn2-bk32-st3-gm8", kPtx, 4, 4, 4, 2,
        32, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk32Wm4Wn4Tm2Tn4,
        "ptx k32 4x4/2x4", "cuda-ptx-mma-s8-v1",
        "wm4-wn4-tm2-tn4-bk32-st3-gm8", kPtx, 4, 4, 2, 4,
        32, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk32Wm8Wn2Tm2Tn2,
        "ptx k32 8x2/2x2", "cuda-ptx-mma-s8-v1",
        "wm8-wn2-tm2-tn2-bk32-st3-gm8", kPtx, 8, 2, 2, 2,
        32, 3, 8, minimumTileArchitecture(80)),
    makeCudaPreparationPlan(CudaPreparationPlanToken::PtxBk32Wm4Wn8Tm2Tn2,
        "ptx k32 4x8/2x2", "cuda-ptx-mma-s8-v1",
        "wm4-wn8-tm2-tn2-bk32-st3-gm8", kPtx, 4, 8, 2, 2,
        32, 3, 8, minimumTileArchitecture(80)),
#if defined(CANARY_PEARL_AMPERE_BUILD)
    // SM86: 128x256 block, 8 warps of 64x64, k128 stages XOR-swizzled instead of
    // padded so two of them fit in 96 KB. noisy_gemm_ampere.cuh; rank 128 only.
    makeCudaPreparationPlan(CudaPreparationPlanToken::AmpereBk128Wm2Wn4Tm4Tn4,
        "amp k128 2x4/4x4", "cuda-ampere-mma-s8-swizzle-v1",
        "wm2-wn4-tm4-tn4-bk128-st2-gm8-swz", kPtx, 2, 4, 4, 4,
        128, 2, 8, minimumTileArchitecture(80)),
#endif
};

inline constexpr size_t kCudaPreparationPlanCount =
    sizeof(kCudaPreparationPlans) / sizeof(kCudaPreparationPlans[0]);
// Isolated exact-profile extension; never part of normal tuning enumeration.
inline constexpr auto kCudaTmaDeliveryPlan = makeCudaPreparationPlan(
    CudaPreparationPlanToken::ExperimentalSm120TmaDeliveryK8192,
    "tma k64 4x4/2x4", "cuda-tma-mma-s8-delivery-v1",
    "wm4-wn4-tm2-tn4-bk64-st3-gm8-tma64", kPtx, 4, 4, 2, 4,
    64, 3, 8, exactTileArchitecture(120));

// Not part of the default catalog or its witness/timing enumeration.
inline constexpr auto kCudaFixedExperimentPlan = makeCudaPreparationPlan(
    CudaPreparationPlanToken::ExperimentalSm120FixedK2048R128,
    "experimental sm120 fixed k2048 r128 4x4/2x4",
    "cuda-ptx-mma-s8-sm120-fixed-k2048-r128-experimental-v1",
    "wm4-wn4-tm2-tn4-bk64-st3-gm8", kPtx, 4, 4, 2, 4,
    64, 3, 8, exactTileArchitecture(120));
inline constexpr const char *kCudaFixedExperimentOptIn = "sm120-fixed-k2048-r128-v1";
inline constexpr auto kCudaWm8ExperimentPlan = makeCudaPreparationPlan(
    CudaPreparationPlanToken::ExperimentalSm120Wm8Runtime,
    "experimental sm120 runtime 8x2/2x4",
    "cuda-ptx-mma-s8-sm120-wm8-runtime-experimental-v1",
    "wm8-wn2-tm2-tn4-bk64-st3-gm8", kPtx, 8, 2, 2, 4,
    64, 3, 8, exactTileArchitecture(120));
inline constexpr const char *kCudaWm8ExperimentOptIn = "sm120-wm8-wn2-tm2-tn4-runtime-v1";
inline constexpr auto kCudaWm8Cap112ExperimentPlan = makeCudaPreparationPlan(
    CudaPreparationPlanToken::ExperimentalSm120Wm8InlineCap112,
    "experimental sm120 runtime inline cap112 8x2/2x4",
    "cuda-ptx-mma-s8-sm120-wm8-runtime-inline-cap112-experimental-v1",
    "wm8-wn2-tm2-tn4-bk64-st3-gm8", kPtx, 8, 2, 2, 4,
    64, 3, 8, exactTileArchitecture(120));
inline constexpr const char *kCudaWm8Cap112ExperimentOptIn =
    "sm120-wm8-wn2-tm2-tn4-runtime-inline-cap112-v1";
inline constexpr auto kCudaWm8FixedExperimentPlan = makeCudaPreparationPlan(
    CudaPreparationPlanToken::ExperimentalSm120Wm8FixedK2048R128,
    "experimental sm120 fixed k2048 r128 8x2/2x4",
    "cuda-ptx-mma-s8-sm120-wm8-fixed-k2048-r128-experimental-v1",
    "wm8-wn2-tm2-tn4-bk64-st3-gm8", kPtx, 8, 2, 2, 4,
    64, 3, 8, exactTileArchitecture(120));
inline constexpr const char *kCudaWm8FixedExperimentOptIn =
    "sm120-wm8-wn2-tm2-tn4-fixed-k2048-r128-v1";
inline constexpr auto kCudaBk32FixedExperimentPlan = makeCudaPreparationPlan(
    CudaPreparationPlanToken::ExperimentalSm120Bk32FixedK2048R128,
    "experimental sm120 bk32 fixed k2048 r128 2x4/4x2",
    "cuda-ptx-mma-s8-sm120-bk32-fixed-k2048-r128-experimental-v1",
    "wm2-wn4-tm4-tn2-bk32-st3-gm8", kPtx, 2, 4, 4, 2,
    32, 3, 8, exactTileArchitecture(120));
inline constexpr const char *kCudaBk32FixedExperimentOptIn =
    "sm120-wm2-wn4-tm4-tn2-bk32-fixed-k2048-r128-v1";
inline constexpr auto kCudaWm4ReuseExperimentPlan = makeCudaPreparationPlan(
    CudaPreparationPlanToken::ExperimentalSm120Wm4ReuseFixedK2048R128,
    "experimental sm120 wm4 reuse fixed k2048 r128 4x2/4x4",
    "cuda-ptx-mma-s8-sm120-wm4-reuse-fixed-k2048-r128-experimental-v1",
    "wm4-wn2-tm4-tn4-bk64-st3-gm8", kPtx, 4, 2, 4, 4,
    64, 3, 8, exactTileArchitecture(120));
inline constexpr const char *kCudaWm4ReuseExperimentOptIn =
    "sm120-wm4-reuse-wn2-tm4-tn4-fixed-k2048-r128-v1";
// TMA changes only loading: retain the canonical16x16 proof layout and WM8
// arithmetic geometry. Shared-memory/barrier bytes belong to the launcher,
// not this descriptor (the TMA allocation must not be inferred as padded PTX).
inline constexpr auto kCudaTmaWm8ExperimentPlan = makeCudaPreparationPlan(
    CudaPreparationPlanToken::ExperimentalSm120TmaWm8FixedK2048R128,
    "experimental sm120 tma wm8 fixed k2048 r128 8x2/2x4",
    "cuda-tma-mma-s8-sm120-wm8-fixed-k2048-r128-experimental-v1",
    "wm8-wn2-tm2-tn4-bk64-st3-gm8-tma", kPtx, 8, 2, 2, 4,
    64, 3, 8, exactTileArchitecture(120));
inline constexpr const char *kCudaTmaWm8ExperimentOptIn =
    "sm120-tma-wm8-fixed-k2048-r128-v1";
inline constexpr auto kCudaTmaTraversalExperimentPlan = makeCudaPreparationPlan(
    CudaPreparationPlanToken::ExperimentalSm120TmaTraversal16x32,
    "experimental sm120 tma wm8 region16x32 k2048 r128",
    "cuda-tma-mma-s8-sm120-wm8-region16x32-experimental-v1",
    "wm8-wn2-tm2-tn4-bk64-st3-region16x32-tma", kPtx, 8, 2, 2, 4,
    64, 3, 16, exactTileArchitecture(120));
inline constexpr const char *kCudaTmaTraversalExperimentOptIn =
    "sm120-tma-wm8-region16x32-k2048-r128-v1";
// Separate default-off identity: packed TMA shared allocation is 98336 bytes
// in its launcher, not the padded PTX allocation implied by this geometry.
inline constexpr auto kCudaTmaStage4ExperimentPlan = makeCudaPreparationPlan(
    CudaPreparationPlanToken::ExperimentalSm120TmaTraversal16x32Stage4,
    "experimental sm120 tma wm8 region16x32 st4 k2048 r128",
    "cuda-tma-mma-s8-sm120-wm8-region16x32-st4-experimental-v1",
    "wm8-wn2-tm2-tn4-bk64-st4-region16x32-tma", kPtx, 8, 2, 2, 4,
    64, 4, 16, exactTileArchitecture(120));
inline constexpr const char *kCudaTmaStage4ExperimentOptIn =
    "sm120-tma-wm8-region16x32-st4-k2048-r128-v1";
// Preserve all existing enum values and their meanings.
enum class CudaFixedExperimentSelection : uint8_t { Disabled, Enabled, Invalid, Wm8Runtime, Wm8InlineCap112, Wm8FixedK2048R128, Bk32FixedK2048R128, Wm4ReuseFixedK2048R128, TmaWm8FixedK2048R128, TmaTraversal16x32, TmaTraversal16x32Stage4 };
constexpr bool cudaIsolatedExperimentEnabled(CudaFixedExperimentSelection selection) noexcept {
    return selection == CudaFixedExperimentSelection::Enabled ||
           selection == CudaFixedExperimentSelection::Wm8Runtime ||
           selection == CudaFixedExperimentSelection::Wm8InlineCap112 ||
           selection == CudaFixedExperimentSelection::Wm8FixedK2048R128 ||
           selection == CudaFixedExperimentSelection::Bk32FixedK2048R128 ||
           selection == CudaFixedExperimentSelection::Wm4ReuseFixedK2048R128 ||
           selection == CudaFixedExperimentSelection::TmaWm8FixedK2048R128 ||
           selection == CudaFixedExperimentSelection::TmaTraversal16x32 ||
           selection == CudaFixedExperimentSelection::TmaTraversal16x32Stage4;
}
constexpr bool cudaIsolatedExperimentCompiled(CudaFixedExperimentSelection selection,
                                               bool fixedCompiled, bool wm8Compiled,
                                               bool wm8Cap112Compiled = false,
                                               bool wm8FixedCompiled = false,
                                               bool bk32FixedCompiled = false,
                                               bool wm4ReuseCompiled = false,
                                               bool tmaWm8Compiled = false,
                                               bool tmaTraversalCompiled = false,
                                               bool tmaStage4Compiled = false) noexcept {
    return selection == CudaFixedExperimentSelection::Enabled ? fixedCompiled :
           selection == CudaFixedExperimentSelection::Wm8Runtime ? wm8Compiled :
           selection == CudaFixedExperimentSelection::Wm8InlineCap112 ? wm8Cap112Compiled :
           selection == CudaFixedExperimentSelection::Wm8FixedK2048R128 ? wm8FixedCompiled :
           selection == CudaFixedExperimentSelection::Bk32FixedK2048R128 ? bk32FixedCompiled :
           selection == CudaFixedExperimentSelection::Wm4ReuseFixedK2048R128 ? wm4ReuseCompiled :
           selection == CudaFixedExperimentSelection::TmaWm8FixedK2048R128 ? tmaWm8Compiled :
           selection == CudaFixedExperimentSelection::TmaTraversal16x32 ? tmaTraversalCompiled :
           selection == CudaFixedExperimentSelection::TmaTraversal16x32Stage4 ? tmaStage4Compiled : false;
}

inline CudaFixedExperimentSelection parseCudaFixedExperiment(const char *value) noexcept {
    if (!value) return CudaFixedExperimentSelection::Disabled;
    if (std::string_view(value) == kCudaFixedExperimentOptIn) return CudaFixedExperimentSelection::Enabled;
    if (std::string_view(value) == kCudaWm8ExperimentOptIn) return CudaFixedExperimentSelection::Wm8Runtime;
    if (std::string_view(value) == kCudaWm8Cap112ExperimentOptIn) return CudaFixedExperimentSelection::Wm8InlineCap112;
    if (std::string_view(value) == kCudaWm8FixedExperimentOptIn) return CudaFixedExperimentSelection::Wm8FixedK2048R128;
    if (std::string_view(value) == kCudaBk32FixedExperimentOptIn) return CudaFixedExperimentSelection::Bk32FixedK2048R128;
    if (std::string_view(value) == kCudaWm4ReuseExperimentOptIn) return CudaFixedExperimentSelection::Wm4ReuseFixedK2048R128;
    if (std::string_view(value) == kCudaTmaWm8ExperimentOptIn) return CudaFixedExperimentSelection::TmaWm8FixedK2048R128;
    if (std::string_view(value) == kCudaTmaTraversalExperimentOptIn) return CudaFixedExperimentSelection::TmaTraversal16x32;
    if (std::string_view(value) == kCudaTmaStage4ExperimentOptIn) return CudaFixedExperimentSelection::TmaTraversal16x32Stage4;
    return CudaFixedExperimentSelection::Invalid;
}

inline bool cudaFixedExperimentConfigurationValid(
    CudaFixedExperimentSelection selection, bool compiled, bool formal,
    bool synthetic, std::string_view poolHost, int poolPort,
    bool shapeOverride, bool tileOverride, bool overlapOverride = false,
    bool rootFusionOverride = false, bool profilingOverride = false) noexcept {
    if (selection == CudaFixedExperimentSelection::Disabled) return true;
    return cudaIsolatedExperimentEnabled(selection) && compiled &&
           !formal && !synthetic && poolHost == "127.0.0.1" &&
           poolPort > 0 && poolPort <= 65535 && !shapeOverride && !tileOverride &&
           ((selection != CudaFixedExperimentSelection::Wm8FixedK2048R128 &&
             selection != CudaFixedExperimentSelection::Bk32FixedK2048R128 &&
             selection != CudaFixedExperimentSelection::Wm4ReuseFixedK2048R128 &&
             selection != CudaFixedExperimentSelection::TmaWm8FixedK2048R128 &&
             selection != CudaFixedExperimentSelection::TmaTraversal16x32 &&
             selection != CudaFixedExperimentSelection::TmaTraversal16x32Stage4) ||
            (!overlapOverride && !rootFusionOverride && !profilingOverride));
}

inline bool cudaWm8FixedEnvironmentValid(CudaFixedExperimentSelection captured,
    const char *current, bool capturedConflict, bool currentConflict) noexcept {
    return (captured != CudaFixedExperimentSelection::Wm8FixedK2048R128 &&
            captured != CudaFixedExperimentSelection::Bk32FixedK2048R128 &&
            captured != CudaFixedExperimentSelection::Wm4ReuseFixedK2048R128 &&
            captured != CudaFixedExperimentSelection::TmaWm8FixedK2048R128 &&
            captured != CudaFixedExperimentSelection::TmaTraversal16x32 &&
            captured != CudaFixedExperimentSelection::TmaTraversal16x32Stage4) ||
        (!capturedConflict && !currentConflict && parseCudaFixedExperiment(current) == captured);
}

constexpr bool cudaFixedExperimentLaunchValid(int architecture, int m, int n,
                                             int k, int rank) noexcept {
    // The small shape is exclusively the unchanged independent selfCheck.
    return architecture == 120 && k == 2048 && rank == 128 &&
           ((m == 4096 && n == 65536) || (m == 256 && n == 256));
}

// Selection-specific dispatch; the original SM120 predicate remains unchanged.
constexpr bool cudaSelectedExperimentLaunchValid(CudaFixedExperimentSelection selection,
    int architecture, int m, int n, int k, int rank) noexcept {
    return cudaIsolatedExperimentEnabled(selection) &&
        cudaFixedExperimentLaunchValid(architecture, m, n, k, rank);
}

// Shared with host tests: default enumeration must not even visit the extension
// as a witness; enabled mode keeps all old witnesses but selects only it.
constexpr bool cudaFixedExperimentVisits(CudaFixedExperimentSelection selection,
                                         bool experimental, bool formal) noexcept {
    return (selection == CudaFixedExperimentSelection::Disabled || cudaIsolatedExperimentEnabled(selection)) &&
           (!experimental || (cudaIsolatedExperimentEnabled(selection) && !formal));
}
constexpr bool cudaFixedExperimentSelectable(CudaFixedExperimentSelection selection,
                                             bool experimental) noexcept {
    return cudaIsolatedExperimentEnabled(selection) ? experimental :
           selection == CudaFixedExperimentSelection::Disabled && !experimental;
}

constexpr bool cudaPreparationTextEqual(const char *a, const char *b) noexcept {
    return a && b && std::string_view(a) == std::string_view(b);
}

constexpr bool cudaPreparationPlanDescriptorEqual(
    const CudaPreparationPlanDescriptor &a,
    const CudaPreparationPlanDescriptor &b) noexcept {
    return a.token == b.token && a.preparationPlanId == b.preparationPlanId &&
           cudaPreparationTextEqual(a.name, b.name) &&
           cudaPreparationTextEqual(a.kernelId, b.kernelId) &&
           cudaPreparationTextEqual(a.tileId, b.tileId) &&
           a.family == b.family && a.wm == b.wm && a.wn == b.wn &&
           a.tm == b.tm && a.tn == b.tn && a.blockM == b.blockM &&
           a.blockN == b.blockN && a.threads == b.threads &&
           a.blockK == b.blockK && a.stages == b.stages &&
           a.groupM == b.groupM && a.nMajorB == b.nMajorB &&
           a.architecture.capability == b.architecture.capability &&
           a.architecture.match == b.architecture.match;
}

constexpr const CudaPreparationPlanDescriptor *findCudaPreparationPlan(
    CudaPreparationPlanToken token) noexcept {
    if (token == kCudaTmaDeliveryPlan.token) return &kCudaTmaDeliveryPlan;
    for (const auto &plan : kCudaPreparationPlans)
        if (plan.token == token) return &plan;
    if (token == kCudaFixedExperimentPlan.token) return &kCudaFixedExperimentPlan;
    if (token == kCudaWm8ExperimentPlan.token) return &kCudaWm8ExperimentPlan;
    if (token == kCudaWm8FixedExperimentPlan.token) return &kCudaWm8FixedExperimentPlan;
    if (token == kCudaBk32FixedExperimentPlan.token) return &kCudaBk32FixedExperimentPlan;
    if (token == kCudaWm4ReuseExperimentPlan.token) return &kCudaWm4ReuseExperimentPlan;
    if (token == kCudaTmaWm8ExperimentPlan.token) return &kCudaTmaWm8ExperimentPlan;
    if (token == kCudaTmaTraversalExperimentPlan.token) return &kCudaTmaTraversalExperimentPlan;
    if (token == kCudaTmaStage4ExperimentPlan.token) return &kCudaTmaStage4ExperimentPlan;
    return token == kCudaWm8Cap112ExperimentPlan.token ? &kCudaWm8Cap112ExperimentPlan : nullptr;
}

constexpr const CudaPreparationPlanDescriptor *findCudaPreparationPlan(
    std::string_view kernelId, std::string_view tileId) noexcept {
    if (kernelId == kCudaTmaDeliveryPlan.kernelId && tileId == kCudaTmaDeliveryPlan.tileId)
        return &kCudaTmaDeliveryPlan;
    for (const auto &plan : kCudaPreparationPlans)
        if (kernelId == plan.kernelId && tileId == plan.tileId) return &plan;
    if (kernelId == kCudaFixedExperimentPlan.kernelId && tileId == kCudaFixedExperimentPlan.tileId)
        return &kCudaFixedExperimentPlan;
    if (kernelId == kCudaWm8ExperimentPlan.kernelId && tileId == kCudaWm8ExperimentPlan.tileId)
        return &kCudaWm8ExperimentPlan;
    if (kernelId == kCudaWm8Cap112ExperimentPlan.kernelId && tileId == kCudaWm8Cap112ExperimentPlan.tileId)
        return &kCudaWm8Cap112ExperimentPlan;
    if (kernelId == kCudaWm8FixedExperimentPlan.kernelId && tileId == kCudaWm8FixedExperimentPlan.tileId)
        return &kCudaWm8FixedExperimentPlan;
    if (kernelId == kCudaBk32FixedExperimentPlan.kernelId && tileId == kCudaBk32FixedExperimentPlan.tileId)
        return &kCudaBk32FixedExperimentPlan;
    if (kernelId == kCudaWm4ReuseExperimentPlan.kernelId && tileId == kCudaWm4ReuseExperimentPlan.tileId)
        return &kCudaWm4ReuseExperimentPlan;
    if (kernelId == kCudaTmaWm8ExperimentPlan.kernelId && tileId == kCudaTmaWm8ExperimentPlan.tileId)
        return &kCudaTmaWm8ExperimentPlan;
    if (kernelId == kCudaTmaTraversalExperimentPlan.kernelId && tileId == kCudaTmaTraversalExperimentPlan.tileId)
        return &kCudaTmaTraversalExperimentPlan;
    if (kernelId == kCudaTmaStage4ExperimentPlan.kernelId && tileId == kCudaTmaStage4ExperimentPlan.tileId)
        return &kCudaTmaStage4ExperimentPlan;
    return nullptr;
}

template <CudaPreparationPlanToken Token>
constexpr const CudaPreparationPlanDescriptor &cudaPreparationPlan() noexcept {
    constexpr size_t index = []() constexpr {
        for (size_t i = 0; i < kCudaPreparationPlanCount; ++i)
            if (kCudaPreparationPlans[i].token == Token) return i;
        return kCudaPreparationPlanCount;
    }();
    static_assert(index < kCudaPreparationPlanCount,
                  "unknown CUDA preparation plan token");
    return kCudaPreparationPlans[index];
}

constexpr bool cudaPreparationPlanCatalogValid() noexcept {
#if defined(CANARY_PEARL_AMPERE_BUILD)
    if (kCudaPreparationPlanCount != 39) return false;
#else
    if (kCudaPreparationPlanCount != 38) return false;
#endif
    for (size_t i = 0; i < kCudaPreparationPlanCount; ++i) {
        const auto &plan = kCudaPreparationPlans[i];
        if (static_cast<uint16_t>(plan.token) == 0 ||
            plan.preparationPlanId != cudaPreparationPlanId(plan.token) ||
            !plan.name || !*plan.name || !plan.kernelId || !*plan.kernelId ||
            !plan.tileId || !*plan.tileId || plan.wm <= 0 || plan.wn <= 0 ||
            plan.tm <= 0 || plan.tn <= 0 ||
            plan.blockM != plan.wm * plan.tm * 16 ||
            plan.blockN != plan.wn * plan.tn * 16 ||
            plan.threads != plan.wm * plan.wn * 32 ||
            plan.architecture.capability <= 0)
            return false;
        if (plan.architecture.match != TileArchMatch::AtLeast &&
            plan.architecture.match != TileArchMatch::Exact)
            return false;
        switch (plan.family) {
            case kSingle:
                if (plan.nMajorB || plan.blockK != 32 ||
                    plan.stages != 1 || plan.groupM != 0) return false;
                break;
            case kDoubleBuffered:
                if (plan.nMajorB || plan.blockK != 32 ||
                    plan.stages != 2 || plan.groupM != 0) return false;
                break;
            case kAsync:
                if (plan.nMajorB || plan.blockK != 32 ||
                    plan.stages != 4 || plan.groupM != 0) return false;
                break;
            case kPtx:
#if defined(CANARY_PEARL_AMPERE_BUILD)
                if (plan.token == CudaPreparationPlanToken::AmpereBk128Wm2Wn4Tm4Tn4) {
                    if (!plan.nMajorB || plan.blockK != 128 || plan.stages != 2 ||
                        plan.groupM != 8) return false;
                    break;
                }
#endif
                if (!plan.nMajorB ||
                    (plan.blockK != 32 && plan.blockK != 64) ||
                    plan.stages != 3 || plan.groupM != 8) return false;
                break;
            default: return false;
        }
        for (size_t j = i + 1; j < kCudaPreparationPlanCount; ++j) {
            const auto &other = kCudaPreparationPlans[j];
            if (plan.token == other.token ||
                plan.preparationPlanId == other.preparationPlanId ||
                (cudaPreparationTextEqual(plan.kernelId, other.kernelId) &&
                 cudaPreparationTextEqual(plan.tileId, other.tileId)))
                return false;
        }
    }
    return true;
}

static_assert(cudaPreparationPlanCatalogValid(),
              "CUDA preparation catalog must be unique and self-consistent");
static_assert(kCudaTmaStage4ExperimentPlan.preparationPlanId == UINT64_C(0x4355444100010031),
              "TMA stage4 permanent distinct identity");
static_assert([]() constexpr {
    for (const auto *plan : {&kCudaFixedExperimentPlan, &kCudaWm8ExperimentPlan,
         &kCudaWm8Cap112ExperimentPlan, &kCudaWm8FixedExperimentPlan,
         &kCudaBk32FixedExperimentPlan,
         &kCudaWm4ReuseExperimentPlan, &kCudaTmaWm8ExperimentPlan,
         &kCudaTmaTraversalExperimentPlan})
        if (plan->token == kCudaTmaStage4ExperimentPlan.token ||
            plan->preparationPlanId == kCudaTmaStage4ExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan->kernelId,kCudaTmaStage4ExperimentPlan.kernelId)) return false;
    for (const auto &plan : kCudaPreparationPlans)
        if (plan.token == kCudaTmaStage4ExperimentPlan.token ||
            plan.preparationPlanId == kCudaTmaStage4ExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan.kernelId,kCudaTmaStage4ExperimentPlan.kernelId)) return false;
    return true;
}(), "TMA stage4 cannot alias any default or prior experiment");
static_assert(kCudaTmaTraversalExperimentPlan.preparationPlanId == UINT64_C(0x435544410001002f),
              "TMA traversal permanent distinct identity");
static_assert([]() constexpr {
    for (const auto *plan : {&kCudaFixedExperimentPlan, &kCudaWm8ExperimentPlan,
         &kCudaWm8Cap112ExperimentPlan, &kCudaWm8FixedExperimentPlan,
         &kCudaBk32FixedExperimentPlan,
         &kCudaWm4ReuseExperimentPlan, &kCudaTmaWm8ExperimentPlan})
        if (plan->token == kCudaTmaTraversalExperimentPlan.token ||
            plan->preparationPlanId == kCudaTmaTraversalExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan->kernelId,kCudaTmaTraversalExperimentPlan.kernelId)) return false;
    for (const auto &plan : kCudaPreparationPlans)
        if (plan.token == kCudaTmaTraversalExperimentPlan.token ||
            plan.preparationPlanId == kCudaTmaTraversalExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan.kernelId,kCudaTmaTraversalExperimentPlan.kernelId)) return false;
    return true;
}(), "TMA traversal cannot alias any default or prior experiment");
static_assert(kCudaTmaWm8ExperimentPlan.preparationPlanId == UINT64_C(0x435544410001002e),
              "TMA WM8 experiment permanent plan identity");
static_assert([]() constexpr {
    for (const auto *plan : {&kCudaFixedExperimentPlan, &kCudaWm8ExperimentPlan,
                            &kCudaWm8Cap112ExperimentPlan, &kCudaWm8FixedExperimentPlan,
                            &kCudaBk32FixedExperimentPlan,
                            &kCudaWm4ReuseExperimentPlan})
        if (plan->token == kCudaTmaWm8ExperimentPlan.token ||
            plan->preparationPlanId == kCudaTmaWm8ExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan->kernelId,kCudaTmaWm8ExperimentPlan.kernelId)) return false;
    for (const auto &plan : kCudaPreparationPlans)
        if (plan.token == kCudaTmaWm8ExperimentPlan.token ||
            plan.preparationPlanId == kCudaTmaWm8ExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan.kernelId,kCudaTmaWm8ExperimentPlan.kernelId)) return false;
    return true;
}(), "TMA WM8 cannot alias any default or previous experiment");
static_assert(kCudaWm4ReuseExperimentPlan.preparationPlanId == UINT64_C(0x435544410001002d),
              "WM4 reuse experiment permanent plan identity");
static_assert([] {
    for (const auto *plan : {&kCudaFixedExperimentPlan, &kCudaWm8ExperimentPlan,
                            &kCudaWm8Cap112ExperimentPlan, &kCudaWm8FixedExperimentPlan,
                            &kCudaBk32FixedExperimentPlan})
        if (plan->token == kCudaWm4ReuseExperimentPlan.token ||
            plan->preparationPlanId == kCudaWm4ReuseExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan->kernelId, kCudaWm4ReuseExperimentPlan.kernelId)) return false;
    for (const auto &plan : kCudaPreparationPlans)
        if (plan.token == kCudaWm4ReuseExperimentPlan.token ||
            plan.preparationPlanId == kCudaWm4ReuseExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan.kernelId, kCudaWm4ReuseExperimentPlan.kernelId)) return false;
    return true;
}(), "WM4 reuse identity is separate from every prior and default plan");
static_assert(kCudaBk32FixedExperimentPlan.preparationPlanId == UINT64_C(0x435544410001002c),
              "BK32 fixed experiment permanent plan identity");
static_assert([] {
    for (const auto *plan : {&kCudaFixedExperimentPlan, &kCudaWm8ExperimentPlan,
                            &kCudaWm8Cap112ExperimentPlan, &kCudaWm8FixedExperimentPlan,
                            })
        if (plan->token == kCudaBk32FixedExperimentPlan.token ||
            plan->preparationPlanId == kCudaBk32FixedExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan->kernelId, kCudaBk32FixedExperimentPlan.kernelId)) return false;
    for (const auto &plan : kCudaPreparationPlans)
        if (plan.token == kCudaBk32FixedExperimentPlan.token ||
            plan.preparationPlanId == kCudaBk32FixedExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan.kernelId, kCudaBk32FixedExperimentPlan.kernelId)) return false;
    return true;
}(), "BK32 fixed identity is separate from every prior and default plan");

static_assert(kCudaFixedExperimentPlan.preparationPlanId == UINT64_C(0x4355444100010027),
              "experimental plan identity is permanent");
static_assert(kCudaWm8ExperimentPlan.preparationPlanId == UINT64_C(0x4355444100010028),
              "WM8 experimental identity is permanent");
static_assert(kCudaWm8Cap112ExperimentPlan.preparationPlanId == UINT64_C(0x4355444100010029),
              "WM8 inline cap112 experimental identity is permanent");
static_assert(kCudaWm8FixedExperimentPlan.preparationPlanId == UINT64_C(0x435544410001002a),
              "fixed WM8 experimental identity is permanent");
static_assert([]() constexpr {
    for (const auto *plan : {&kCudaFixedExperimentPlan, &kCudaWm8ExperimentPlan,
                             &kCudaWm8Cap112ExperimentPlan})
        if (plan->token == kCudaWm8FixedExperimentPlan.token ||
            plan->preparationPlanId == kCudaWm8FixedExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan->kernelId, kCudaWm8FixedExperimentPlan.kernelId)) return false;
    return true;
}(), "fixed WM8 cannot alias an earlier experiment");
static_assert(kCudaWm8Cap112ExperimentPlan.token != kCudaWm8ExperimentPlan.token &&
              kCudaWm8Cap112ExperimentPlan.token != kCudaFixedExperimentPlan.token &&
              kCudaWm8Cap112ExperimentPlan.preparationPlanId != kCudaWm8ExperimentPlan.preparationPlanId &&
              kCudaWm8Cap112ExperimentPlan.preparationPlanId != kCudaFixedExperimentPlan.preparationPlanId &&
              !cudaPreparationTextEqual(kCudaWm8Cap112ExperimentPlan.kernelId, kCudaWm8ExperimentPlan.kernelId) &&
              !cudaPreparationTextEqual(kCudaWm8Cap112ExperimentPlan.kernelId, kCudaFixedExperimentPlan.kernelId),
              "inline cap112 identity cannot alias existing experiments");
static_assert(kCudaFixedExperimentPlan.token != kCudaWm8ExperimentPlan.token &&
              kCudaFixedExperimentPlan.preparationPlanId != kCudaWm8ExperimentPlan.preparationPlanId &&
              !cudaPreparationTextEqual(kCudaFixedExperimentPlan.kernelId, kCudaWm8ExperimentPlan.kernelId),
              "isolated experiments cannot alias");
static_assert([]() constexpr {
    for (const auto &plan : kCudaPreparationPlans)
        if (plan.token == kCudaFixedExperimentPlan.token ||
            plan.preparationPlanId == kCudaFixedExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan.kernelId, kCudaFixedExperimentPlan.kernelId) ||
            plan.token == kCudaWm8ExperimentPlan.token ||
            plan.preparationPlanId == kCudaWm8ExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan.kernelId, kCudaWm8ExperimentPlan.kernelId) ||
            plan.token == kCudaWm8Cap112ExperimentPlan.token ||
            plan.preparationPlanId == kCudaWm8Cap112ExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan.kernelId, kCudaWm8Cap112ExperimentPlan.kernelId) ||
            plan.token == kCudaWm8FixedExperimentPlan.token ||
            plan.preparationPlanId == kCudaWm8FixedExperimentPlan.preparationPlanId ||
            cudaPreparationTextEqual(plan.kernelId, kCudaWm8FixedExperimentPlan.kernelId)) return false;
    return true;
}(), "experimental identity must not alias any default plan");

enum class CudaPreparationPlanError : uint8_t {
    None = 0,
    NullOutput,
    UnknownPlan,
    MetadataMismatch,
    InvalidArchitecture,
    InvalidShape,
    InvalidMode,
};

constexpr bool cudaPreparationPowerOfTwo(uint32_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

// Admit only the exact live shape catalog, then check CUDA/kernel/count
// constraints. Available VRAM remains a separate runtime allocation check.
inline bool cudaPreparationShapeValid(
    const CudaPreparationPlanDescriptor &plan, uint32_t m,
    uint32_t n) noexcept {
    const MiningConfig config{kCudaCommonDim, 128, 16, 16};
    if (!findCudaPreparationShape(m, n) ||
        !preparedMiningShapeValid(config, m, n) ||
        !cudaPreparationPowerOfTwo(m) || !cudaPreparationPowerOfTwo(n) ||
        m % 128 != 0 || n % 128 != 0 ||
        plan.blockM <= 0 || plan.blockN <= 0 || plan.blockK <= 0 ||
        m % static_cast<uint32_t>(plan.blockM) != 0 ||
        n % static_cast<uint32_t>(plan.blockN) != 0 ||
        kCudaCommonDim % plan.blockK != 0)
        return false;
    // CUDA's two-dimensional launch uses m/blockM as grid.y. Keep its
    // portable 16-bit bound and all current uint32 kernel counts explicit.
    if (m / static_cast<uint32_t>(plan.blockM) > 65535u) return false;
    const uint64_t tiles = (uint64_t{m} / 16) * (uint64_t{n} / 16);
    if (tiles == 0 || tiles > std::numeric_limits<uint32_t>::max() - 255u)
        return false;
    const uint64_t aBytes = uint64_t{m} * kCudaCommonDim;
    const uint64_t bBytes = uint64_t{n} * kCudaCommonDim;
    if (aBytes > std::numeric_limits<uint32_t>::max() - 255u ||
        bBytes > std::numeric_limits<uint32_t>::max() - 255u ||
        uint64_t{m} * 128 > std::numeric_limits<uint32_t>::max() - 255u ||
        uint64_t{n} * 128 > std::numeric_limits<uint32_t>::max() - 255u)
        return false;
    const uint64_t chunks = (aBytes > bBytes ? aBytes : bBytes) / 1024;
    const uint64_t required =
        aBytes * 2 + bBytes * 2 + uint64_t{m} * 128 +
        uint64_t{n} * 128 * 2 + uint64_t{kCudaCommonDim} * 8 + chunks * 32 * 2 +
        tiles * 16 * sizeof(uint32_t) + 64 * 36 + 4096;
    return required <= std::numeric_limits<size_t>::max();
}

// No CUDA calls, allocations, environment reads, or target resolution here.
// A caller must subsequently buildPreparedJobHost(job, explicitPolicy, request)
// to validate the canonical source context and resolve its target. Failure
// preserves the output request; the optional fixed-size error is reported.
inline bool buildCudaPreparedJobRequest(
    const CudaPreparationPlanDescriptor &selected, uint32_t m, uint32_t n,
    int architecture, PreparedJobMode mode, PreparedJobRequest *out,
    CudaPreparationPlanError *error = nullptr) noexcept {
    const auto fail = [error](CudaPreparationPlanError value) noexcept {
        if (error) *error = value;
        return false;
    };
    if (!out) return fail(CudaPreparationPlanError::NullOutput);
    const auto *canonical = findCudaPreparationPlan(selected.token);
    if (!canonical) return fail(CudaPreparationPlanError::UnknownPlan);
    if (!cudaPreparationPlanDescriptorEqual(selected, *canonical))
        return fail(CudaPreparationPlanError::MetadataMismatch);
    if (!tileArchitectureEligible(canonical->architecture, architecture))
        return fail(CudaPreparationPlanError::InvalidArchitecture);
    if (!cudaPreparationShapeValid(*canonical, m, n))
        return fail(CudaPreparationPlanError::InvalidShape);
    if (canonical->token == kCudaTmaDeliveryPlan.token) {
        if (kCudaCommonDim!=8192 || m!=16384 || n!=131072)
            return fail(CudaPreparationPlanError::InvalidShape);
        // The unchanged startup tuner uses a validated SyntheticBenchmark
        // job and SyntheticHardTarget preparation. Keep the shared closed
        // Normal/SyntheticHardTarget mode check below; live evidence remains
        // exact-profile, loopback-owned and catalog bound.
    }
    if (canonical->token == kCudaFixedExperimentPlan.token || canonical->token == kCudaWm8ExperimentPlan.token ||
        canonical->token == kCudaWm8Cap112ExperimentPlan.token ||
        canonical->token == kCudaWm8FixedExperimentPlan.token ||
        canonical->token == kCudaBk32FixedExperimentPlan.token ||
        canonical->token == kCudaWm4ReuseExperimentPlan.token ||
        canonical->token == kCudaTmaWm8ExperimentPlan.token ||
        canonical->token == kCudaTmaTraversalExperimentPlan.token ||
        canonical->token == kCudaTmaStage4ExperimentPlan.token) {
        if (kCudaCommonDim != 2048) return fail(CudaPreparationPlanError::InvalidShape);
        if (m != 4096 || n != 65536) return fail(CudaPreparationPlanError::InvalidShape);
        if (mode != PreparedJobMode::Normal) return fail(CudaPreparationPlanError::InvalidMode);
    }
    if (mode != PreparedJobMode::Normal &&
        mode != PreparedJobMode::SyntheticHardTarget)
        return fail(CudaPreparationPlanError::InvalidMode);
    PreparedJobRequest next;
    next.config = MiningConfig{kCudaCommonDim, 128, 16, 16};
    next.m = m;
    next.n = n;
    next.preparationPlanId = canonical->preparationPlanId;
    next.mode = mode;
    *out = next;
    if (error) *error = CudaPreparationPlanError::None;
    return true;
}

}  // namespace om::pearl
