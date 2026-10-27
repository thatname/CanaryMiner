// Pearl (pearl-pow) - the Algorithm implementation.
//
// Pearl does not fit the shape of every other PoW in this repo, and it is
// worth saying how before reading the code.
//
// There is no nonce. The miner picks the matrices A (m x k) and B (k x n), and
// their Merkle roots ARE the noise seeds and the PoW key, so choosing them is
// the whole of the search. `nonceBase` here indexes a choice of A rather than
// a value that gets hashed.
//
// One attempt is a whole GEMM and yields (m/16)*(n/16) transcripts, each
// a candidate digest; statistical independence is not established by this
// count. So an attempt is milliseconds, not nanoseconds, and
// `count` is read as "this many candidates" rather than "this many nonces" -
// which keeps the core's hashrate figure honest, since it is candidates per
// second either way.
//
// The one structural trick, measured rather than assumed (see the header of
// prepare.cuh):
//
//   commitment_B = blake3(job_key || salted B^t root) does not depend on A.
//   E_BL and E_BR come from it, so B_noised is bit-identical for every attempt
//   in a job. Generating, hashing and noising B are per-JOB work. Doing it
//   per attempt costs 48% overhead; doing it once costs 3%.
//
// A win is not a nonce either. It is a Merkle opening of the winning tile's
// sixteen rows of A and sixteen columns of B, tens of kilobytes, which travels
// in Solution::extra.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../../core/algo.h"
#include "../../core/pearl_job_extra.h"
#include "../../core/pearl_progress.h"
#include "../../core/pearl_speculative_attempt.h"
#include "../../core/pearl_integrated_pipeline.h"
#include "../../core/pearl_proof_binding.h"
#include "../../core/pearl_proof_timing.h"
#include "../../core/run.h"
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
#include "../../core/pearl_guard_retirement_v2.h"
#endif
#include "../../core/pearl_generic_native_evidence.h"
#include "../../core/preparation_cancellation.h"
#include "cuda_preparation_plan.h"
#include "attempt_profile.h"
#include "exact_stage_timing.h"
#include "attempt_overlap.h"
#include "integrated_pipeline.h"
#include "fused_scan.h"
#include "a_root_fusion.h"
#include "job.h"
#include "merkle_proof_cache.h"
#include "compact_merkle_proof.h"
#include "noisy_gemm.cuh"
#include "tma_fused_scan_policy.h"
#include "noisy_gemm_tma_delivery.cuh"
#include "noisy_gemm_combined.cuh"
#define CANARY_N64_TMA_SCAN_OUTPUT 1
#ifndef N64_GM
#define N64_GM 32
#endif
#include "noisy_gemm_n64.cuh"
#include "prepare_a_packed.cuh"
#if defined(CANARY_PEARL_ENABLE_TMA_WM8_EXPERIMENT)
#if CANARY_PEARL_ENABLE_TMA_WM8_EXPERIMENT != 1
#error "CANARY_PEARL_ENABLE_TMA_WM8_EXPERIMENT must be absent or exactly 1"
#endif
#if defined(CANARY_PEARL_TEST_PTX_INLINE_BODY) || defined(CANARY_PEARL_TEST_WM8_REGCAP) || defined(CANARY_PEARL_DIRECT_FOLD_MODE) || defined(CANARY_PEARL_TEST_RESOURCE_ONLY) || defined(CANARY_PEARL_ENABLE_WM8_CAP112_EXPERIMENT)
#error "TMA WM8 backend rejects altered baseline/resource-probe admission"
#endif
#include "noisy_gemm_tma.cuh"
#endif
#if defined(CANARY_PEARL_ENABLE_TMA_TRAVERSAL_EXPERIMENT)
#if CANARY_PEARL_ENABLE_TMA_TRAVERSAL_EXPERIMENT != 1 || !defined(CANARY_PEARL_ENABLE_TMA_WM8_EXPERIMENT)
#error "TMA traversal requires explicit value1 and the unchanged TMA baseline build"
#endif
#endif
#ifndef CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT
#define CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT 0
#endif
#if CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT != 0 && CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT != 1
#error "CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT must be 0 or 1"
#endif
#if CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT == 1
#if !defined(CANARY_PEARL_ENABLE_TMA_WM8_EXPERIMENT) || !defined(CANARY_PEARL_ENABLE_TMA_TRAVERSAL_EXPERIMENT)
#error "TMA stage4 requires explicitly compiled original TMA and traversal controls"
#endif
static_assert(CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT == 1, "exact TMA stage4 admission");
#include "../../../tests/pearl_tma_stage4_probe.cuh"
#endif
#include "prepare.cuh"
#include "mixed_schedule.h"
#include "search_state.h"
#include "tile_arch.h"
#include "tuner_stats.h"
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
#include "../../../tests/pearl_transcript_snapshot.h"
#include "../../../tests/pearl_fused_scan_snapshot.h"
#endif

namespace om {
namespace {

using namespace om::pearl;

#if defined(CANARY_PEARL_ENABLE_A_ROOT_FUSION)
#if CANARY_PEARL_ENABLE_A_ROOT_FUSION != 1 || !defined(CANARY_PEARL_TEST_ROOT_COMMIT_FUSION) || CANARY_PEARL_TEST_ROOT_COMMIT_FUSION != 1
#error "A root fusion backend requires exact enable and test-kernel flags"
#endif
constexpr bool kARootFusionCompiled = true;
#else
constexpr bool kARootFusionCompiled = false;
#endif

#if defined(CANARY_PEARL_ENABLE_FIXED_SCHEDULE_EXPERIMENT) && CANARY_PEARL_ENABLE_FIXED_SCHEDULE_EXPERIMENT == 1
constexpr bool kFixedExperimentCompiled = true;
#else
constexpr bool kFixedExperimentCompiled = false;
#endif
#if defined(CANARY_PEARL_ENABLE_WM8_EXPERIMENT) && CANARY_PEARL_ENABLE_WM8_EXPERIMENT != 1
#error "CANARY_PEARL_ENABLE_WM8_EXPERIMENT must be absent or exactly 1"
#endif
#if defined(CANARY_PEARL_ENABLE_WM8_EXPERIMENT) && CANARY_PEARL_ENABLE_WM8_EXPERIMENT == 1
constexpr bool kWm8ExperimentCompiled = true;
#else
constexpr bool kWm8ExperimentCompiled = false;
#endif
#if defined(CANARY_PEARL_ENABLE_WM8_CAP112_EXPERIMENT)
constexpr bool kWm8Cap112ExperimentCompiled = true;
#else
constexpr bool kWm8Cap112ExperimentCompiled = false;
#endif
#if defined(CANARY_PEARL_ENABLE_WM8_FIXED_EXPERIMENT)
#if CANARY_PEARL_ENABLE_WM8_FIXED_EXPERIMENT != 1
#error "CANARY_PEARL_ENABLE_WM8_FIXED_EXPERIMENT must be absent or exactly 1"
#endif
#if defined(CANARY_PEARL_TEST_PTX_INLINE_BODY) || defined(CANARY_PEARL_TEST_WM8_REGCAP) || defined(CANARY_PEARL_ENABLE_WM8_CAP112_EXPERIMENT)
#error "Fixed WM8 backend requires the original global body and launch qualifier"
#endif
constexpr bool kWm8FixedExperimentCompiled = true;
#else
constexpr bool kWm8FixedExperimentCompiled = false;
#endif
#if defined(CANARY_PEARL_ENABLE_BK32_FIXED_EXPERIMENT)
#if CANARY_PEARL_ENABLE_BK32_FIXED_EXPERIMENT != 1
#error "CANARY_PEARL_ENABLE_BK32_FIXED_EXPERIMENT must be absent or exactly 1"
#endif
#if defined(CANARY_PEARL_TEST_PTX_INLINE_BODY) || defined(CANARY_PEARL_TEST_WM8_REGCAP) || defined(CANARY_PEARL_ENABLE_WM8_CAP112_EXPERIMENT)
#error "BK32 fixed backend requires the original global body and launch qualifier"
#endif
constexpr bool kBk32FixedExperimentCompiled = true;
#else
constexpr bool kBk32FixedExperimentCompiled = false;
#endif
#if defined(CANARY_PEARL_ENABLE_WM4_REUSE_EXPERIMENT)
// Exact flag/body/resource-probe conflicts are checked in ptx_fixed_contract.h.
constexpr bool kWm4ReuseExperimentCompiled = true;
#else
constexpr bool kWm4ReuseExperimentCompiled = false;
#endif
constexpr bool experimentCompiled(CudaFixedExperimentSelection selection) {
    return cudaIsolatedExperimentCompiled(selection, kFixedExperimentCompiled, kWm8ExperimentCompiled,
                                          kWm8Cap112ExperimentCompiled, kWm8FixedExperimentCompiled,
                                          kBk32FixedExperimentCompiled,
                                          kWm4ReuseExperimentCompiled,
#if defined(CANARY_PEARL_ENABLE_TMA_WM8_EXPERIMENT)
                                          true,
#else
                                          false,
#endif
#if defined(CANARY_PEARL_ENABLE_TMA_TRAVERSAL_EXPERIMENT)
                                          true,
#else
                                          false,
#endif
                                          CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT == 1);
}
#if defined(CANARY_PEARL_ENABLE_ATTEMPT_OVERLAP) && CANARY_PEARL_ENABLE_ATTEMPT_OVERLAP != 1
#error "CANARY_PEARL_ENABLE_ATTEMPT_OVERLAP must be absent or exactly 1"
#endif
#if defined(CANARY_PEARL_ENABLE_ATTEMPT_OVERLAP) && CANARY_PEARL_ENABLE_ATTEMPT_OVERLAP == 1
constexpr bool kAttemptOverlapCompiled = true;
#else
constexpr bool kAttemptOverlapCompiled = false;
#endif

// Diagnostic only: never authorizes work/proofs or changes a CUDA failure's
// normal mining handling. No additional stream/event waits are introduced.
struct CudaProfileAdapter {
    bool exactStageTiming = false;
    using Event = cudaEvent_t;
    using Stream = cudaStream_t;
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    bool create(Event *event) noexcept { return cudaEventCreate(event) == cudaSuccess; }
    bool record(Event event, Stream stream) noexcept {
        return cudaEventRecord(event, stream) == cudaSuccess;
    }
    bool elapsed(float *ms, Event first, Event last) noexcept {
        return cudaEventElapsedTime(ms, first, last) == cudaSuccess;
    }
    void destroy(Event event) noexcept { cudaEventDestroy(event); }
    TimePoint now() noexcept { return Clock::now(); }
    double milliseconds(TimePoint first, TimePoint last) noexcept {
        return std::chrono::duration<double, std::milli>(last - first).count();
    }
    void publish(const profile::Sample &sample, uint32_t interval,
                 uint64_t nonce, uint32_t m, uint32_t n, size_t tile,
                 int arch, std::string_view device) noexcept {
        // tuningKey_ contains binary UUID bytes, not a printable string.
        char uuid[33] = "unknown";
        if (device.size() == 16) {
            constexpr char hex[] = "0123456789abcdef";
            for (size_t i = 0; i < 16; ++i) {
                const auto byte = static_cast<unsigned char>(device[i]);
                uuid[2 * i] = hex[byte >> 4];
                uuid[2 * i + 1] = hex[byte & 15];
            }
            uuid[32] = '\0';
        }
        if (exactStageTiming) {
            fprintf(stderr,
                "[pearl-stage-timing] v=1 profile=k8192-v1 scope=attempt interval=%u "
                "nonce=%016llx m=%u n=%u k=8192 rank=128 tile=%zu sm=%d device=%s "
                "a_gen_gpu_ms=%.6f merkle_gpu_ms=%.6f commit_gpu_ms=%.6f noise_gen_gpu_ms=%.6f "
                "apply_noise_gpu_ms=%.6f gemm_fold_gpu_ms=%.6f scan_gpu_ms=%.6f gpu_total_ms=%.6f "
                "host_enqueue_ms=%.6f host_readback_wait_ms=%.6f gpu_clock=cuda_events host_clock=steady ledger=none\n",
                interval, static_cast<unsigned long long>(nonce), m, n, tile, arch, uuid,
                sample.gpuMs[0], sample.gpuMs[1], sample.gpuMs[2], sample.gpuMs[3],
                sample.gpuMs[4], sample.gpuMs[5], sample.gpuMs[6], sample.gpuTotalMs(),
                sample.enqueueMs, sample.readbackWaitMs);
            return;
        }
        fprintf(stderr,
            "[pearl-profile] v=1 interval=%u nonce=%llu shape=%ux%u tile=%zu sm=%d device=%s "
            "a_gen_ms=%.6f merkle_ms=%.6f commit_ms=%.6f noise_gen_ms=%.6f "
            "apply_noise_ms=%.6f gemm_fold_ms=%.6f scan_ms=%.6f gpu_total_ms=%.6f "
            "host_enqueue_ms=%.6f host_readback_wait_ms=%.6f ledger=none\n",
            interval, static_cast<unsigned long long>(nonce), m, n, tile, arch,
            uuid, sample.gpuMs[0], sample.gpuMs[1], sample.gpuMs[2],
            sample.gpuMs[3], sample.gpuMs[4], sample.gpuMs[5],
            sample.gpuMs[6], sample.gpuTotalMs(), sample.enqueueMs,
            sample.readbackWaitMs);
    }
    void disabled(const char *reason) noexcept {
        fprintf(stderr, "%s disabled: %s; no timing sample published\n",
                exactStageTiming ? "[pearl-stage-timing]" : "[pearl-profile]", reason);
    }
};
using AttemptProfiler = profile::AttemptProfiler<CudaProfileAdapter>;

struct JobStageAdapter : proof_timing::SystemAdapter {
    void publish(const stage_timing::JobSample &sample) const noexcept {
        fprintf(stderr,
            "[pearl-stage-timing] v=1 profile=k8192-v1 scope=job-setup "
            "m=%u n=%u k=%u rank=%u sm=%d status=%s host_elapsed_ns=%llu "
            "start_ns=%llu end_ns=%llu clock=host_steady ledger=none\n",
            sample.m, sample.n, sample.k, sample.rank, sample.sm,
            !sample.valid ? "invalid" : sample.complete ? "complete" : "partial",
            static_cast<unsigned long long>(sample.elapsedNs),
            static_cast<unsigned long long>(sample.startNs),
            static_cast<unsigned long long>(sample.endNs));
    }
};

/**
 * Shapes to mine at, largest first, chosen by measurement.
 *
 * Ranked on candidates per second, not on prepare overhead - those disagree,
 * and overhead is the misleading one. On a 4090, 4096x65536 has the lowest
 * overhead of anything measured and is the slowest, because B^t at 128 MB
 * stops being servable from L2 and the GEMM falls from 67 to 45 TOPS.
 *
 * m and n must be powers of two so the padded matrices have a power-of-two
 * chunk count and the device Merkle reduction is exactly blake3's tree, and
 * multiples of the rank so the transcript index inverts cleanly.
 */
using Shape = CudaPreparationShapeDescriptor;

// Ordered by measured candidates/s, largest-first only incidentally. The
// ranking MOVED when the kernel got faster: 4096x32768 was the winner
// while the GEMM was compute-bound and is well behind now that it is not,
// because its working set no longer fits L2. Bigger is not better here.
// Per-attempt prepare is dominated by A (Merkle root + noising, both ~m*k)
// while candidates go as m*n, so overhead per candidate falls as 1/n and
// the widest shape the card can hold tends to win. On a 5080 the sweep
// puts 4096x65536 top at 2.1% overhead against 4.1% for 4096x32768 -
// and that shape was not in this list at all, so the tuner could not
// reach it however hard it measured.
// The ranking MOVES whenever the kernel changes - staging 64 of k instead
// of 32 pushed a 4090 from 4096x32768 down to 2048x16384 - so this list is
// deliberately denser than the shapes that have ever won. A shape absent
// here cannot be measured however good the tuner is.
// bytesFor() is ~360.5 MiB at this shape. VRAM capacity is not the reason
// to exclude it on 8/12 GB cards; measure the card's L2/DRAM behavior.
constexpr const auto &kShapes = kCudaPreparationShapes;

constexpr int kTileSide = 16;

/**
 * The tile configurations worth trying, and the launcher for each.
 *
 * This is a runtime choice, not a constant, because **the right answer is not
 * the same on two cards we own**. Measured at 4096x32768, k=2048:
 *
 *                      RTX 4090      RTX 5080
 *     2x4 / 2x2         97.3          58.0
 *     2x4 / 4x2        127.1 best     41.7 WORST
 *     2x4 / 2x4        110.1          68.5 best
 *     2x4 / 4x4        118.4          66.6
 *
 * Shipping the 4090's winner hardcoded would run a 5080 at 41.7 TOPS when that
 * card does 68.5 - giving away 64% for no reason a user could ever diagnose.
 * So prepare() times them on the actual device and picks.
 *
 * Every entry here is held to the reference in tests/test_pearl.cu. A
 * configuration that can be selected but was never verified is a correctness
 * bug waiting for someone else's GPU.
 */
// Scalar/string metadata has one host-testable source of truth. The factory
// below also derives the launch specialization from that same descriptor.
struct TileConfig : CudaPreparationPlanDescriptor {
    using Launch = cudaError_t (*)(dim3, int, cudaStream_t, const int8_t *,
                            const int8_t *, uint32_t *, int, int, int, int);
    Launch launch;

    constexpr TileConfig(const CudaPreparationPlanDescriptor &descriptor,
                         Launch function) noexcept
        : CudaPreparationPlanDescriptor(descriptor), launch(function) {}
};

template <int WM, int WN, int TM, int TN>
cudaError_t launchTiled(dim3 grid, int threads, cudaStream_t s, const int8_t *a,
                 const int8_t *b, uint32_t *t, int m, int n, int k, int rank) {
    noisyGemmMmaTiled<WM, WN, TM, TN>
        <<<grid, threads, 0, s>>>(a, b, nullptr, t, m, n, k, rank, false);
    return cudaGetLastError();
}

template <int WM, int WN, int TM, int TN>
cudaError_t launchTiledDB(dim3 grid, int threads, cudaStream_t s, const int8_t *a,
                   const int8_t *b, uint32_t *t, int m, int n, int k, int rank) {
    noisyGemmMmaTiledDB<WM, WN, TM, TN>
        <<<grid, threads, 0, s>>>(a, b, nullptr, t, m, n, k, rank, false);
    return cudaGetLastError();
}

template <int WM, int WN, int TM, int TN>
cudaError_t launchTiledAsync(dim3 grid, int threads, cudaStream_t s, const int8_t *a,
                      const int8_t *b, uint32_t *t, int m, int n, int k,
                      int rank) {
    noisyGemmMmaTiledAsync<WM, WN, TM, TN>
        <<<grid, threads, 0, s>>>(a, b, nullptr, t, m, n, k, rank, false);
    return cudaGetLastError();
}

/**
 * The raw-mma kernel now stages kKB of k at a time and pads BOTH shared
 * strides, which puts it past the 48 KB static shared limit, so its shared
 * memory is dynamic and the opt-in has to be requested once per kernel.
 * A process-global success flag is not valid for different device contexts,
 * and concurrent workers must not race on it. Set the attribute on the current
 * context and propagate failure. Context-scoped caching is a later measured
 * optimization; it must not cache a failed request as success.
 */
template<bool Early=false> cudaError_t launchTmaSeparate(dim3 grid, int threads, cudaStream_t s, const int8_t *a,
               const int8_t *b, uint32_t *t, int m, int n, int k, int rank) {
        if(!a || !b || !t || k!=8192 || rank!=128 ||
           !((m==16384 && n==131072) || (m==256 && n==256)) ||
           threads!=512 || grid.x!=static_cast<unsigned>(n/256) ||
           grid.y!=static_cast<unsigned>(m/128) || grid.z!=1)
            return cudaErrorInvalidValue;
        int device=-1; cudaDeviceProp properties{};
        cudaError_t error=cudaGetDevice(&device);
        if(error!=cudaSuccess) return error;
        error=cudaGetDeviceProperties(&properties,device);
        if(error!=cudaSuccess) return error;
        cudaFuncAttributes attributes{};
        error=cudaFuncGetAttributes(&attributes,tma_delivery::kernel<false,Early>);
        if(error!=cudaSuccess) return error;
        if(properties.major!=12 || properties.minor!=0 ||
           properties.maxThreadsPerBlock<512 ||
           properties.sharedMemPerBlockOptin<tma_delivery::SharedBytes+attributes.sharedSizeBytes)
            return cudaErrorNotSupported;
        CUtensorMap aMap{},bMap{};
        if(tma_delivery::encode(&aMap,const_cast<int8_t*>(a),m,128)!=CUDA_SUCCESS ||
           tma_delivery::encode(&bMap,const_cast<int8_t*>(b),n,256)!=CUDA_SUCCESS)
            return cudaErrorInvalidValue;
        error=cudaFuncSetAttribute(tma_delivery::kernel<false,Early>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,tma_delivery::SharedBytes);
        if(error!=cudaSuccess) return error;
        tma_delivery::kernel<false,Early><<<grid,threads,tma_delivery::SharedBytes,s>>>(
            aMap,bMap,nullptr,t,m,n);
        return cudaGetLastError();
}
cudaError_t launchTmaDelivery(dim3 grid,int threads,cudaStream_t s,const int8_t* a,
    const int8_t* b,uint32_t* t,int m,int n,int k,int rank) {
    return launchTmaSeparate<false>(grid,threads,s,a,b,t,m,n,k,rank);
}
template<bool Early=false,bool AsymmetricB=false> cudaError_t launchTmaFused(dim3 grid,cudaStream_t stream,const int8_t* a,
    const int8_t* b,int m,int n,tma_delivery::ScanOutput out) {
    if(!a||!b||!out.key||!out.target||!out.indices||!out.digests||!out.count||
       !out.retainedTranscripts||out.capacity!=kMaxHits||m!=16384||n!=131072||
       grid.x!=unsigned(n/256)||grid.y!=unsigned(m/128)||grid.z!=1)
        return cudaErrorInvalidValue;
    int device=-1; cudaDeviceProp prop{}; cudaFuncAttributes attr{};
    cudaError_t error=cudaGetDevice(&device);
    if(error==cudaSuccess) error=cudaGetDeviceProperties(&prop,device);
    constexpr int sharedBytes=AsymmetricB?tma_delivery::AsymSharedBytes:tma_delivery::SharedBytes;
    if(error==cudaSuccess) error=cudaFuncGetAttributes(&attr,tma_delivery::kernel<true,Early,AsymmetricB>);
    if(error!=cudaSuccess) return error;
    if(prop.major!=12||prop.minor!=0||prop.maxThreadsPerBlock<512||
       prop.sharedMemPerBlockOptin<sharedBytes+attr.sharedSizeBytes)
        return cudaErrorNotSupported;
    CUtensorMap aMap{},bMap{};
    if(tma_delivery::encode(&aMap,const_cast<int8_t*>(a),m,128)!=CUDA_SUCCESS||
       tma_delivery::encode(&bMap,const_cast<int8_t*>(b),n,256,AsymmetricB?128:64)!=CUDA_SUCCESS)
        return cudaErrorInvalidValue;
    error=cudaFuncSetAttribute(tma_delivery::kernel<true,Early,AsymmetricB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,sharedBytes);
    if(error!=cudaSuccess) return error;
    tma_delivery::kernel<true,Early,AsymmetricB><<<grid,512,sharedBytes,stream>>>(
        aMap,bMap,nullptr,nullptr,m,n,out);
    return cudaGetLastError();
}
cudaError_t launchTmaCombined(cudaStream_t stream,const int8_t* a,const int8_t* b,
    int m,int n,tma_delivery::ScanOutput out) {
    if(!a||!b||!out.key||!out.target||!out.indices||!out.digests||!out.count||
       !out.retainedTranscripts||out.capacity!=kMaxHits||m!=16384||n!=131072)
        return cudaErrorInvalidValue;
    int device=-1;cudaDeviceProp prop{};cudaFuncAttributes attr{};
    cudaError_t error=cudaGetDevice(&device);
    if(error==cudaSuccess) error=cudaGetDeviceProperties(&prop,device);
    if(error==cudaSuccess) error=cudaFuncGetAttributes(&attr,noisyGemmN64Combined);
    if(error!=cudaSuccess) return error;
    if(prop.major!=12||prop.minor!=0||prop.maxThreadsPerBlock<n64::Threads||attr.maxThreadsPerBlock<n64::Threads||
       prop.sharedMemPerBlockOptin<n64::SharedBytes+attr.sharedSizeBytes)
        return cudaErrorNotSupported;
    error=cudaFuncSetAttribute(noisyGemmN64Combined,cudaFuncAttributeMaxDynamicSharedMemorySize,n64::SharedBytes);
    if(error!=cudaSuccess) return error;
    noisyGemmN64Combined<<<n64::grid(m,n),n64::Threads,n64::SharedBytes,stream>>>(
        a,b,nullptr,m,n,8192,128,false,out);
    return cudaGetLastError();
}
__global__ void gatherTmaRetainedTranscripts(const uint32_t* full,const uint32_t* indices,
                                            uint32_t* compact,uint32_t retained) {
    const unsigned item=threadIdx.x+blockIdx.x*blockDim.x;
    if(item<retained*16) compact[item]=full[size_t(indices[item/16])*16+item%16];
}
bool tmaDeliveryTileFilterMatches(const char *tile) noexcept {
    return tile && (!std::strcmp(tile,kCudaTmaDeliveryPlan.name) ||
                    !std::strcmp(tile,kPearlExactK8192Tile));
}
template <int WM, int WN, int TM, int TN, int KKB, int ST = 3, int GM = 8>
cudaError_t launchPtx(dim3 grid, int threads, cudaStream_t s, const int8_t *a,
               const int8_t *b, uint32_t *t, int m, int n, int k, int rank) {
    constexpr int kStages = ST;
    constexpr int aStride = KKB + 16, bStride = KKB + 16;
    constexpr int blockM = WM * TM * 16, blockN = WN * TN * 16;
    constexpr int smem = kStages * (blockM * aStride + blockN * bStride);
    const cudaError_t attributeError = cudaFuncSetAttribute(
        noisyGemmPtx<WM, WN, TM, TN, KKB, ST, GM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (attributeError != cudaSuccess) return attributeError;
    noisyGemmPtx<WM, WN, TM, TN, KKB, ST, GM>
        <<<grid, threads, smem, s>>>(a, b, nullptr, t, m, n, k, rank, false);
    return cudaGetLastError();
}

template <CudaPreparationPlanToken Token>
constexpr TileConfig makeTileConfig() noexcept {
    constexpr const auto &plan = cudaPreparationPlan<Token>();
    if constexpr (plan.family == kSingle) {
        return {plan, &launchTiled<plan.wm, plan.wn, plan.tm, plan.tn>};
    } else if constexpr (plan.family == kDoubleBuffered) {
        return {plan, &launchTiledDB<plan.wm, plan.wn, plan.tm, plan.tn>};
    } else if constexpr (plan.family == kAsync) {
        return {plan, &launchTiledAsync<plan.wm, plan.wn, plan.tm, plan.tn>};
    } else {
        static_assert(plan.family == kPtx, "unknown CUDA launch family");
        return {plan, &launchPtx<plan.wm, plan.wn, plan.tm, plan.tn,
                                plan.blockK, plan.stages, plan.groupM>};
    }
}

constexpr TileConfig kTileConfigs[] = {
    makeTileConfig<CudaPreparationPlanToken::SingleWm2Wn4Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::SingleWm2Wn4Tm4Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::SingleWm2Wn4Tm2Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::SingleWm2Wn4Tm4Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::SingleWm4Wn2Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::SingleWm4Wn4Tm2Tn2>(),
    // Double-buffered: prefetch the next k-slice while the tensor cores chew
    // the current one. 1.6x the single-buffered kernel in isolation.
    makeTileConfig<CudaPreparationPlanToken::DoubleBufferedWm2Wn4Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::DoubleBufferedWm2Wn4Tm4Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::DoubleBufferedWm2Wn4Tm2Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::DoubleBufferedWm2Wn4Tm4Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::DoubleBufferedWm4Wn2Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::DoubleBufferedWm4Wn4Tm2Tn2>(),
    // cp.async, four stages. Ampere and later only - on an older card the
    // launch fails with "invalid device function" and pickTileConfig() skips
    // it, which is exactly the behaviour wanted rather than a build-time split.
    makeTileConfig<CudaPreparationPlanToken::AsyncWm2Wn4Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::AsyncWm2Wn4Tm4Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::AsyncWm2Wn4Tm2Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::AsyncWm2Wn4Tm4Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::AsyncWm4Wn2Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::AsyncWm4Wn4Tm2Tn2>(),
    // Raw mma.m16n8k32. WMMA's m16n16k16 needs two issues to cover the same
    // k that Ampere's native shape does in one, and the profiler had Tensor as
    // the busiest pipeline, so halving the issues is the point.
    // A DEEPER SHARED K-BLOCK IS THE SINGLE BIGGEST WIN IN THIS KERNEL.
    // Pearl's own reference GEMM stages 128 of k at a time; this staged 32,
    // which is the mma's k and no more. At 32 the loop takes a barrier every
    // 32 of k and each row's global read is 32 contiguous bytes out of a
    // 2048-byte row. At 64 it is 420 TOPS against 289 for the same geometry,
    // and 361 for the older static-shared kernel it replaced. 128 does not
    // fit three stages of padded shared, so it is not offered.
    // k32 is kept below it for cards where the deeper block will not fit.
    makeTileConfig<CudaPreparationPlanToken::PtxBk64Wm2Wn4Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk64Wm2Wn4Tm4Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk64Wm2Wn4Tm2Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk64Wm2Wn4Tm4Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk64Wm4Wn2Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk64Wm4Wn4Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk64Wm4Wn4Tm4Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk64Wm4Wn4Tm2Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk64Wm8Wn2Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk64Wm4Wn8Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk32Wm2Wn4Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk32Wm2Wn4Tm4Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk32Wm2Wn4Tm2Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk32Wm2Wn4Tm4Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk32Wm4Wn2Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk32Wm4Wn4Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk32Wm4Wn4Tm4Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk32Wm4Wn4Tm2Tn4>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk32Wm8Wn2Tm2Tn2>(),
    makeTileConfig<CudaPreparationPlanToken::PtxBk32Wm4Wn8Tm2Tn2>(),
    // Measured and REJECTED, so not carried as candidates (each one costs
    // tuning time on every start). All judged end to end, on both a 4090 and
    // a 5080, by adding them here and seeing whether the tuner ever chose one:
    //   k128 at 2 stages (2x4/4x2, 4x4/2x2) - never chosen. Only fits a
    //     128x128 block, and the smaller block costs more than the depth pays.
    //   k64 at 2 stages (three geometries)  - never chosen. The third stage is
    //     worth more than the shared memory it frees.
    //   swizzle group 2, 4, 16, 32          - never chosen; the default 8 won
    //     on both cards despite their different L2 sizes.
    // kStagesP and kGroupMP stay template parameters so this is re-testable.
};

constexpr bool cudaTileCatalogMatchesDescriptorCatalog() noexcept {
    if (sizeof(kTileConfigs) / sizeof(kTileConfigs[0]) !=
        kCudaPreparationPlanCount)
        return false;
    for (size_t i = 0; i < kCudaPreparationPlanCount; ++i)
        if (!kTileConfigs[i].launch ||
            !cudaPreparationPlanDescriptorEqual(kTileConfigs[i],
                                                kCudaPreparationPlans[i]))
            return false;
    return true;
}

static_assert(cudaTileCatalogMatchesDescriptorCatalog(),
              "CUDA launch rows must preserve the host descriptor catalog");

// Separate from the unchanged 38-row default array: normal tuning never visits
// this kernel, even as a correctness witness. This index is internal only;
// public identity is the permanent descriptor token/plan ID.
constexpr size_t kFixedExperimentTile = kCudaPreparationPlanCount;
constexpr size_t kWm8ExperimentTile = kCudaPreparationPlanCount + 1;
constexpr size_t kWm8Cap112ExperimentTile = kCudaPreparationPlanCount + 2;
constexpr size_t kWm8FixedExperimentTile = kCudaPreparationPlanCount + 3;
constexpr size_t kBk32FixedExperimentTile = kCudaPreparationPlanCount + 5;
constexpr size_t kWm4ReuseExperimentTile = kCudaPreparationPlanCount + 6;
constexpr size_t kTmaWm8ExperimentTile = kCudaPreparationPlanCount + 7;
constexpr size_t kTmaTraversalExperimentTile = kCudaPreparationPlanCount + 8;
constexpr size_t kTmaStage4ExperimentTile = kCudaPreparationPlanCount + 10;
constexpr size_t kTmaDeliveryTile = kCudaPreparationPlanCount + 11;
constexpr TileConfig kTmaDeliveryConfig{kCudaTmaDeliveryPlan, &launchTmaDelivery};
constexpr size_t experimentTile(CudaFixedExperimentSelection selection) {
    if (selection == CudaFixedExperimentSelection::TmaTraversal16x32Stage4) return kTmaStage4ExperimentTile;
    if (selection == CudaFixedExperimentSelection::TmaWm8FixedK2048R128) return kTmaWm8ExperimentTile;
    if (selection == CudaFixedExperimentSelection::TmaTraversal16x32) return kTmaTraversalExperimentTile;
    if (selection == CudaFixedExperimentSelection::Wm4ReuseFixedK2048R128) return kWm4ReuseExperimentTile;
    if (selection == CudaFixedExperimentSelection::Bk32FixedK2048R128) return kBk32FixedExperimentTile;
    if (selection == CudaFixedExperimentSelection::Wm8FixedK2048R128) return kWm8FixedExperimentTile;
    if (selection == CudaFixedExperimentSelection::Wm8InlineCap112) return kWm8Cap112ExperimentTile;
    return selection == CudaFixedExperimentSelection::Wm8Runtime ? kWm8ExperimentTile : kFixedExperimentTile;
}
#if defined(CANARY_PEARL_ENABLE_FIXED_SCHEDULE_EXPERIMENT) && CANARY_PEARL_ENABLE_FIXED_SCHEDULE_EXPERIMENT == 1
cudaError_t launchFixedExperiment(dim3 grid, int threads, cudaStream_t stream,
    const int8_t *a, const int8_t *b, uint32_t *transcripts,
    int m, int n, int k, int rank) {
    if (!a || !b || !transcripts || threads != 512 || grid.z != 1 ||
        !cudaFixedExperimentLaunchValid(120, m, n, k, rank) ||
        grid.x != static_cast<unsigned>(n / 256) ||
        grid.y != static_cast<unsigned>(m / 128)) return cudaErrorInvalidValue;
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess) error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    if (!cudaFixedExperimentLaunchValid(properties.major * 10 + properties.minor,
                                         m, n, k, rank)) return cudaErrorInvalidDeviceFunction;
    constexpr int shared = 3 * (128 + 256) * (64 + 16);
    error = cudaFuncSetAttribute(noisyGemmPtx<4, 4, 2, 4, 64, 3, 8, true>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (error != cudaSuccess) return error;
    noisyGemmPtx<4, 4, 2, 4, 64, 3, 8, true><<<grid, threads, shared, stream>>>(
        a, b, nullptr, transcripts, m, n, k, rank, false);
    return cudaGetLastError();
}
constexpr TileConfig kFixedExperimentConfig{kCudaFixedExperimentPlan, &launchFixedExperiment};
#endif
#if defined(CANARY_PEARL_ENABLE_WM8_EXPERIMENT) && CANARY_PEARL_ENABLE_WM8_EXPERIMENT == 1
cudaError_t launchWm8Experiment(dim3 grid, int threads, cudaStream_t stream,
    const int8_t *a, const int8_t *b, uint32_t *transcripts,
    int m, int n, int k, int rank) {
    if (!a || !b || !transcripts || threads != 512 || grid.z != 1 ||
        !cudaFixedExperimentLaunchValid(120, m, n, k, rank) ||
        grid.x != static_cast<unsigned>(n / 128) ||
        grid.y != static_cast<unsigned>(m / 256)) return cudaErrorInvalidValue;
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess) error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    if (!cudaFixedExperimentLaunchValid(properties.major * 10 + properties.minor,
                                         m, n, k, rank)) return cudaErrorInvalidDeviceFunction;
    constexpr int shared = 3 * (256 + 128) * (64 + 16);
    error = cudaFuncSetAttribute(noisyGemmPtx<8, 2, 2, 4, 64, 3, 8, false, false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (error != cudaSuccess) return error;
    noisyGemmPtx<8, 2, 2, 4, 64, 3, 8, false, false><<<grid, threads, shared, stream>>>(
        a, b, nullptr, transcripts, m, n, k, rank, false);
    return cudaGetLastError();
}
constexpr TileConfig kWm8ExperimentConfig{kCudaWm8ExperimentPlan, &launchWm8Experiment};
#endif
#if defined(CANARY_PEARL_ENABLE_WM8_CAP112_EXPERIMENT)
cudaError_t launchWm8Cap112Experiment(dim3 grid, int threads, cudaStream_t stream,
    const int8_t *a, const int8_t *b, uint32_t *transcripts,
    int m, int n, int k, int rank) {
    if (!a || !b || !transcripts || threads != 512 || grid.z != 1 ||
        !cudaFixedExperimentLaunchValid(120, m, n, k, rank) ||
        grid.x != static_cast<unsigned>(n / 128) ||
        grid.y != static_cast<unsigned>(m / 256)) return cudaErrorInvalidValue;
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess) error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    if (!cudaFixedExperimentLaunchValid(properties.major * 10 + properties.minor,
                                         m, n, k, rank)) return cudaErrorInvalidDeviceFunction;
    constexpr int shared = 3 * (256 + 128) * (64 + 16);
    error = cudaFuncSetAttribute(noisyGemmPtxWm8InlineCap112,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (error != cudaSuccess) return error;
    noisyGemmPtxWm8InlineCap112<<<grid, threads, shared, stream>>>(
        a, b, nullptr, transcripts, m, n, k, rank, false);
    return cudaGetLastError();
}
constexpr TileConfig kWm8Cap112ExperimentConfig{kCudaWm8Cap112ExperimentPlan, &launchWm8Cap112Experiment};
#endif

#if defined(CANARY_PEARL_ENABLE_WM8_FIXED_EXPERIMENT)
cudaError_t launchWm8FixedExperiment(dim3 grid, int threads, cudaStream_t stream,
    const int8_t *a, const int8_t *b, uint32_t *transcripts,
    int m, int n, int k, int rank) {
    if (!a || !b || !transcripts || threads != 512 || grid.z != 1 ||
        !cudaFixedExperimentLaunchValid(120, m, n, k, rank) ||
        grid.x != static_cast<unsigned>(n / 128) ||
        grid.y != static_cast<unsigned>(m / 256)) return cudaErrorInvalidValue;
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess) error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    if (!cudaFixedExperimentLaunchValid(properties.major * 10 + properties.minor,
                                         m, n, k, rank)) return cudaErrorInvalidDeviceFunction;
    constexpr int shared = 3 * (256 + 128) * (64 + 16);
    if (properties.maxThreadsPerBlock < 512 || properties.sharedMemPerBlockOptin < shared)
        return cudaErrorInvalidConfiguration;
    error = cudaFuncSetAttribute(noisyGemmPtx<8, 2, 2, 4, 64, 3, 8, true, false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (error != cudaSuccess) return error;
    noisyGemmPtx<8, 2, 2, 4, 64, 3, 8, true, false><<<grid, threads, shared, stream>>>(
        a, b, nullptr, transcripts, m, n, k, rank, false);
    return cudaGetLastError();
}
constexpr TileConfig kWm8FixedExperimentConfig{kCudaWm8FixedExperimentPlan, &launchWm8FixedExperiment};
#endif



#if defined(CANARY_PEARL_ENABLE_BK32_FIXED_EXPERIMENT)
cudaError_t launchBk32FixedExperiment(dim3 grid, int threads, cudaStream_t stream,
    const int8_t *a, const int8_t *b, uint32_t *transcripts,
    int m, int n, int k, int rank) {
    constexpr auto selection = CudaFixedExperimentSelection::Bk32FixedK2048R128;
    if (!a || !b || !transcripts || threads != 256 || grid.z != 1 ||
        !cudaSelectedExperimentLaunchValid(selection, 120, m, n, k, rank) ||
        grid.x != static_cast<unsigned>(n / 128) ||
        grid.y != static_cast<unsigned>(m / 128)) return cudaErrorInvalidValue;
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess) error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    if (!cudaSelectedExperimentLaunchValid(selection, properties.major * 10 + properties.minor,
                                          m, n, k, rank)) return cudaErrorInvalidDeviceFunction;
    constexpr int shared = 3 * (128 + 128) * (32 + 16);
    if (properties.maxThreadsPerBlock < 256 || properties.sharedMemPerBlockOptin < shared)
        return cudaErrorInvalidConfiguration;
    error = cudaFuncSetAttribute(noisyGemmPtx<2, 4, 4, 2, 32, 3, 8, true, false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (error != cudaSuccess) return error;
    noisyGemmPtx<2, 4, 4, 2, 32, 3, 8, true, false><<<grid, threads, shared, stream>>>(
        a, b, nullptr, transcripts, m, n, k, rank, false);
    return cudaGetLastError();
}
constexpr TileConfig kBk32FixedExperimentConfig{kCudaBk32FixedExperimentPlan, &launchBk32FixedExperiment};
#endif

#if defined(CANARY_PEARL_ENABLE_WM4_REUSE_EXPERIMENT)
cudaError_t launchWm4ReuseExperiment(dim3 grid, int threads, cudaStream_t stream,
    const int8_t *a, const int8_t *b, uint32_t *transcripts,
    int m, int n, int k, int rank) {
    constexpr auto selection = CudaFixedExperimentSelection::Wm4ReuseFixedK2048R128;
    if (!a || !b || !transcripts || threads != 256 || grid.z != 1 ||
        !cudaSelectedExperimentLaunchValid(selection, 120, m, n, k, rank) ||
        grid.x != static_cast<unsigned>(n / 128) ||
        grid.y != static_cast<unsigned>(m / 256)) return cudaErrorInvalidValue;
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess) error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    if (!cudaSelectedExperimentLaunchValid(selection, properties.major * 10 + properties.minor,
                                          m, n, k, rank)) return cudaErrorInvalidDeviceFunction;
    constexpr int shared = 3 * (256 + 128) * (64 + 16);
    if (properties.maxThreadsPerBlock < 256 || properties.sharedMemPerBlockOptin < shared)
        return cudaErrorInvalidConfiguration;
    error = cudaFuncSetAttribute(noisyGemmPtx<4, 2, 4, 4, 64, 3, 8, true, false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (error != cudaSuccess) return error;
    noisyGemmPtx<4, 2, 4, 4, 64, 3, 8, true, false><<<grid, threads, shared, stream>>>(
        a, b, nullptr, transcripts, m, n, k, rank, false);
    return cudaGetLastError();
}
constexpr TileConfig kWm4ReuseExperimentConfig{kCudaWm4ReuseExperimentPlan, &launchWm4ReuseExperiment};
#endif

#if defined(CANARY_PEARL_ENABLE_TMA_WM8_EXPERIMENT)
template<bool Traversal16x32=false>
cudaError_t launchTmaWm8Experiment(dim3 grid, int threads, cudaStream_t stream,
    const int8_t *a, const int8_t *b, uint32_t *transcripts,
    int m, int n, int k, int rank) {
    constexpr auto selection = Traversal16x32 ? CudaFixedExperimentSelection::TmaTraversal16x32 :
                                               CudaFixedExperimentSelection::TmaWm8FixedK2048R128;
    if (!a || !b || !transcripts || threads != 512 || grid.z != 1 ||
        (reinterpret_cast<uintptr_t>(a) & 15) || (reinterpret_cast<uintptr_t>(b) & 15) ||
        (reinterpret_cast<uintptr_t>(transcripts) & 3) ||
        !cudaSelectedExperimentLaunchValid(selection, 120, m, n, k, rank) ||
        grid.x != static_cast<unsigned>(n / 128) ||
        grid.y != static_cast<unsigned>(m / 256)) return cudaErrorInvalidValue;
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess) error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    if (!cudaSelectedExperimentLaunchValid(selection, properties.major * 10 + properties.minor,
                                          m, n, k, rank)) return cudaErrorInvalidDeviceFunction;
    if (properties.maxThreadsPerBlock < 512 || properties.sharedMemPerBlockOptin < tma_wm8::SharedBytes)
        return cudaErrorInvalidConfiguration;
    // Immutable by-value kernel arguments bind this launch's actual buffers.
    // No global/cache state survives A replacement, job changes, or release().
    CUtensorMap aMap{}, bMap{};
    if (tma_wm8::encode(&aMap, const_cast<int8_t*>(a), m, 256) != CUDA_SUCCESS ||
        tma_wm8::encode(&bMap, const_cast<int8_t*>(b), n, 128) != CUDA_SUCCESS)
        return cudaErrorInvalidValue;
    error = cudaFuncSetAttribute(tma_wm8::kernel<Traversal16x32>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, tma_wm8::SharedBytes);
    if (error != cudaSuccess) return error;
    tma_wm8::kernel<Traversal16x32><<<grid, threads, tma_wm8::SharedBytes, stream>>>(
        aMap, bMap, nullptr, transcripts, m, n);
    return cudaGetLastError();
}
constexpr TileConfig kTmaWm8ExperimentConfig{kCudaTmaWm8ExperimentPlan, &launchTmaWm8Experiment<false>};
#if defined(CANARY_PEARL_ENABLE_TMA_TRAVERSAL_EXPERIMENT)
constexpr TileConfig kTmaTraversalExperimentConfig{kCudaTmaTraversalExperimentPlan, &launchTmaWm8Experiment<true>};
#endif
#endif

#if CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT == 1
cudaError_t launchTmaStage4Experiment(dim3 grid, int threads, cudaStream_t stream,
    const int8_t *a, const int8_t *b, uint32_t *transcripts,
    int m, int n, int k, int rank) {
    constexpr auto selection = CudaFixedExperimentSelection::TmaTraversal16x32Stage4;
    constexpr int shared = tma_stage4_probe::SharedBytes;
    static_assert(tma_stage4_probe::Stages == 4 && shared == 98336,
                  "exact packed TMA stage4 allocation including four barriers");
    if (!a || !b || !transcripts || threads != 512 || grid.z != 1 ||
        (reinterpret_cast<uintptr_t>(a) & 15) || (reinterpret_cast<uintptr_t>(b) & 15) ||
        (reinterpret_cast<uintptr_t>(transcripts) & 3) ||
        !cudaSelectedExperimentLaunchValid(selection, 120, m, n, k, rank) ||
        grid.x != static_cast<unsigned>(n / 128) ||
        grid.y != static_cast<unsigned>(m / 256)) return cudaErrorInvalidValue;
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess) error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    if (!cudaSelectedExperimentLaunchValid(selection, properties.major * 10 + properties.minor,
                                          m, n, k, rank)) return cudaErrorInvalidDeviceFunction;
    if (properties.maxThreadsPerBlock < 512 || properties.sharedMemPerBlockOptin < shared)
        return cudaErrorInvalidConfiguration;
    // Per-launch by-value descriptors bind the actual owned A/B allocations.
    // Preserve the original encoder and keep host descriptor alignment explicit.
    alignas(64) CUtensorMap aMap{}, bMap{};
    if (tma_wm8::encode(&aMap, const_cast<int8_t*>(a), m, 256) != CUDA_SUCCESS ||
        tma_wm8::encode(&bMap, const_cast<int8_t*>(b), n, 128) != CUDA_SUCCESS)
        return cudaErrorInvalidValue;
    error = cudaFuncSetAttribute(tma_stage4_probe::kernel<true>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (error != cudaSuccess) return error;
    tma_stage4_probe::kernel<true><<<grid, threads, shared, stream>>>(
        aMap, bMap, nullptr, transcripts, m, n);
    return cudaGetLastError();
}
constexpr TileConfig kTmaStage4ExperimentConfig{kCudaTmaStage4ExperimentPlan, &launchTmaStage4Experiment};
#endif

const TileConfig &tileConfig(size_t index) {
    if (index == kTmaDeliveryTile) return kTmaDeliveryConfig;
#if CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT == 1
    if (index == kTmaStage4ExperimentTile) return kTmaStage4ExperimentConfig;
#endif
#if defined(CANARY_PEARL_ENABLE_TMA_TRAVERSAL_EXPERIMENT)
    if (index == kTmaTraversalExperimentTile) return kTmaTraversalExperimentConfig;
#endif
#if defined(CANARY_PEARL_ENABLE_TMA_WM8_EXPERIMENT)
    if (index == kTmaWm8ExperimentTile) return kTmaWm8ExperimentConfig;
#endif
#if defined(CANARY_PEARL_ENABLE_WM4_REUSE_EXPERIMENT)
    if (index == kWm4ReuseExperimentTile) return kWm4ReuseExperimentConfig;
#endif
#if defined(CANARY_PEARL_ENABLE_BK32_FIXED_EXPERIMENT)
    if (index == kBk32FixedExperimentTile) return kBk32FixedExperimentConfig;
#endif
#if defined(CANARY_PEARL_ENABLE_WM8_FIXED_EXPERIMENT)
    if (index == kWm8FixedExperimentTile) return kWm8FixedExperimentConfig;
#endif
#if defined(CANARY_PEARL_ENABLE_WM8_CAP112_EXPERIMENT)
    if (index == kWm8Cap112ExperimentTile) return kWm8Cap112ExperimentConfig;
#endif
#if defined(CANARY_PEARL_ENABLE_FIXED_SCHEDULE_EXPERIMENT) && CANARY_PEARL_ENABLE_FIXED_SCHEDULE_EXPERIMENT == 1
    if (index == kFixedExperimentTile) return kFixedExperimentConfig;
#endif
#if defined(CANARY_PEARL_ENABLE_WM8_EXPERIMENT) && CANARY_PEARL_ENABLE_WM8_EXPERIMENT == 1
    if (index == kWm8ExperimentTile) return kWm8ExperimentConfig;
#endif
    return kTileConfigs[index];
}

/** A malformed or ambiguous formal catalog invalidates every formal run. */
bool formalBenchmarkCatalogIsUnique() {
    const size_t shapeCount = sizeof(kShapes) / sizeof(kShapes[0]);
    for (size_t i = 0; i < shapeCount; ++i) {
        const Shape &shape = kShapes[i];
        if (!canonicalBenchmarkId(shape.id) ||
            std::string(shape.id) != "m" + std::to_string(shape.m) + "-n" +
                                         std::to_string(shape.n))
            return false;
        for (size_t j = i + 1; j < shapeCount; ++j)
            if (std::string_view(shape.id) == kShapes[j].id) return false;
    }

    const size_t tileCount = sizeof(kTileConfigs) / sizeof(kTileConfigs[0]);
    for (size_t i = 0; i < tileCount; ++i) {
        const TileConfig &tile = kTileConfigs[i];
        if (!canonicalBenchmarkId(tile.kernelId) ||
            !canonicalBenchmarkId(tile.tileId))
            return false;
        for (size_t j = i + 1; j < tileCount; ++j) {
            if (std::string_view(tile.kernelId) == kTileConfigs[j].kernelId &&
                std::string_view(tile.tileId) == kTileConfigs[j].tileId)
                return false;
        }
    }
    return true;
}

class PearlPow : public Algorithm {
    struct ASlot {
        int8_t *raw = nullptr;
        int8_t *noised = nullptr;
        uint8_t *root = nullptr;
        uint8_t *commit = nullptr;
    };
   public:
    PearlPow() = default;

    ~PearlPow() override {
        release();
        profiler_.close();
        if (stream_) cudaStreamDestroy(stream_);
        fprintf(stderr,"[pearl-tma-scan] mode=%s early_attempts=%llu fused_attempts=%llu recovery_attempts=%llu completed_attempts=%llu\n",
            tma_fused::name(fusedMode_), (unsigned long long)earlyAttempts_, (unsigned long long)fusedAttempts_,
            (unsigned long long)fusedRecoveries_, (unsigned long long)completedAttempts_);
    }

    const char *name() const override { return "pearl-pow"; }
    PearlRuntimeCapability pearlRuntimeCapability() const noexcept override {
        return PearlRuntimeCapability::CanonicalCudaPoolV1;
    }
    void setPreparationStopFlag(volatile sig_atomic_t *stop) noexcept override {
        preparationCancellation_.bind(stop);
    }
    bool pearlIntegratedPipelineCompiled() const noexcept override {
        return kIntegratedPipelineCompiled;
    }
    bool pearlExperimentConfigurationValid(
        const RunOptions &options, const char *selection) const noexcept override {
        if(getenv("CANARY_PEARL_TMA_FUSED_SCAN") || getenv("CANARY_PEARL_TMA_REUSE") ||
           getenv("CANARY_PEARL_TMA_SCAN_TIMING")) return false;
        const bool tmaRequested=tmaDeliveryTileFilterMatches(getenv("CANARY_PEARL_TILE"));
        const bool productFixed = pearlProductFixedSelectionActive();
        if (productFixed && !pearlProductFixedSelectionMatches(
                getenv("CANARY_PEARL_SHAPE"), getenv("CANARY_PEARL_TILE"))) return false;
#if defined(CANARY_PEARL_ENABLE_TMA_EARLY_FUSION) && CANARY_PEARL_ENABLE_TMA_EARLY_FUSION == 1
        constexpr bool fusedCompiled=true;
#else
        constexpr bool fusedCompiled=false;
#endif
        if(tma_fused::parse(getenv("CANARY_PEARL_TMA_DATAFLOW"))!=fusedMode_ ||
           (productFixed && !tma_fused::productModeAllowed(fusedMode_,true,fusedCompiled,tmaRequested)) ||
           (!productFixed && !tma_fused::admitted(fusedMode_,fusedCompiled,tmaRequested,
                tma_fused::ownedLoopbackGuard(options),
                options.pearlAcceptedQuota || options.pearlSustainedSeconds,
                attemptOverlapEnabled(overlapMode_),options.pearlProduction))) return false;
        if(options.pearlPublicExecution!=publicExecution_ || !pearlPublicEnvironment(publicExecution_))return false;
        if(pearlPublicEnabled(publicExecution_)) {
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
            if(!pearlPublicPolicy(publicExecution_,options.pearlGuardRetirementSession && options.poolHost=="127.0.0.1",
                options.pearlGuardRetirementSession?options.pearlGuardRetirementSession->cap():0,
                options.batch,options.pearlAcceptedQuota,options.pearlSustainedSeconds,options.pearlSustainedPublic))return false;
#else
            return false;
#endif
        }
        if (options.pearlPrivateExecution!=privateExecution_ || (!pearlPrivateEnvironment(privateExecution_) || !pearlPublicEnvironment(publicExecution_))) return false;
        if (pearlPrivateEnabled(privateExecution_)) {
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
            if (!options.pearlGuardRetirementSession || options.poolHost!="127.0.0.1" ||
                options.pearlSustainedPublic || options.batch!=pearlPrivateBatch(privateExecution_) ||
                (!options.pearlAcceptedQuota && !options.pearlSustainedSeconds)) return false;
#else
            return false;
#endif
        }
        return fusedScanAdmission(options) && integratedAdmission(options) && fixedWm8EnvironmentValid() && rootFusionEnvironmentValid() &&
            aRootFusionAdmission(rootFusionMode_, kARootFusionCompiled,
                options.formalBenchmark.enabled,
                rootFusionProfilingCaptured_ || getenv("CANARY_PEARL_PROFILE_INTERVAL") != nullptr,
                options.bench, options.poolHost, options.poolPort,
                experimentSelection_ == CudaFixedExperimentSelection::Wm8Runtime) &&
            parseAttemptOverlap(getenv("CANARY_PEARL_ATTEMPT_OVERLAP")) == overlapMode_ &&
            attemptOverlapAdmission(overlapMode_, kAttemptOverlapCompiled,
                options.formalBenchmark.enabled,
                getenv("CANARY_PEARL_PROFILE_INTERVAL") != nullptr,
                options.bench,
                options.poolHost, options.poolPort) &&
            parseCudaFixedExperiment(selection) == experimentSelection_ &&
            cudaFixedExperimentConfigurationValid(
                experimentSelection_, experimentCompiled(experimentSelection_),
                options.formalBenchmark.enabled, options.bench,
                options.poolHost, options.poolPort,
                getenv("CANARY_PEARL_SHAPE") != nullptr,
                getenv("CANARY_PEARL_TILE") != nullptr,
                getenv("CANARY_PEARL_ATTEMPT_OVERLAP") != nullptr,
                getenv("CANARY_PEARL_A_ROOT_FUSION") != nullptr,
                rootFusionProfilingCaptured_ || getenv("CANARY_PEARL_PROFILE_INTERVAL") != nullptr);
    }
    std::string launcherGlob() const override { return "mine_pearl_*"; }

    /**
     * Pearl's job does not fit `Job`'s fixed fields: a 76-byte header stub and
     * a certificate version ride in `extra`. Benchmark mode builds a generic
     * job and cannot know that, so `--bench --algo pearl-pow` used to reach
     * `prepare()`, print "job carries no Pearl header" and exit — a crash on
     * the first command most people type, invisible to CI until `--bench`
     * entered the gate.
     *
     * The header is the fixed pearl-synthetic-v1 fixture. It is deliberately
     * NOT a real chain header: benchmark mode has a zero target and submits
     * nothing, so a bench solution cannot be mistaken for a submittable one.
     */
    void fillBenchJob(Job *job) const override {
        // Failure leaves the output invalid; callers must never force it valid
        // or turn this no-share fixture into live work by replacing its target.
        buildPearlSyntheticBenchmarkJob(job);
    }

    void configure(const RunOptions &options) override {
        // Policy is immutable for the lifetime of this algorithm instance.
        // A caller needing a different source must create a new instance.
        if (configured_) return;
        configured_ = true;
        privateStats_.valid=executionAccountingEnabled();
        sourcePolicy_ = options.pearlPolicy;
        syntheticRun_ = options.bench;
        formalRequest_ = options.formalBenchmark;
        shapeConfigurationValid_ = options.pearlShapeExperiment == shapeExperiment_.value();
        shapeBounded_ = options.pearlAcceptedQuota || options.pearlSustainedSeconds || options.pearlProduction;
#ifdef CANARY_PEARL_GUARD_RETIREMENT_EXPERIMENT
        shapeOwnedGuard_ = options.pearlGuardRetirementSession && options.poolHost == "127.0.0.1";
#endif
        experimentConfigurationValid_ = pearlExperimentConfigurationValid(
            options, getenv("CANARY_PEARL_EXPERIMENT"));
    }

    BenchmarkIdentity benchmarkIdentity() const override { return identity_; }

    PearlNativeSelection pearlNativeSelection() const noexcept override {
        return nativeSelection_.read(currentNativeSelection());
    }
    PearlPrivateExecutionStats pearlPrivateExecutionStats() const noexcept override {
        auto value=privateStats_;
        value.ready=lookahead_.ready?1:0;
        value.valid=value.valid && pearlPrivateEnvironment(privateExecution_) && pearlPublicEnvironment(publicExecution_);
        return value;
    }

    size_t memoryBytes(const Job &job) const override {
        (void)job;
        return bytesFor(shape_);
    }

    /**
     * Pick a shape that fits and allocate for it.
     *
     * Called once, because Pearl has no epoch: there is no dataset to rebuild
     * when the height moves. What DOES have to be redone per job is the B
     * side, and that is keyed on the header rather than on the epoch, so it
     * lives in search().
     */
    bool prepare(const Job &job) override {
        if (preparationCancellation_.requested()) return false;
        if (!discardPearlLookahead()) return false;
        nativeSelection_.invalidate();
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
        aRootEvidence_.invalidate();
#endif
        wins_.beginSearch();
        publication_.abandonRetryable();
        if ((attemptOverlapEnabled(overlapMode_) &&
             (parseAttemptOverlap(getenv("CANARY_PEARL_ATTEMPT_OVERLAP")) != overlapMode_ ||
              getenv("CANARY_PEARL_PROFILE_INTERVAL"))) ||
            (!pearlPrivateEnvironment(privateExecution_) || !pearlPublicEnvironment(publicExecution_)) || !fusedScanEnvironmentValid() || !integratedEnvironmentValid() || !shapeEnvironmentValid() ||
            !experimentConfigurationValid_ || !fixedWm8EnvironmentValid() || !rootFusionEnvironmentValid() ||
            (formalRequest_.enabled && getenv("CANARY_PEARL_EXPERIMENT"))) {
            identity_ = {};
            fprintf(stderr, "[pearl-pow] invalid or non-isolated experimental selection\n");
            return false;
        }
        if (formalRequest_.enabled &&
            !profiler_.formalCompatible(getenv("CANARY_PEARL_PROFILE_INTERVAL"))) {
            identity_ = {};
            fprintf(stderr, "[pearl-pow] formal benchmark rejects profiling environment\n");
            return false;
        }
        PearlJobContext context;
        if (!validateRunJob(job, &context) || publication_.poisoned())
            return false;
        if (!stream_ && cudaStreamCreate(&stream_) != cudaSuccess) {
            stream_ = nullptr;
            publication_.poison();
            return false;
        }
        if (allocated_) return completeNativePreparation();

        cudaDeviceProp prop{};
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
            arch10_ = prop.major * 10 + prop.minor;
        if (!exactStageTiming_.architecture(arch10_)) {
            fprintf(stderr, "[pearl-stage-timing] exact SM120 device required; refusing configuration\n");
            return false;
        }
        if(pearlPublicEnabled(publicExecution_) && arch10_!=static_cast<int>(pearlPublicArchitecture(publicExecution_)))
            return false; // Captured public architecture must match before tuning or allocation.
        if (attemptOverlapEnabled(overlapMode_) && arch10_ != 120)
            return false; // No architecture-range fallback.
        if (!arch10_) {
            fprintf(stderr,
                    "[pearl-pow] cannot identify the CUDA device compute "
                    "capability - refusing architecture dispatch\n");
            return false;
        }
        tuningKey_.assign(reinterpret_cast<const char *>(prop.uuid.bytes),
                          sizeof(prop.uuid.bytes));

        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) != cudaSuccess) return false;

        // Leave the driver and whatever is on screen a real margin.
        const size_t headroom = 512ULL << 20;

        if (cudaIsolatedExperimentEnabled(experimentSelection_)) {
            const Shape requested = kShapes[0];
            if (!cudaSelectedExperimentLaunchValid(experimentSelection_, arch10_, requested.m, requested.n, kK, kRank) ||
                bytesFor(requested) + headroom > freeB) return false;
            size_t selected = 0;
            if (tuneShape(requested, &selected) <= 0 || selected != experimentTile(experimentSelection_))
                return false;  // No baseline fallback, even when it is correct.
            shape_ = requested;
            tile_ = selected;
            if (!allocate()) { release(); return false; }
            allocated_ = true;
            if (!selfCheck()) { publication_.poison(); release(); return false; }
#if defined(CANARY_PEARL_ENABLE_TMA_WM8_EXPERIMENT)
            if (experimentSelection_ == CudaFixedExperimentSelection::TmaWm8FixedK2048R128 ||
                experimentSelection_ == CudaFixedExperimentSelection::TmaTraversal16x32) {
                const void* kernel = reinterpret_cast<const void*>(tma_wm8::kernel<false>);
#if defined(CANARY_PEARL_ENABLE_TMA_TRAVERSAL_EXPERIMENT)
                if (experimentSelection_ == CudaFixedExperimentSelection::TmaTraversal16x32)
                    kernel = reinterpret_cast<const void*>(tma_wm8::kernel<true>);
#endif
                cudaFuncAttributes attributes{};
                int activeBlocks = 0;
                constexpr int shared = tma_wm8::SharedBytes;
                if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess ||
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks,
                        kernel, 512, shared) != cudaSuccess ||
                    attributes.maxThreadsPerBlock < 512 || attributes.maxDynamicSharedSizeBytes < shared ||
                    activeBlocks < 1) { publication_.poison(); release(); return false; }
                fprintf(stderr, "[pearl-experiment-resource] plan=0x%016llx registers=%d local_bytes=%zu "
                    "static_shared_bytes=%zu dynamic_shared_bytes=%d threads=512 active_blocks_per_sm=%d\n",
                    static_cast<unsigned long long>(tileConfig(tile_).preparationPlanId),
                    attributes.numRegs, attributes.localSizeBytes, attributes.sharedSizeBytes, shared, activeBlocks);
            }
#endif
#if CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT == 1
            if (experimentSelection_ == CudaFixedExperimentSelection::TmaTraversal16x32Stage4) {
                constexpr int shared = tma_stage4_probe::SharedBytes;
                cudaFuncAttributes attributes{};
                int activeBlocks = 0;
                if (cudaFuncGetAttributes(&attributes, tma_stage4_probe::kernel<true>) != cudaSuccess ||
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks,
                        tma_stage4_probe::kernel<true>, 512, shared) != cudaSuccess ||
                    attributes.binaryVersion != 120 || attributes.numRegs != 128 ||
                    attributes.localSizeBytes != 8 || attributes.sharedSizeBytes != 0 ||
                    attributes.maxThreadsPerBlock != 512 || attributes.maxDynamicSharedSizeBytes < shared ||
                    activeBlocks != 1) { publication_.poison(); release(); return false; }
                fprintf(stderr, "[pearl-experiment-resource] plan=0x%016llx registers=%d local_bytes=%zu "
                    "static_shared_bytes=%zu dynamic_shared_bytes=%d threads=512 active_blocks_per_sm=%d\n",
                    static_cast<unsigned long long>(tileConfig(tile_).preparationPlanId),
                    attributes.numRegs, attributes.localSizeBytes, attributes.sharedSizeBytes, shared, activeBlocks);
            }
#endif
#if defined(CANARY_PEARL_ENABLE_WM8_FIXED_EXPERIMENT)
            if (experimentSelection_ == CudaFixedExperimentSelection::Wm8FixedK2048R128) {
                cudaFuncAttributes attributes{};
                int activeBlocks = 0;
                constexpr int shared = 92160;
                if (cudaFuncGetAttributes(&attributes, noisyGemmPtx<8,2,2,4,64,3,8,true,false>) != cudaSuccess ||
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks,
                        noisyGemmPtx<8,2,2,4,64,3,8,true,false>, 512, shared) != cudaSuccess ||
                    attributes.maxThreadsPerBlock < 512 || attributes.maxDynamicSharedSizeBytes < shared ||
                    activeBlocks < 1) { publication_.poison(); release(); return false; }
                fprintf(stderr, "[pearl-experiment-resource] plan=0x%016llx registers=%d local_bytes=%zu "
                    "static_shared_bytes=%zu dynamic_shared_bytes=%d threads=512 active_blocks_per_sm=%d\n",
                    static_cast<unsigned long long>(kCudaWm8FixedExperimentPlan.preparationPlanId),
                    attributes.numRegs, attributes.localSizeBytes, attributes.sharedSizeBytes, shared, activeBlocks);
            }
#endif
#if defined(CANARY_PEARL_ENABLE_BK32_FIXED_EXPERIMENT)
            if (experimentSelection_ == CudaFixedExperimentSelection::Bk32FixedK2048R128) {
                cudaFuncAttributes attributes{};
                int activeBlocks = 0;
                constexpr int shared = 36864;
                if (cudaFuncGetAttributes(&attributes, noisyGemmPtx<2,4,4,2,32,3,8,true,false>) != cudaSuccess ||
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks,
                        noisyGemmPtx<2,4,4,2,32,3,8,true,false>, 256, shared) != cudaSuccess ||
                    attributes.maxThreadsPerBlock < 256 || attributes.maxDynamicSharedSizeBytes < shared ||
                    activeBlocks < 1) { publication_.poison(); release(); return false; }
                fprintf(stderr, "[pearl-experiment-resource] plan=0x%016llx registers=%d local_bytes=%zu "
                    "static_shared_bytes=%zu dynamic_shared_bytes=%d threads=256 active_blocks_per_sm=%d\n",
                    static_cast<unsigned long long>(kCudaBk32FixedExperimentPlan.preparationPlanId),
                    attributes.numRegs, attributes.localSizeBytes, attributes.sharedSizeBytes, shared, activeBlocks);
            }
#endif
#if defined(CANARY_PEARL_ENABLE_WM4_REUSE_EXPERIMENT)
            if (experimentSelection_ == CudaFixedExperimentSelection::Wm4ReuseFixedK2048R128) {
                cudaFuncAttributes attributes{};
                int activeBlocks = 0;
                constexpr int shared = 92160;
                if (cudaFuncGetAttributes(&attributes, noisyGemmPtx<4,2,4,4,64,3,8,true,false>) != cudaSuccess ||
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks,
                        noisyGemmPtx<4,2,4,4,64,3,8,true,false>, 256, shared) != cudaSuccess ||
                    attributes.maxThreadsPerBlock < 256 || attributes.maxDynamicSharedSizeBytes < shared ||
                    activeBlocks < 1) { publication_.poison(); release(); return false; }
                fprintf(stderr, "[pearl-experiment-resource] plan=0x%016llx registers=%d local_bytes=%zu "
                    "static_shared_bytes=%zu dynamic_shared_bytes=%d threads=256 active_blocks_per_sm=%d\n",
                    static_cast<unsigned long long>(kCudaWm4ReuseExperimentPlan.preparationPlanId),
                    attributes.numRegs, attributes.localSizeBytes, attributes.sharedSizeBytes, shared, activeBlocks);
            }
#endif
            const auto &plan = tileConfig(tile_);
            fprintf(stderr, "[pearl-experiment] selected kernel=%s tile=%s token=0x%04x "
                    "plan=0x%016llx shape=%ux%u sm=%d\n", plan.kernelId, plan.tileId,
                    static_cast<unsigned>(plan.token),
                    static_cast<unsigned long long>(plan.preparationPlanId),
                    shape_.m, shape_.n, arch10_);
            return completeNativePreparation();
        }

        if (formalRequest_.enabled) {
            identity_ = {};
            if (!validFormalBenchmarkRequest(formalRequest_) ||
                !formalBenchmarkCatalogIsUnique() ||
                getenv("CANARY_PEARL_SHAPE") || getenv("CANARY_PEARL_TILE") ||
                getenv("CANARY_PEARL_BENCH_ALL_HIT") ||
                getenv("CANARY_PEARL_HARD_TARGET")) {
                fprintf(stderr,
                        "[pearl-pow] invalid or environment-modified formal "
                        "benchmark request\n");
                return false;
            }

            const Shape *requestedShape = nullptr;
            size_t shapeMatches = 0;
            for (const Shape &candidate : kShapes) {
                if (formalRequest_.shapeId == candidate.id) {
                    requestedShape = &candidate;
                    ++shapeMatches;
                }
            }
            size_t tileMatches = 0;
            for (size_t i = 0;
                 i < sizeof(kTileConfigs) / sizeof(kTileConfigs[0]); ++i) {
                const TileConfig &candidate = kTileConfigs[i];
                if (formalRequest_.kernelId == candidate.kernelId &&
                    formalRequest_.tileId == candidate.tileId) {
                    formalTile_ = i;
                    ++tileMatches;
                }
            }
            if (shapeMatches != 1 || tileMatches != 1) {
                fprintf(stderr,
                        "[pearl-pow] formal shape and kernel/tile IDs must "
                        "resolve to exactly one table entry\n");
                return false;
            }
            const TileConfig &requestedTile = kTileConfigs[formalTile_];
            if (bytesFor(*requestedShape) + headroom > freeB ||
                !tileArchitectureEligible(requestedTile.architecture, arch10_) ||
                requestedTile.blockK <= 0 ||
                (kK / 32) % (requestedTile.blockK / 32) ||
                requestedShape->m % requestedTile.blockM ||
                requestedShape->n % requestedTile.blockN) {
                fprintf(stderr,
                        "[pearl-pow] requested formal shape/kernel/tile is "
                        "ineligible on this CUDA device\n");
                return false;
            }

            size_t selected = 0;
            if (tuneShape(*requestedShape, &selected) <= 0 ||
                selected != formalTile_) {
                fprintf(stderr,
                        "[pearl-pow] requested formal kernel failed the "
                        "cross-family correctness quorum\n");
                return false;
            }
            shape_ = *requestedShape;
            tile_ = selected;
            if (!allocate()) {
                release();
                return false;
            }
            allocated_ = true;
            if (!selfCheck()) {
                publication_.poison();
                release();
                return false;
            }
            identity_.valid = true;
            identity_.formal = true;
            identity_.schema = kPearlBenchmarkSchema;
            identity_.fixtureId = kPearlBenchmarkFixture;
            identity_.startingNonce = formalRequest_.startingNonce;
            identity_.shapeId = requestedShape->id;
            identity_.m = requestedShape->m;
            identity_.n = requestedShape->n;
            identity_.k = kK;
            identity_.rank = kRank;
            identity_.kernelId = requestedTile.kernelId;
            identity_.tileId = requestedTile.tileId;
            fprintf(stderr,
                    "[pearl-pow] formal benchmark selected %s / %s / %s\n",
                    requestedShape->id, requestedTile.kernelId,
                    requestedTile.tileId);
            return completeNativePreparation();
        }

        // Tune SHAPE and TILE CONFIGURATION together, because they interact.
        // Neither can be chosen first: the best shape moved once the kernel got
        // faster (a bigger working set stopped fitting L2), and the best
        // configuration differs per card - a 4090 wants 4x2 tiles per warp, a
        // 5080 wants 2x4, and each is near the other's worst. Choosing one and
        // then the other picks a local optimum on some card.
        //
        // The work is fixed and bounded: correctness admission, one warmup and
        // seven samples per eligible tile, then seven one-sample full-attempt
        // rounds per surviving shape, with at most one COMPLETE epoch retry
        // for excessive timing spread only. A shape can be pinned. Pinning lets the
        // robust choice be checked against the miner's own candidate rate:
        //   CANARY_PEARL_SHAPE=4096x65536 ./canaryminer --algo pearl-pow ...
        unsigned pinM = 0, pinN = 0;
        if (shapeExperiment_.enabled()) {
            pinM = shapeExperiment_.m(); pinN = shapeExperiment_.n();
        } else if (const char *pin = getenv("CANARY_PEARL_SHAPE")) {
            if (sscanf(pin, "%ux%u", &pinM, &pinN) == 2 && pinM && pinN) {
                fprintf(stderr, "[pearl-pow] shape pinned to %ux%u\n", pinM, pinN);
            } else {
                fprintf(stderr, "[pearl-pow] CANARY_PEARL_SHAPE wants MxN, e.g. "
                                "4096x65536 - ignoring \"%s\"\n", pin);
                pinM = pinN = 0;
            }
        }

        exactPinnedConfiguration_ = false;
        bool exactTileNamed = false;
        if (tileFilter_) {
            forEachArchitectureEligible(
                kTileConfigs, arch10_, [&](size_t, const TileConfig &tc) {
                    if (strcmp(tc.name, tileFilter_) == 0) exactTileNamed = true;
                });
        }
        // Reuse the demonstrated exact-TMA startup fix without Green code.
        // The appended TMA row is absent from the main kTileConfigs loop.
#if defined(CANARY_PEARL_ENABLE_TMA_EARLY_FUSION) && CANARY_PEARL_ENABLE_TMA_EARLY_FUSION == 1
        if(tma_fused::exactTmaPin(pearlProductFixedSelectionActive() ||
                shapeExperiment_.value()==
                PearlShapeExperiment::Exact16384x131072K8192V1,
                pinM,pinN,kK,kRank,uint32_t(arch10_),
                tmaDeliveryTileFilterMatches(tileFilter_)))
            exactTileNamed=true;
#endif
        // Prefix filters still request automatic tuning within that family.
#if defined(CANARY_PEARL_AMPERE_BUILD)
        // The Ampere build mines at K2048 on a desktop GPU, whose timing never meets the tuner's spread bound, so it
        // accepts the same exact operator pin the K8192 build does.
        constexpr bool kExactPinAllowed = true;
#else
        constexpr bool kExactPinAllowed = kK == 8192;
#endif
        if (kExactPinAllowed && getenv("CANARY_PEARL_SHAPE") != nullptr &&
            exactTileNamed) {
            uint32_t exactM = 0, exactN = 0;
            if (!pearl::tuner::parseExactShapePin(getenv("CANARY_PEARL_SHAPE"),
                                                 &exactM, &exactN) ||
                exactM != pinM || exactN != pinN) {
                fprintf(stderr, "[pearl-pow] invalid exact shape/tile pin - refusing configuration\n");
                return false;
            }
            exactPinnedConfiguration_ = true;
        }

        // The GEMM sweep is a good filter and a bad judge. It compares
        // configurations at one shape honestly, because they do identical
        // work, and it compares shapes against each other not at all. So it
        // picks the configuration here and the shape is settled below, by
        // running the real thing.
        struct Cand { Shape s; size_t cfg; };
        Cand cands[sizeof(kShapes) / sizeof(kShapes[0])];
        int nCand = 0;
        for (const Shape &cand : kShapes) {
            if (pinM && (cand.m != pinM || cand.n != pinN)) continue;
            if (bytesFor(cand) + headroom > freeB) continue;
            size_t cfg = 0;
            if (tuneShape(cand, &cfg) > 0) cands[nCand++] = Cand{cand, cfg};
            if (publication_.poisoned() || preparationCancellation_.requested()) return false;
        }
        if (!nCand) {
            fprintf(stderr,
                    "[pearl-pow] no fitting shape passed the stable, "
                    "cross-family tile tuner in %.0f MB of free device memory\n",
                    freeB / 1e6);
            return false;
        }

        const bool exactPinnedConfiguration = exactPinnedConfiguration_;
        if (exactPinnedConfiguration && nCand != 1) return false;

        const pearl::tuner::SevenRoundOrder attemptRounds =
            pearl::tuner::makeSevenRoundOrder(
                static_cast<size_t>(nCand), tuningKey_,
                UINT64_C(0x66756c6c6174746d));
        pearl::tuner::PairedRateSelection selection;
        if (exactPinnedConfiguration) {
            selection.valid = true;
            selection.winner = 0;
        }
        for (size_t epoch = 0;
             !exactPinnedConfiguration &&
             epoch < pearl::tuner::kMaximumFullAttemptTuningEpochs; ++epoch) {
            // Reset the WHOLE matrix. No row/sample from an earlier rejected
            // epoch contributes to this epoch. Raw observations remain logged.
            std::vector<std::vector<double>> attemptRates(nCand);
            fprintf(stderr, "[pearl-pow] full-attempt epoch=%zu/%zu begin shapes=%d\n",
                    epoch + 1, pearl::tuner::kMaximumFullAttemptTuningEpochs, nCand);
            for (size_t round = 0; round < attemptRounds.size(); ++round) {
                for (size_t idx : attemptRounds[round]) {
                    double rate = 0.0;
                    if (!measureAttemptSample(cands[idx].s, cands[idx].cfg,
                                              epoch, round, &rate)) {
                        // CUDA/invalid/partial samples never receive a retry.
                        // Continuing would shift every later sample's global
                        // position and invalidate forward/reverse pair balance.
                        fprintf(stderr,
                                "[pearl-pow] full-attempt timing schedule aborted "
                                "after a sample failure - refusing to mine\n");
                        return false;
                    }
                    attemptRates[idx].push_back(rate);
                }
            }

            selection = pearl::tuner::selectFullAttemptRateWinner(attemptRates);
            for (int i = 0; i < nCand; i++) {
                const pearl::tuner::PairedRateSummary &summary =
                    selection.summaries[i];
                const auto &disposition = selection.dispositions[i];
                fprintf(stderr,
                        "[pearl-pow] full-attempt disposition epoch=%zu m=%u n=%u "
                        "winner=%d stable=%d paired_median=%.17g spread=%.17g "
                        "observed_low=%.17g observed_high=%.17g reason=%s "
                        "confirmation_passed=%d\n",
                        epoch + 1, cands[i].s.m, cands[i].s.n,
                        i == static_cast<int>(selection.winner), disposition.stable,
                        summary.medianPairScore, summary.relativePairSpread,
                        disposition.observedLow, disposition.observedHigh,
                        disposition.reason, disposition.confirmationPassed);
                if (!summary.valid) {
                    fprintf(stderr,
                            "[pearl-pow] %ux%u paired full-attempt timing "
                            "rejected: %s\n",
                            cands[i].s.m, cands[i].s.n,
                            pearl::tuner::rejectionReasonName(summary.rejection));
                }
            }
            fprintf(stderr, "[pearl-pow] full-attempt epoch=%zu result=%s reason=%s\n",
                    epoch + 1, selection.valid ? "accepted" : "rejected",
                    pearl::tuner::rejectionReasonName(selection.rejection));
            // The first stable epoch wins. Never sample/rank a later epoch
            // once this one is accepted, even if it might produce a higher rate.
            if (selection.valid) break;
            if (publication_.poisoned() ||
                !pearl::tuner::retryFullAttemptEpoch(
                    epoch, selection.valid, selection.rejection)) break;
            fprintf(stderr, "[pearl-pow] repeating complete full-attempt timing epoch; "
                    "prior rejected samples retained in log, not reused\n");
        }
        if (!selection.valid) {
            fprintf(stderr,
                    "[pearl-pow] paired full-attempt shape selection "
                    "rejected: %s - refusing to mine\n",
                    pearl::tuner::rejectionReasonName(selection.rejection));
            return false;
        }
        const int bestIdx = static_cast<int>(selection.winner);
        const double bestRate =
            exactPinnedConfiguration ? 0.0 :
                selection.summaries[selection.winner].medianPairScore / 1e6;
        shape_ = cands[bestIdx].s;
        tile_ = cands[bestIdx].cfg;

        if (!allocate()) {
            fprintf(stderr, "[pearl-pow] could not allocate %.0f MB\n",
                    bytesFor(shape_) / 1e6);
            release();
            return false;
        }
        allocated_ = true;
        fprintf(stderr, "[pearl-matrix-input] profile=%s\n", matrix_input::profile);
        // Say what this number IS, because the one that used to print here
        // was the GEMM alone, read 15-20% high, and got quoted as hashrate in
        // a public README. This one times a whole attempt but still runs them
        // back to back with no job handling, so it lands about 9% above what
        // the miner then reports. Quote the miner's own counter, never this.
        if (exactPinnedConfiguration) {
            fprintf(stderr,
                    "[pearl-pow] explicit configuration %ux%u with %s: "
                    "cross-family transcript quorum passed; timing not ranked, "
                    "self-check pending, %.0f MB\n",
                    shape_.m, shape_.n, tileConfig(tile_).name,
                    bytesFor(shape_) / 1e6);
        } else fprintf(stderr,
                "[pearl-pow] %ux%u with %s: %.1f paired-median M candidates/s "
                "(3 drift-balanced pairs + confirmation, device, no job "
                "overhead - not hashrate), %.0f MB\n",
                shape_.m, shape_.n, tileConfig(tile_).name, bestRate,
                bytesFor(shape_) / 1e6);

        if (!selfCheck()) {
            fprintf(stderr,
                    "[pearl-pow] REFUSING TO MINE: this GPU or driver does not "
                    "compute the kernel correctly. Every share would be "
                    "rejected.\n");
            publication_.poison();
            release();
            return false;
        }
        return completeNativePreparation();
    }

    /**
     * Prove the SELECTED kernel is right on THIS card, before mining a thing.
     *
     * The tuner's fingerprint check proves the configurations agree with each
     * other, which is not the same as being correct - if they were all wrong
     * the same way it would pass. This is the stronger form: a small GEMM on
     * the device against the host's own tileTranscript(), which is the code
     * `recheck()` uses and which Pearl's own Rust verifier has accepted.
     *
     * It exists because offline testing can only ever cover the cards we own.
     * The reference vectors gate every shipped configuration on a 4090 and a
     * 5080 and say nothing whatsoever about a 3070, a 5090 or next quarter's
     * driver. A compiler that miscompiles this kernel is not hypothetical -
     * NVIDIA's Vulkan compiler was found doing exactly that to a byte repack
     * in this repo's other algorithm, deterministically and silently, with
     * every share rejected by the pool. This turns that class of thing from
     * "found by a user mining into a live pool" into a refusal at startup.
     *
     * 256x256 so it fits every block shape in the table. Four tiles distributed
     * across the transcript buffer are checked, including both ends; tile zero
     * alone cannot catch a block-coordinate or transcript-layout error whose
     * first block happens to be correct.
     */
    bool selfCheck() {
        constexpr uint32_t m = 256, n = 256;
        const size_t aBytes = (size_t)m * kK, bBytes = (size_t)n * kK;
        const uint32_t tiles = (m / kTileSide) * (n / kTileSide);

        int8_t *dA = nullptr, *dBk = nullptr, *dBn = nullptr;
        uint32_t *dT = nullptr;
        auto cleanup = [&]() {
            synchronizeStreamStatus();
            freeTemporary(dA);
            freeTemporary(dBk);
            freeTemporary(dBn);
            freeTemporary(dT);
        };
        auto fail = [&](const char *stage, cudaError_t err) {
            fprintf(stderr, "[pearl-pow] self-check could not complete %s: %s\n",
                    stage, cudaGetErrorString(err));
            publication_.poison();
            cleanup();
            return false;
        };
        cudaError_t err = cudaMalloc(&dA, aBytes);
        if (err != cudaSuccess) return fail("A allocation", err);
        err = cudaMalloc(&dBk, bBytes);
        if (err != cudaSuccess) return fail("B allocation", err);
        err = cudaMalloc(&dBn, bBytes);
        if (err != cudaSuccess) return fail("B-layout allocation", err);
        err = cudaMalloc(&dT, (size_t)tiles * 16 * 4);
        if (err != cudaSuccess) return fail("transcript allocation", err);

        genMatrix<<<(aBytes / 8 + 255) / 256, 256, 0, stream_>>>(dA, aBytes, 0x5e1f01);
        err = cudaGetLastError();
        if (err != cudaSuccess) return fail("A fixture generation launch", err);
        genMatrix<<<(bBytes / 8 + 255) / 256, 256, 0, stream_>>>(dBk, bBytes, 0x5e1f02);
        err = cudaGetLastError();
        if (err != cudaSuccess) return fail("B fixture generation launch", err);
        err = cudaMemsetAsync(dT, 0, (size_t)tiles * 16 * 4, stream_);
        if (err != cudaSuccess) return fail("transcript initialization", err);
        err = synchronizeStreamStatus();
        if (err != cudaSuccess) return fail("fixture generation", err);

        std::vector<int8_t> A(aBytes), Bk(bBytes), Bn(bBytes);
        err = cudaMemcpy(A.data(), dA, aBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return fail("A fixture readback", err);
        err = cudaMemcpy(Bk.data(), dBk, bBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return fail("B fixture readback", err);
        for (uint32_t j = 0; j < n; j++)
            for (uint32_t q = 0; q < kK; q++)
                Bn[(size_t)j * kK + q] = Bk[(size_t)q * n + j];
        err = cudaMemcpy(dBn, Bn.data(), bBytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return fail("B-layout upload", err);

        const TileConfig &tc = tileConfig(tile_);
        bool ok = true;
        if (m % tc.blockM || n % tc.blockN) {
            fprintf(stderr,
                    "[pearl-pow] self-check cannot exercise selected geometry "
                    "%s at %ux%u\n", tc.name, m, n);
            ok = false;
        } else {
            dim3 grid(n / tc.blockN, m / tc.blockM);
            err = tc.launch(grid, tc.threads, stream_, dA,
                            tc.nMajorB ? dBn : dBk, dT,
                            (int)m, (int)n, (int)kK, kRank);
            if (err != cudaSuccess) {
                fprintf(stderr, "[pearl-pow] self-check launch failed for %s: %s\n",
                        tc.name, cudaGetErrorString(err));
                ok = false;
            } else {
                err = synchronizeStreamStatus();
                if (err != cudaSuccess) {
                    fprintf(stderr,
                            "[pearl-pow] self-check execution failed for %s: %s\n",
                            tc.name, cudaGetErrorString(err));
                    cleanup();
                    return false;
                }
                std::vector<uint32_t> got((size_t)tiles * 16);
                err = cudaMemcpy(got.data(), dT, (size_t)tiles * 16 * 4,
                                 cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) return fail("transcript readback", err);

                const uint32_t samples[] = {0, tiles / 3, tiles / 2, tiles - 1};
                const uint32_t tilesPerSide = kRank / kTileSide;
                const uint32_t blocksPerRow = n / kRank;
                for (uint32_t flat : samples) {
                    const uint32_t wi = flat % tilesPerSide;
                    const uint32_t hi = (flat / tilesPerSide) % tilesPerSide;
                    const uint32_t block = flat / (tilesPerSide * tilesPerSide);
                    const uint32_t jIdx = block % blocksPerRow;
                    const uint32_t iIdx = block / blocksPerRow;
                    const uint32_t tRow = iIdx * kRank + hi * kTileSide;
                    const uint32_t tCol = jIdx * kRank + wi * kTileSide;

                    // bStrip is k-major over 16 columns, matching recheck().
                    std::vector<int8_t> aStrip((size_t)kTileSide * kK);
                    std::vector<int8_t> bStrip((size_t)kK * kTileSide);
                    for (int i = 0; i < kTileSide; i++)
                        memcpy(&aStrip[(size_t)i * kK],
                               &A[(size_t)(tRow + i) * kK], kK);
                    for (size_t q = 0; q < kK; q++)
                        for (int j = 0; j < kTileSide; j++)
                            bStrip[q * kTileSide + j] = Bk[q * n + tCol + j];

                    uint32_t want[kTranscriptWords];
                    tileTranscript(aStrip.data(), bStrip.data(), kK, kTileSide,
                                   kRank, 0, 0, kTileSide, kTileSide, want);
                    for (uint32_t w = 0; w < kTranscriptWords; w++) {
                        if (got[(size_t)flat * kTranscriptWords + w] != want[w]) {
                            ok = false;
                            fprintf(stderr,
                                    "[pearl-pow] self-check FAILED for %s: "
                                    "device/host disagree on tile %u (%u,%u)\n",
                                    tc.name, flat, tRow, tCol);
                            break;
                        }
                    }
                    if (!ok) break;
                }
            }
        }

        cleanup();
        return ok && !publication_.poisoned();
    }

    /**
     * Time every eligible configuration at one shape; return the best rate in
     * millions of candidates per second and which configuration got it.
     *
     * Candidates per second, not milliseconds: shapes produce different numbers
     * of transcripts per GEMM, so raw time is not comparable across them.
     *
     * The buffers are filled before timing. A GEMM reading pages that were
     * never written is not measuring the same thing as one reading real data,
     * which already cost an hour once in this project's history.
     */
    double tuneShape(const Shape &s, size_t *bestCfg) {
        for (size_t attempt = 0;
             attempt < pearl::tuner::kMaximumTileTuningAttempts; ++attempt) {
            if (preparationCancellation_.requested()) return 0;
            bool retryTiming = false;
            size_t candidate = 0;
            double rate = 0;
            if (!preparationCancellation_.runStage([&] {
                    rate = tuneShapeOnce(s, &candidate, &retryTiming);
                    return true;
                })) return 0;
            if (rate > 0 && !publication_.poisoned()) {
                *bestCfg = candidate;
                return rate;
            }
            if (!retryTiming || publication_.poisoned()) return 0;
            if (attempt + 1 < pearl::tuner::kMaximumTileTuningAttempts)
                fprintf(stderr,
                        "[pearl-pow] timing-only refusal; repeating complete "
                        "shape %ux%u epoch %zu/%zu (no retained samples)\n",
                        s.m, s.n, attempt + 2,
                        pearl::tuner::kMaximumTileTuningAttempts);
        }
        return 0;
    }

    double tuneShapeOnce(const Shape &s, size_t *bestCfg, bool *retryTiming) {
        *retryTiming = false;
        const size_t aBytes = (size_t)s.m * kK;
        const size_t bBytes = (size_t)kK * s.n;
        const size_t tiles = (size_t)(s.m / kTileSide) * (s.n / kTileSide);

        int8_t *tA = nullptr, *tB = nullptr, *tBn = nullptr;
        uint32_t *tT = nullptr;
        TranscriptWitness *tWitness = nullptr;
        auto cleanupBuffers = [&]() {
            synchronizeStreamStatus();
            freeTemporary(tA);
            freeTemporary(tB);
            freeTemporary(tBn);
            freeTemporary(tT);
            freeTemporary(tWitness);
        };
        if (cudaMalloc(&tA, aBytes) != cudaSuccess) return 0;
        if (cudaMalloc(&tB, bBytes) != cudaSuccess) {
            cleanupBuffers();
            return 0;
        }
        if (cudaMalloc(&tT, tiles * 16 * sizeof(uint32_t)) != cudaSuccess) {
            cleanupBuffers();
            return 0;
        }
        // The raw-mma family reads B n-major. Feed every candidate the layout
        // it expects, or it computes a different product from the same bytes
        // and the fingerprint check below throws it out - which is exactly
        // what happened: every ptx configuration was rejected and the miner
        // silently settled for async at 253 M candidates/s instead of 406.
        if (cudaMalloc(&tBn, bBytes) != cudaSuccess) {
            cleanupBuffers();
            return 0;
        }
        // Allocated here rather than in allocate(), which runs AFTER tuning.
        if (cudaMalloc(&tWitness, sizeof(TranscriptWitness)) != cudaSuccess) {
            cleanupBuffers();
            return 0;
        }
        genMatrix<<<(aBytes / 8 + 255) / 256, 256, 0, stream_>>>(tA, aBytes, 1);
        if (cudaGetLastError() != cudaSuccess) {
            cleanupBuffers();
            return 0;
        }
        genMatrix<<<(bBytes / 8 + 255) / 256, 256, 0, stream_>>>(tB, bBytes, 2);
        if (cudaGetLastError() != cudaSuccess) {
            cleanupBuffers();
            return 0;
        }
        transposeKtoN<<<(bBytes + 255) / 256, 256, 0, stream_>>>(tB, tBn, s.n, kK);
        if (cudaGetLastError() != cudaSuccess ||
            synchronizeStreamStatus() != cudaSuccess) {
            cleanupBuffers();
            return 0;
        }

        double best = 0;
        *bestCfg = 0;
        struct Observation {
            size_t cfg;
            TranscriptWitness witness;
            uint8_t family;
            bool selectable;
            std::vector<double> rates;
            pearl::tuner::PairedRateSummary summary;
        };
        std::vector<Observation> observations;
        // Rejecting one configuration is routine. Rejecting an entire FAMILY
        // means the harness is feeding it something it does not read, not that
        // the kernel is broken - that exact mistake cost 38% of hashrate and
        // was invisible because every test launches kernels directly. Counted
        // per family so it can be said out loud.
        int seen[4] = {0, 0, 0, 0}, rejected[4] = {0, 0, 0, 0};
        struct Candidate {
            size_t cfg;
            uint8_t family;
            bool selectable;
        };
        std::vector<Candidate> candidates;
        if (!formalRequest_.enabled && arch10_==120 && kK==8192 && kRank==128 &&
            (pearlProductFixedSelectionActive() ||
             shapeExperiment_.value()==PearlShapeExperiment::Exact16384x131072K8192V1) &&
            tmaDeliveryTileFilterMatches(tileFilter_) &&
            s.m==16384 && s.n==131072)
            candidates.push_back(Candidate{kTmaDeliveryTile,static_cast<uint8_t>(kPtx),true});
        forEachArchitectureEligible(
            kTileConfigs, arch10_, [&](size_t i, const TileConfig &tc) {
                if (tc.blockK <= 0 || (kK / 32) % (tc.blockK / 32)) return;
                // Escape hatch for A/B work: CANARY_PEARL_TILE=ptx restricts
                // which configuration may WIN to names starting with that
                // string. The other families still run because correctness
                // now requires an independent-family witness quorum.
                const bool selectable = formalRequest_.enabled
                    ? i == formalTile_
                    : cudaFixedExperimentSelectable(experimentSelection_, false) &&
                          (!tileFilter_ || (exactPinnedConfiguration_
                              ? strcmp(tc.name, tileFilter_) == 0
                              : !strncmp(tc.name, tileFilter_, strlen(tileFilter_))));
                if (s.m % tc.blockM || s.n % tc.blockN) return;
                candidates.push_back(Candidate{
                    i, static_cast<uint8_t>(tc.family), selectable});
            });
#if defined(CANARY_PEARL_ENABLE_FIXED_SCHEDULE_EXPERIMENT) && CANARY_PEARL_ENABLE_FIXED_SCHEDULE_EXPERIMENT == 1
        if (experimentSelection_ == CudaFixedExperimentSelection::Enabled &&
            cudaFixedExperimentVisits(experimentSelection_, true, formalRequest_.enabled) &&
            cudaFixedExperimentLaunchValid(arch10_, s.m, s.n, kK, kRank))
            candidates.push_back(Candidate{kFixedExperimentTile, static_cast<uint8_t>(kPtx), true});
#endif
#if defined(CANARY_PEARL_ENABLE_WM8_EXPERIMENT) && CANARY_PEARL_ENABLE_WM8_EXPERIMENT == 1
        if (experimentSelection_ == CudaFixedExperimentSelection::Wm8Runtime &&
            cudaFixedExperimentVisits(experimentSelection_, true, formalRequest_.enabled) &&
            cudaFixedExperimentLaunchValid(arch10_, s.m, s.n, kK, kRank))
            candidates.push_back(Candidate{kWm8ExperimentTile, static_cast<uint8_t>(kPtx), true});
#endif

#if defined(CANARY_PEARL_ENABLE_WM8_CAP112_EXPERIMENT)
        if (experimentSelection_ == CudaFixedExperimentSelection::Wm8InlineCap112 &&
            cudaFixedExperimentVisits(experimentSelection_, true, formalRequest_.enabled) &&
            cudaFixedExperimentLaunchValid(arch10_, s.m, s.n, kK, kRank))
            candidates.push_back(Candidate{kWm8Cap112ExperimentTile, static_cast<uint8_t>(kPtx), true});
#endif

#if defined(CANARY_PEARL_ENABLE_WM8_FIXED_EXPERIMENT)
        if (experimentSelection_ == CudaFixedExperimentSelection::Wm8FixedK2048R128 &&
            cudaFixedExperimentVisits(experimentSelection_, true, formalRequest_.enabled) &&
            cudaFixedExperimentLaunchValid(arch10_, s.m, s.n, kK, kRank))
            candidates.push_back(Candidate{kWm8FixedExperimentTile, static_cast<uint8_t>(kPtx), true});
#endif
#if defined(CANARY_PEARL_ENABLE_BK32_FIXED_EXPERIMENT)
        if (experimentSelection_ == CudaFixedExperimentSelection::Bk32FixedK2048R128 &&
            cudaFixedExperimentVisits(experimentSelection_, true, formalRequest_.enabled) &&
            cudaSelectedExperimentLaunchValid(experimentSelection_, arch10_, s.m, s.n, kK, kRank))
            candidates.push_back(Candidate{kBk32FixedExperimentTile, static_cast<uint8_t>(kPtx), true});
#endif

#if defined(CANARY_PEARL_ENABLE_WM4_REUSE_EXPERIMENT)
        if (experimentSelection_ == CudaFixedExperimentSelection::Wm4ReuseFixedK2048R128 &&
            cudaFixedExperimentVisits(experimentSelection_, true, formalRequest_.enabled) &&
            cudaSelectedExperimentLaunchValid(experimentSelection_, arch10_, s.m, s.n, kK, kRank))
            candidates.push_back(Candidate{kWm4ReuseExperimentTile, static_cast<uint8_t>(kPtx), true});
#endif

#if defined(CANARY_PEARL_ENABLE_TMA_WM8_EXPERIMENT)
        if (experimentSelection_ == CudaFixedExperimentSelection::TmaWm8FixedK2048R128 &&
            cudaFixedExperimentVisits(experimentSelection_, true, formalRequest_.enabled) &&
            cudaSelectedExperimentLaunchValid(experimentSelection_, arch10_, s.m, s.n, kK, kRank))
            candidates.push_back(Candidate{kTmaWm8ExperimentTile, static_cast<uint8_t>(kPtx), true});
#endif
#if defined(CANARY_PEARL_ENABLE_TMA_TRAVERSAL_EXPERIMENT)
        if (experimentSelection_ == CudaFixedExperimentSelection::TmaTraversal16x32 &&
            cudaFixedExperimentVisits(experimentSelection_, true, formalRequest_.enabled) &&
            cudaSelectedExperimentLaunchValid(experimentSelection_, arch10_, s.m, s.n, kK, kRank))
            candidates.push_back(Candidate{kTmaTraversalExperimentTile, static_cast<uint8_t>(kPtx), true});
#endif

#if CANARY_PEARL_ENABLE_TMA_STAGE4_EXPERIMENT == 1
        if (experimentSelection_ == CudaFixedExperimentSelection::TmaTraversal16x32Stage4 &&
            cudaFixedExperimentVisits(experimentSelection_, true, formalRequest_.enabled) &&
            cudaSelectedExperimentLaunchValid(experimentSelection_, arch10_, s.m, s.n, kK, kRank))
            candidates.push_back(Candidate{kTmaStage4ExperimentTile, static_cast<uint8_t>(kPtx), true});
#endif

        auto captureWitness = [&](const Candidate &candidate,
                                  const char *stage,
                                  TranscriptWitness *witness) {
            if (publication_.poisoned()) return false;
            const TileConfig &tc = tileConfig(candidate.cfg);
            const dim3 grid(s.n / tc.blockN, s.m / tc.blockM);
            cudaError_t err = cudaMemsetAsync(
                tT, 0, tiles * 16 * sizeof(uint32_t), stream_);
            if (err == cudaSuccess) {
                err = tc.launch(grid, tc.threads, stream_, tA,
                          tc.nMajorB ? tBn : tB, tT, (int)s.m, (int)s.n,
                          (int)kK, kRank);
            }
            if (err == cudaSuccess) err = synchronizeStreamStatus();
            if (err == cudaSuccess)
                err = cudaMemsetAsync(tWitness, 0, sizeof(TranscriptWitness),
                                      stream_);
            if (err == cudaSuccess) {
                transcriptFingerprint<<<256, 256, 0, stream_>>>(
                    tT, (uint64_t)tiles * 16, tWitness);
                err = cudaGetLastError();
            }
            if (err == cudaSuccess) err = synchronizeStreamStatus();
            if (err == cudaSuccess) {
                err = cudaMemcpy(witness, tWitness, sizeof(TranscriptWitness),
                                 cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) publication_.poison();
            }
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[pearl-pow] %s %s failed for %s: %s\n",
                        stage, "witness", tc.name, cudaGetErrorString(err));
                return false;
            }
            return true;
        };

        // Correctness admission happens before any timing. A configuration
        // that cannot produce the accepted cross-family witness never gets a
        // rate, regardless of how quickly it returns.
        for (const Candidate &candidate : candidates) {
            if (preparationCancellation_.requested()) { cleanupBuffers(); return 0; }
            TranscriptWitness witness = {};
            if (!captureWitness(candidate, "initial", &witness)) {
                if (publication_.poisoned()) {
                    cleanupBuffers();
                    return 0;
                }
                continue;
            }
            seen[candidate.family]++;
            if (witness.sum[0] == 0 && witness.sum[1] == 0) {
                rejected[candidate.family]++;
                continue;   // wrote nothing
            }
            observations.push_back(Observation{candidate.cfg, witness,
                                               candidate.family,
                                               candidate.selectable, {}, {}});
        }
        if (preparationCancellation_.requested()) { cleanupBuffers(); return 0; }

        std::vector<TranscriptWitness> witnesses;
        std::vector<uint8_t> families;
        witnesses.reserve(observations.size());
        families.reserve(observations.size());
        for (const Observation &o : observations) {
            witnesses.push_back(o.witness);
            families.push_back(o.family);
        }
        TranscriptWitness accepted = {};
        const int support = transcriptWitnessQuorum(
            witnesses.data(), families.data(), witnesses.size(), &accepted);
        if (!support) {
            fprintf(stderr,
                    "[pearl-pow] no unambiguous transcript witness has quorum "
                    "across two kernel families - refusing this shape\n");
        } else {
            std::vector<size_t> quorate;
            for (size_t i = 0; i < observations.size(); ++i) {
                Observation &o = observations[i];
                if (!transcriptWitnessEqual(o.witness, accepted)) {
                    rejected[o.family]++;
                    fprintf(stderr,
                            "[pearl-pow] %s disagrees with the %d-family "
                            "transcript quorum on this card - not selected\n",
                            tileConfig(o.cfg).name, support);
                    continue;
                }
                quorate.push_back(i);
            }

            // Exact formal/experimental selection does not use timing as an
            // eligibility condition. The requested row must belong to the
            // unchanged cross-family correctness quorum; never fall back.
            if (formalRequest_.enabled || cudaIsolatedExperimentEnabled(experimentSelection_)) {
                const size_t exactTile = formalRequest_.enabled ? formalTile_ : experimentTile(experimentSelection_);
                for (size_t observationIndex : quorate) {
                    const Observation &observation =
                        observations[observationIndex];
                    if (observation.cfg == exactTile) {
                        *bestCfg = observation.cfg;
                        cleanupBuffers();
                        return publication_.poisoned() ? 0 : 1.0;
                    }
                }
                cleanupBuffers();
                return 0;
            }

            // An exact shape AND tile pin is an operator choice, not a
            // comparative timing result. Keep the complete cross-family
            // transcript quorum above; malformed or ambiguous pins cannot
            // fall back to another configuration. configure() still performs
            // allocation and selfCheck() before accepting work.
            if (exactPinnedConfiguration_) {
                std::vector<bool> mask;
                for (size_t i : quorate) mask.push_back(observations[i].selectable);
                std::vector<size_t> indices;
                if (!pearl::tuner::timingRowIndices(mask, true, &indices)) {
                    cleanupBuffers();
                    return 0;
                }
                *bestCfg = observations[quorate[indices[0]]].cfg;
                cleanupBuffers();
                // Positive eligibility sentinel, never reported as a rate.
                return publication_.poisoned() ? 0 : 1.0;
            }

            cudaEvent_t evA = nullptr, evB = nullptr;
            bool eventsOk = true;
            auto eventCall = [&](cudaError_t err, const char *stage) {
                if (err == cudaSuccess) return true;
                publication_.poison();
                fprintf(stderr, "[pearl-pow] CUDA event %s failed: %s\n",
                        stage, cudaGetErrorString(err));
                eventsOk = false;
                return false;
            };
            if (!eventCall(cudaEventCreate(&evA), "creation (start)")) {
                cleanupBuffers();
                return 0;
            }
            if (!eventCall(cudaEventCreate(&evB), "creation (stop)")) {
                eventCall(cudaEventDestroy(evA), "destruction (start)");
                cleanupBuffers();
                return 0;
            }

            // Keep quorate intact for independent post-timing witnesses.
            std::vector<size_t> timed = quorate;
#if defined(CANARY_PEARL_EXPERIMENT_PINNED_TIMING_ONLY)
            static_assert(kK == 8192, "pinned timing experiment requires K8192");
            if (getenv("CANARY_PEARL_TILE") != nullptr) {
                std::vector<bool> mask;
                for (size_t i : quorate) mask.push_back(observations[i].selectable);
                std::vector<size_t> indices;
                if (!pearl::tuner::timingRowIndices(mask, true, &indices)) {
                    eventCall(cudaEventDestroy(evA), "destruction (start)");
                    eventCall(cudaEventDestroy(evB), "destruction (stop)");
                    cleanupBuffers();
                    return 0;
                }
                timed.clear();
                for (size_t i : indices) timed.push_back(quorate[i]);
            }
#endif
            const pearl::tuner::SevenRoundOrder rounds =
                pearl::tuner::makeSevenRoundOrder(
                    timed.size(), tuningKey_, shapeKey(s));
            // One complete untimed sweep pays launch/JIT warmup without
            // privileging a source-order family.
            for (size_t orderIndex : rounds[0]) {
                if (preparationCancellation_.requested()) { eventsOk = false; break; }
                const Observation &o = observations[timed[orderIndex]];
                const TileConfig &tc = tileConfig(o.cfg);
                const dim3 grid(s.n / tc.blockN, s.m / tc.blockM);
                cudaError_t err = cudaMemsetAsync(
                    tT, 0, tiles * 16 * sizeof(uint32_t), stream_);
                if (err == cudaSuccess) {
                    err = tc.launch(grid, tc.threads, stream_, tA,
                              tc.nMajorB ? tBn : tB, tT, (int)s.m, (int)s.n,
                              (int)kK, kRank);
                }
                if (err == cudaSuccess) err = synchronizeStreamStatus();
                if (err != cudaSuccess) {
                    fprintf(stderr, "[pearl-pow] tile warmup failed for %s: %s\n",
                            tc.name, cudaGetErrorString(err));
                    eventsOk = false;
                    break;
                }
            }

            for (size_t round = 0; round < pearl::tuner::kSampleCount && eventsOk;
                 ++round) {
                if (preparationCancellation_.requested()) { eventsOk = false; break; }
                for (size_t orderIndex : rounds[round]) {
                    if (preparationCancellation_.requested()) { eventsOk = false; break; }
                    Observation &o = observations[timed[orderIndex]];
                    const TileConfig &tc = tileConfig(o.cfg);
                    const dim3 grid(s.n / tc.blockN, s.m / tc.blockM);
                    if (!eventCall(cudaEventRecord(evA, stream_),
                                   "recording (start)"))
                        break;
                    const cudaError_t launchErr = tc.launch(grid, tc.threads, stream_, tA,
                              tc.nMajorB ? tBn : tB, tT, (int)s.m, (int)s.n,
                              (int)kK, kRank);
                    if (launchErr != cudaSuccess) {
                        fprintf(stderr,
                                "[pearl-pow] timed tile launch failed for %s: %s\n",
                                tc.name, cudaGetErrorString(launchErr));
                        eventsOk = false;
                        break;
                    }
                    if (!eventCall(cudaEventRecord(evB, stream_),
                                   "recording (stop)") ||
                        !eventCall(cudaEventSynchronize(evB),
                                   "synchronization"))
                        break;
                    float elapsedMs = 0.0f;
                    if (!eventCall(cudaEventElapsedTime(&elapsedMs, evA, evB),
                                   "elapsed-time query"))
                        break;
                    const pearl::tuner::RateSample rate =
                        pearl::tuner::rateFromDuration(tiles, elapsedMs);
                    if (!rate.valid) {
                        fprintf(stderr,
                                "[pearl-pow] tile timing rejected for %s: %s\n",
                                tc.name,
                                pearl::tuner::rejectionReasonName(rate.rejection));
                        // Continuing would shift every later candidate's
                        // global position and invalidate pair balancing.
                        eventsOk = false;
                        break;
                    }
                    o.rates.push_back(rate.candidatesPerSecond);
                }
            }
            if (preparationCancellation_.requested()) eventsOk = false;

            eventCall(cudaEventDestroy(evA), "destruction (start)");
            eventCall(cudaEventDestroy(evB), "destruction (stop)");
            if (eventsOk) {
                std::vector<std::vector<double>> quorateRates;
                std::vector<bool> selectable;
                size_t bestObservation = observations.size();
                for (size_t observationIndex : timed) {
                    Observation &o = observations[observationIndex];
                    o.summary = pearl::tuner::summarizePairedRates(o.rates);
                    if (!o.summary.valid) {
                        fprintf(stderr,
                                "[pearl-pow] %s paired timing rejected: %s\n",
                                tileConfig(o.cfg).name,
                                pearl::tuner::rejectionReasonName(
                                    o.summary.rejection));
                    }
                    quorateRates.push_back(o.rates);
                    selectable.push_back(o.selectable);
                }

                const pearl::tuner::PairedRateSelection selection =
                    (kK == 8192 && getenv("CANARY_PEARL_TILE") != nullptr)
                        ? pearl::tuner::selectExplicitPinnedRate(quorateRates, selectable)
                        : pearl::tuner::selectPairedRateWinner(quorateRates, selectable);
                if (selection.valid) {
                    bestObservation = timed[selection.winner];
                    best = selection.summaries[selection.winner]
                               .medianPairScore;
                    *bestCfg = observations[bestObservation].cfg;
                } else {
                    *retryTiming = pearl::tuner::retryableTimingRejection(
                        selection.rejection);
                    fprintf(stderr,
                            "[pearl-pow] paired tile selection rejected: %s\n",
                            pearl::tuner::rejectionReasonName(
                                selection.rejection));
                    // Preserve the actual observations on failure. A short
                    // fixture can be timing-unstable without having produced
                    // an incorrect transcript; a reason alone cannot tell us
                    // which row or timing interval caused the refusal.
                    for (size_t i = 0; i < timed.size(); ++i) {
                        const Observation &o = observations[timed[i]];
                        fprintf(stderr,
                                "[pearl-pow] tile timing shape=%ux%u tile=%s "
                                "selectable=%d rates_cps=",
                                s.m, s.n, tileConfig(o.cfg).name,
                                o.selectable ? 1 : 0);
                        for (double rate : o.rates)
                            fprintf(stderr, " %.9g", rate);
                        fprintf(stderr, "\n");
                    }
                }

                // Bind the proposed rate to fresh post-timing correctness
                // observations from the winner and another supporting family.
                if (bestObservation != observations.size()) {
                    const Observation &winner = observations[bestObservation];
                    const Candidate winnerCandidate{winner.cfg, winner.family,
                                                    winner.selectable};
                    TranscriptWitness post = {};
                    bool confirmed =
                        captureWitness(winnerCandidate, "post-timing", &post) &&
                        transcriptWitnessEqual(post, accepted);
                    bool crossFamilyConfirmed = false;
                    for (size_t observationIndex : quorate) {
                        const Observation &other = observations[observationIndex];
                        if (other.family == winner.family) continue;
                        const Candidate otherCandidate{other.cfg, other.family,
                                                       other.selectable};
                        post = {};
                        crossFamilyConfirmed =
                            captureWitness(otherCandidate, "post-timing", &post) &&
                            transcriptWitnessEqual(post, accepted);
                        break;
                    }
                    if (!confirmed || !crossFamilyConfirmed) {
                        fprintf(stderr,
                                "[pearl-pow] post-timing cross-family witness "
                                "confirmation failed - refusing this shape\n");
                        best = 0;
                    }
                }
            }
        }
        if (preparationCancellation_.requested()) { cleanupBuffers(); return 0; }
        // A whole family failing is a harness bug until proven otherwise.
        static const char *kFamily[4] = {"single", "dbuf", "async", "ptx"};
        for (int f = 0; f < 4; f++)
            if (seen[f] > 0 && rejected[f] == seen[f])
                fprintf(stderr,
                        "[pearl-pow] WARNING: every '%s' kernel (%d of %d) was "
                        "rejected. One bad kernel is plausible; a whole family "
                        "usually means the tuner fed it the wrong B layout. "
                        "Hashrate may be far below what this card can do.\n",
                        kFamily[f], rejected[f], seen[f]);

        cleanupBuffers();
        return publication_.poisoned() ? 0 : best / 1e6;
    }

    bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                std::vector<Solution> *out) override {
        // A search request is an ownership boundary even when it fails before
        // launching CUDA.  Otherwise a malformed replacement job, an invalid
        // batch, or a setup failure could leave the previous proof authorized.
        wins_.beginSearch();
        privateStats_.lastValid=false;
        privateSearchCacheHit_=false;
        if ((!pearlPrivateEnvironment(privateExecution_) || !pearlPublicEnvironment(publicExecution_))) return failSearch();
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
        if (!lookahead_.ready && !integratedAttempt_.pending()) aRootEvidence_.invalidate();
#endif
        if (attemptOverlapEnabled(overlapMode_) &&
            (parseAttemptOverlap(getenv("CANARY_PEARL_ATTEMPT_OVERLAP")) != overlapMode_ ||
             getenv("CANARY_PEARL_PROFILE_INTERVAL"))) return failSearch();
        if (!out || !allocated_ || !fusedScanEnvironmentValid() || !integratedEnvironmentValid() || !shapeEnvironmentValid() || !fixedWm8EnvironmentValid() || !rootFusionEnvironmentValid()) return failSearch();
        if (!job.valid) {
            fprintf(stderr, "[pearl-pow] refusing an invalid job\n");
            return failSearch();
        }
        // Selection exists only after prepare succeeds. Enforce the private
        // single-attempt request before beginJob can enqueue replacement work.
        if (executionAccountingEnabled() &&
            (!(pearlPublicEnabled(publicExecution_)?pearlPublicSelectionValid(currentNativeSelection(),publicExecution_):pearlPrivateSelectionValid(currentNativeSelection(),executionAccountingMode())) || count!=currentNativeSelection().workUnitsPerAttempt)) return failSearch();

        if (integratedAttempt_.pending() && !integratedJobMatches(job,nonceBase)) {
            if (!discardIntegratedAttempt()) return failDeviceSearch();
        }
        if (lookahead_.ready && !lookaheadJobMatches(job)) {
            if (!discardPreparedLookahead()) return failDeviceSearch();
        }
        if (!beginJob(job)) return failSearch();

        // `count` has already been planned into complete tile grids by the run
        // loop. Refuse a caller that bypasses that contract instead of doing a
        // different amount of work than the core will report.
        const uint32_t perAttempt = tiles();
        if (count < perAttempt || count % perAttempt) {
            fprintf(stderr,
                    "[pearl-pow] candidate batch %llu is not a whole %u-tile "
                    "attempt\n",
                    (unsigned long long)count, perAttempt);
            return failSearch();
        }
        const uint64_t attempts = count / perAttempt;

        if (integratedEnabled())
            return searchIntegrated(job, nonceBase, attempts, out);
        if (attemptOverlapEnabled(overlapMode_))
            return searchOverlap(job, nonceBase, attempts, out);

        AttemptSchedule schedule(nonceBase, attempts);
        uint64_t nonce = 0;
        while (schedule.next(&nonce)) {
            progress(PearlAlgorithmProgressEvent::AttemptStarted);
            profiler_.begin();
            if (!runAttempt(nonce)) return failDeviceSearch();
            profiler_.enqueued();

            // Measured, not assumed: removing this host round trip entirely
            // is worth nothing (409.3 vs 410.3 M candidates/s). A ~1.3 ms GEMM
            // amortises a memcpy and a stream sync completely, so batching
            // attempts to avoid it would add real complexity - deterministic A
            // regeneration for a hit - to buy noise. Do not re-propose it
            // without a shorter kernel to justify it.
            hitReadback_ = 0;
            if (cudaMemcpyAsync(&hitReadback_, dHitCount_, sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, stream_) != cudaSuccess)
                return failDeviceSearch();
            const cudaError_t attemptErr = cudaStreamSynchronize(stream_);
            if (attemptErr != cudaSuccess) {
                fprintf(stderr, "[pearl-pow] attempt failed: %s\n",
                        cudaGetErrorString(attemptErr));
                return failDeviceSearch();
            }
            if (!recoverFusedOverflow(primaryA())) return failDeviceSearch();
            profiler_.complete(nonce, shape_.m, shape_.n, tile_, arch10_,
                               tuningKey_);
            ++completedAttempts_;
            progress(PearlAlgorithmProgressEvent::AttemptCompleted);
            const HitWindow window = hitWindow(hitReadback_);
            progress(PearlAlgorithmProgressEvent::HitsObserved,
                     window.observed);
            progress(PearlAlgorithmProgressEvent::HitsRetained,
                     window.retained);
            if (window.saturated)
                progress(PearlAlgorithmProgressEvent::HitBufferSaturated,
                         window.observed - window.retained);
            if (!captureRetiredAttempt(nonce, primaryA(), false, 0)) return failDeviceSearch();
            if (window.retained == 0) continue;
            if (window.saturated)
                fprintf(stderr,
                        "[pearl-pow] hit buffer saturated: observed %u, "
                        "retaining %u entries from this attempt\n",
                        window.observed, window.retained);

            // Capture the winning opening before its A slot is reused. Full
            // readback remains the reference; compact mode copies selected
            // rows and Merkle siblings while B stays immutable for this job.
            progress(PearlAlgorithmProgressEvent::FinalizerStarted);
            if (!openWin(job, nonce, window, out, primaryA())) return failSearch();
        }
        if (!schedule.complete()) {
            fprintf(stderr,
                    "[pearl-pow] nonce attempt schedule would wrap after %llu "
                    "of %llu attempts\n",
                    (unsigned long long)schedule.completed(),
                    (unsigned long long)attempts);
            return failSearch();
        }
        return true;
    }

    /**
     * Recompute the winning tile on the host and re-check it against the bound.
     *
     * This is the same arithmetic the node does from the opened rows and
     * columns: sixteen by sixteen over k, half a million multiply-accumulates.
     * A card overclocked into instability produces wrong accumulators, and
     * there is no other way to tell that from a genuine win.
     */
    /**
     * MACs one candidate costs, so the readout can report the unit the field
     * is measured in.
     *
     * A candidate is a kTileSide x kTileSide tile of the product and every
     * element of it is a length-k dot product, so the cost is tile area times
     * k. Note what is NOT in here: the cooperative-matrix K. That is the depth
     * of one hardware matrix instruction - 32 on Ada, 16 on RDNA3 - and a tile
     * takes k/K of them, so K cancels. Multiplying by the config's K instead
     * would be 64x low on one vendor and 128x low on the other, which is worse
     * than a hardcoded constant because it looks vendor-aware while being
     * vendor-specifically wrong.
     *
     * The CUDA inner dimension is fixed by this artifact, including before
     * the first beginJob publishes a runtime configuration.
     */
    double macsPerUnit() const override {
        // cfg_ initially contains the legacy default dimension. The run loop
        // may cache this value after prepare but before beginJob updates cfg_.
        // Use the same compiled K as the CUDA kernels and native selection.
        return (double)kTileSide * (double)kTileSide * (double)kK;
    }

    uint64_t workUnitsPerNonce() const override { return tiles(); }

    uint64_t completedSearchAttempts() const override {
        return completedAttempts_;
    }

    bool setPearlProgressSink(PearlAlgorithmProgressSink *sink) override {
        progressSink_ = sink;
        return true;
    }

    bool setPearlProofBindingSink(PearlProofBindingSink *sink) override {
        proofBindingSink_ = sink;
        return true;
    }

    bool verify(const Job &job, const Solution &sol) const override {
        PearlJobContext context;
        if (!validateRunJob(job, &context) || !publication_.ready())
            return false;
        // Newest first: if a nonce ever repeats, the current attempt's answer
        // is the right one.
        const Win *win = wins_.newest(
            sol.nonce, [](const Win &candidate) { return candidate.nonce; });
        // `sol.hit` came from the GPU scan and `sol.extra` is the proof emitted
        // after the host recheck. Do not merely check that the hit is below a
        // retained bound: a caller may have advanced to another job, changed a
        // pool job id, or paired the hit with different proof bytes. Only the
        // exact job and exact Solution payload owned by this search call may
        // authorize a submission.
        return job.valid && win && win->verified && win->jobValid &&
               winMatchesJob(*win, job) &&
               sol.extra == win->proof &&
               memcmp(sol.hit, win->digest.v, sizeof(sol.hit)) == 0 &&
               U256::fromLimbs(sol.hit).le(win->bound);
    }

#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
    bool testFusedScanSnapshot(PearlFusedScanSnapshot *out) noexcept {
        if (!out) return false;
        *out={};
        if (!allocated_ || !completedAttempts_ || !publication_.ready() || publication_.poisoned() || !fusedScanEnvironmentValid() ||
            !fusedScanSelectionValid() || integratedEnabled() || attemptOverlapEnabled(overlapMode_) ||
            !drainAttemptStreams()) return false;
        try {
            PearlFusedScanSnapshot value;
            value.fused=fusedScanEnabled(); value.completedAttempts=completedAttempts_;
            if (cudaMemcpy(&value.observed,dHitCount_,sizeof(uint32_t),cudaMemcpyDeviceToHost)!=cudaSuccess ||
                cudaMemcpy(value.target.data(),dTarget_,32,cudaMemcpyDeviceToHost)!=cudaSuccess) return false;
            const size_t retained=std::min<size_t>(value.observed,kMaxHits);
            value.indices.resize(retained); value.digests.resize(retained*8);
            if (retained && (cudaMemcpy(value.indices.data(),dHitIndex_,retained*4,cudaMemcpyDeviceToHost)!=cudaSuccess ||
                cudaMemcpy(value.digests.data(),dHitDigest_,retained*32,cudaMemcpyDeviceToHost)!=cudaSuccess)) return false;
            uint32_t indexCanary=0; std::array<uint32_t,8> digestCanary{};
            if (cudaMemcpy(&indexCanary,dHitIndex_+kMaxHits,4,cudaMemcpyDeviceToHost)!=cudaSuccess ||
                cudaMemcpy(digestCanary.data(),dHitDigest_+kMaxHits*8,32,cudaMemcpyDeviceToHost)!=cudaSuccess) return false;
            value.queueCanariesIntact=indexCanary==UINT32_C(0xa5a5a5a5);
            for (auto word:digestCanary) value.queueCanariesIntact &= word==UINT32_C(0xa5a5a5a5);
            value.valid=true; *out=std::move(value); return true;
        } catch (...) { return false; }
    }
    bool testFusedScanTargetProbe(const std::array<uint32_t,8> &target,
            PearlFusedScanSnapshot *out) noexcept {
        if (!out) return false;
        *out={};
        if (!allocated_ || !completedAttempts_ || !publication_.ready() || publication_.poisoned() || batchCaptures_ || singleCaptureActive_ ||
            integratedEnabled() || attemptOverlapEnabled(overlapMode_) ||
            !fusedScanEnvironmentValid() || !fusedScanSelectionValid() ||
            !drainAttemptStreams()) return false;
        // Diagnostic execution never authorizes a share, and cannot preserve an
        // old solution's submission authority after overwriting the hit queue.
        wins_.beginSearch();
        std::array<uint32_t,8> original{};
        if (cudaMemcpy(original.data(),dTarget_,32,cudaMemcpyDeviceToHost)!=cudaSuccess) return false;
        bool ok=cudaMemcpy(dTarget_,target.data(),32,cudaMemcpyHostToDevice)==cudaSuccess;
        if (ok) ok=cudaMemsetAsync(dHitCount_,0,4,stream_)==cudaSuccess && computeA(primaryA(),false);
        // Drain even a partially enqueued failure before restoring its inputs.
        const bool drained=drainAttemptStreams();
        if (ok && drained) ok=testFusedScanSnapshot(out);
        const bool restored=drained && cudaMemcpy(dTarget_,original.data(),32,cudaMemcpyHostToDevice)==cudaSuccess;
        if (!restored) publication_.poison();
        if (!ok || !drained || !restored) { *out={}; return false; }
        return true;
    }
    bool testIntegratedPipelineStats(PearlSpeculativeAttemptCounters *out) const noexcept {
        if (!out || !integratedAttempt_.healthy() || !integratedConserved()) return false;
        *out = integratedAttempt_.counters();
        return true;
    }
    bool testIntegratedPipelineTimings(IntegratedPipelineHostTimings *out) const noexcept {
        if (!out) return false;
        *out = integratedTimings_;
        return true;
    }
    // Snapshot-only reference factory. Caller temporarily selects the absent
    // overlap environment for the new serial instance; source is idle and
    // remains an ordinary-prepared APrepV1 instance, never a caller tuple.
    Algorithm *testCreateSerialReplay(const Job &job) noexcept {
        try {
            for(const char *key:{"CANARY_PEARL_EXPERIMENT","CANARY_PEARL_SHAPE","CANARY_PEARL_SHAPE_EXPERIMENT",
                    "CANARY_PEARL_TILE","CANARY_PEARL_PROFILE_INTERVAL","CANARY_PEARL_A_ROOT_FUSION",
                    "CANARY_PEARL_ATTEMPT_OVERLAP","CANARY_PEARL_PRIVATE_EXECUTION","CANARY_PEARL_PRIVATE_EXECUTION_MODE",
                    "CANARY_PEARL_NATIVE_EVIDENCE","CANARY_PEARL_HARD_TARGET","CANARY_PEARL_BENCH_ALL_HIT"})
                if(getenv(key))return nullptr;
            if(!configured_ || !allocated_ || syntheticRun_ || formalRequest_.enabled ||
                (privateExecution_!=PearlPrivateExecution::Disabled || publicExecution_!=PearlPublicExecution::Disabled) ||
                experimentSelection_!=CudaFixedExperimentSelection::Disabled ||
                overlapMode_!=AttemptOverlapMode::APrepV1 || rootFusionMode_!=ARootFusionMode::Disabled ||
                batchCaptures_ || singleCaptureActive_ || completedAttempts_ || lookahead_.ready ||
                !drainAttemptStreams())return nullptr;
            const auto selected=pearlNativeSelection();
            bool replaySelection=pearlSm120NativeSelectionValid(selected);
            if(!replaySelection)return nullptr;
            PearlJobContext context;
            if(!validateRunJob(job,&context))return nullptr;
            cudaDeviceProp prop{};int dev=0;size_t freeB=0,totalB=0;
            if(cudaGetDevice(&dev)!=cudaSuccess || cudaGetDeviceProperties(&prop,dev)!=cudaSuccess ||
                prop.major*10+prop.minor!=static_cast<int>(selected.sm) || tuningKey_!=std::string(reinterpret_cast<const char *>(prop.uuid.bytes),sizeof(prop.uuid.bytes)) ||
                cudaMemGetInfo(&freeB,&totalB)!=cudaSuccess)return nullptr;
            std::unique_ptr<PearlPow> replay(new PearlPow());
            RunOptions options;options.poolHost="127.0.0.1";options.poolPort=1;options.pearlPolicy=sourcePolicy_;
            replay->configure(options);
            if(!replay->experimentConfigurationValid_ || replay->overlapMode_!=AttemptOverlapMode::Disabled)return nullptr;
            size_t matches=0;
            for(size_t i=0;i<kCudaPreparationPlanCount;++i) {
                const auto &plan=tileConfig(i);
                if(plan.preparationPlanId==selected.preparationPlanId && plan.kernelId==selected.kernelId && plan.tileId==selected.tileId) {
                    replay->tile_=i;++matches;
                }
            }
            const auto *shape=findCudaPreparationShape(selected.m,selected.n);
            if(matches!=1 || !shape)return nullptr;
            replay->shape_=*shape;
            replay->arch10_=static_cast<int>(selected.sm);replay->tuningKey_=tuningKey_;
            constexpr size_t headroom=512ULL*1024*1024;
            if(replay->bytesFor(replay->shape_)+headroom>freeB ||
                cudaStreamCreate(&replay->stream_)!=cudaSuccess || !replay->allocate())return nullptr;
            replay->allocated_=true;
            if(!replay->selfCheck() || !replay->beginJob(job) || !replay->completeNativePreparation() ||
                !pearlNativeSelectionEqual(replay->pearlNativeSelection(),selected))return nullptr;
            return replay.release();
        } catch(...) { return nullptr; }
    }

    bool testRetirementStats(PearlRetirementStats *out) const noexcept {
        if (!out) return false;
        *out = PearlRetirementStats{};
        if (!retirementStatsValid_) return false;
        *out = retirementStats_;
        out->snapshotCaptureActive = batchCaptures_ != nullptr || singleCaptureActive_;
        return true;
    }

    bool testSearchBatchAndCapture(const Job &job, uint64_t nonce, uint64_t attempts,
                    std::vector<PearlTranscriptSnapshot> *captures,
                    std::vector<Solution> *solutions) noexcept {
        if (!captures || !solutions || batchCaptures_ || singleCaptureActive_) return false;
        captures->clear();
        solutions->clear();
        try {
            if (!allocated_ || !attempts || attempts > UINT64_MAX / tiles() ||
                attempts > UINT64_MAX - completedAttempts_) return failSearch();
            std::vector<PearlTranscriptSnapshot> next;
            std::vector<Solution> proofs;
            batchCaptures_ = &next;
            bool ok = search(job, nonce, attempts * tiles(), &proofs);
            batchCaptures_ = nullptr;
            ok = ok && next.size() == attempts && publication_.ready();
            for (const auto &proof : proofs) ok = ok && verify(job, proof);
            if (!ok) return failSearch();
            *captures = std::move(next);
            *solutions = std::move(proofs);
            return true;
        } catch (...) {
            batchCaptures_ = nullptr;
            return failSearch();
        }
    }

    bool testSearchAndCapture(const Job &job, uint64_t nonce,
                              PearlTranscriptSnapshot *out) noexcept {
        if (!out || batchCaptures_ || singleCaptureActive_) return false;
        *out = PearlTranscriptSnapshot{};
        try {
            // Common retirement owns the actual A slot. The primary slot may
            // already contain a successor while the current nonce retires slot1.
            std::vector<PearlTranscriptSnapshot> captures;
            std::vector<Solution> solutions;
            if (!testSearchBatchAndCapture(job,nonce,1,&captures,&solutions) ||
                captures.size() != 1 || solutions.size() != 1 ||
                solutions.front().nonce != nonce || captures.front().nonce != nonce)
                return failSearch();
            *out = std::move(captures.front());
            return true;
        } catch (...) { return failSearch(); }
    }
#endif

    void pearlNextAttemptHint(bool valid, uint64_t nonce) noexcept override {
        nextHintValid_ = valid && (attemptOverlapEnabled(overlapMode_) || integratedEnabled());
        nextHintNonce_ = nonce;
    }
    bool discardPearlLookahead() noexcept override {
        nextHintValid_ = false;
        return discardPreparedLookahead();
    }

    void release() override {
        discardPearlLookahead();
        nativeSelection_.invalidate();
        profiler_.cancel();
        drainAttemptStreams();
        releaseIntegrated();
        std::vector<int8_t>().swap(proofReadbackA_);
        std::vector<int8_t>().swap(proofReadbackB_);
        bReadbackValid_ = false;
        bCompactTreeValid_ = false;
        bProofCache_.clear();
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
        aRootEvidence_.invalidate();
#endif
        for (auto &event : aReady_) {
            if (event && cudaEventDestroy(event) != cudaSuccess) publication_.poison();
            event = nullptr;
        }
        if (prepStream_ && cudaStreamDestroy(prepStream_) != cudaSuccess)
            publication_.poison();
        prepStream_ = nullptr;
        for (void *p : owned_)
            if (cudaFree(p) != cudaSuccess) publication_.poison();
        owned_.clear();
        alternateA_ = ASlot{};
        allocated_ = false;
        publication_.abandonRetryable();
        wins_.beginSearch();
    }

#include "mixed_schedule_probe.inc"
   private:
#include "integrated_pipeline.inc"
#include "fused_scan_backend.inc"
    PearlNativeSelection currentNativeSelection() const noexcept {
        if (!configured_ || !allocated_ || publication_.poisoned() || arch10_ <= 0 ||
            !shapeEnvironmentValid() || !shapeExperiment_.observeShape(shape_.m, shape_.n) ||
            !experimentConfigurationValid_ || !fixedWm8EnvironmentValid() ||
            !fusedScanSelectionValid() || !integratedSelectionValid() || !rootFusionEnvironmentValid() || (!pearlPrivateEnvironment(privateExecution_) || !pearlPublicEnvironment(publicExecution_))) return {};
        // tileConfig's fallback indexes the default array. Reject any index
        // outside it unless it names the enabled, compiled isolated launch.
        const bool tmaDelivery = tile_==kTmaDeliveryTile && arch10_==120 &&
            shape_.m==16384 && shape_.n==131072 && kK==8192 && kRank==128 &&
            (pearlProductFixedSelectionActive() ||
             shapeExperiment_.value()==PearlShapeExperiment::Exact16384x131072K8192V1) &&
            tmaDeliveryTileFilterMatches(tileFilter_);
        if (tile_ >= kCudaPreparationPlanCount && !tmaDelivery &&
            (!cudaIsolatedExperimentEnabled(experimentSelection_) ||
             !experimentCompiled(experimentSelection_) ||
             tile_ != experimentTile(experimentSelection_))) return {};
        const auto &plan = tileConfig(tile_);
        if (!exactStageTiming_.configuration(arch10_, shape_.m, shape_.n, kK, kRank,
                                             plan.preparationPlanId)) return {};
        PreparedJobRequest request;
        const auto mode = syntheticRun_ ? PreparedJobMode::SyntheticHardTarget
                                        : PreparedJobMode::Normal;
        if (!plan.launch || !buildCudaPreparedJobRequest(
                plan, shape_.m, shape_.n, arch10_, mode, &request)) return {};
        PearlNativeSelection selection;
        selection.valid = true;
        selection.m = request.m;
        selection.n = request.n;
        selection.k = kK;
        selection.rank = kRank;
        selection.sm = static_cast<uint32_t>(arch10_);
        selection.preparationPlanId = request.preparationPlanId;
        selection.kernelId = plan.kernelId;
        selection.tileId = plan.tileId;
        if (!pearlNativeSelectionWork(selection.m, selection.n,
                                      &selection.workUnitsPerAttempt) ||
            !pearlNativeSelectionValid(selection)) return {};
        return selection;
    }

    bool completeNativePreparation() noexcept {
        return nativeSelection_.publish(currentNativeSelection());
    }

    bool exactStageTimingActive() const noexcept {
        return exactStageTiming_.enabled() && nativeSelection_.read(currentNativeSelection()).valid;
    }
    const char *proofTimingOption() const noexcept {
        return exactStageTimingActive() ? "1" : getenv("CANARY_PEARL_PROOF_TIMING");
    }
    bool shapeEnvironmentValid() const noexcept {
        if (pearlProductFixedSelectionActive())
            return shapeConfigurationValid_ && shapeExperiment_.admitProductFixed(
                getenv("CANARY_PEARL_SHAPE"), getenv("CANARY_PEARL_TILE"));
        return shapeConfigurationValid_ &&
            exactStageTiming_.environment(getenv("CANARY_PEARL_STAGE_TIMING"),
                getenv("CANARY_PEARL_PROFILE_INTERVAL"), formalRequest_.enabled ||
                getenv("CANARY_PEARL_EXPERIMENT") || getenv("CANARY_PEARL_ATTEMPT_OVERLAP") ||
                getenv("CANARY_PEARL_A_ROOT_FUSION")) && shapeExperiment_.revalidate(
            getenv("CANARY_PEARL_SHAPE_EXPERIMENT"), getenv("CANARY_PEARL_SHAPE"),
            pearlGenericNativeEvidenceMode(getenv("CANARY_PEARL_NATIVE_EVIDENCE")),
            shapeBounded_, shapeOwnedGuard_,
            getenv("CANARY_PEARL_EXPERIMENT") || !shapeExperiment_.tileAllowed(getenv("CANARY_PEARL_TILE")) ||
            getenv("CANARY_PEARL_PROFILE_INTERVAL") || getenv("CANARY_PEARL_A_ROOT_FUSION") ||
            getenv("CANARY_PEARL_ATTEMPT_OVERLAP") || getenv("CANARY_PEARL_HARD_TARGET") ||
            getenv("CANARY_PEARL_BENCH_ALL_HIT"), getenv("CANARY_PEARL_TILE"));
    }

    bool fixedWm8EnvironmentValid() const noexcept {
        // Constructor-captured conflicts cannot be removed to bypass admission;
        // later insertions must revoke work before prepare/search device effects.
        // Formal catalog selection has the same isolation requirement even
        // though it deliberately captures the Disabled experiment enum.  A
        // control inserted after prepare must not change or relabel its search.
        if (formalRequest_.enabled)
            return !getenv("CANARY_PEARL_EXPERIMENT") &&
                   !getenv("CANARY_PEARL_ATTEMPT_OVERLAP") &&
                   !getenv("CANARY_PEARL_A_ROOT_FUSION") &&
                   !getenv("CANARY_PEARL_PROFILE_INTERVAL") &&
                   !getenv("CANARY_PEARL_SHAPE") &&
                   !getenv("CANARY_PEARL_TILE") &&
                   !getenv("CANARY_PEARL_HARD_TARGET") &&
                   !getenv("CANARY_PEARL_BENCH_ALL_HIT");
        if (experimentSelection_ != CudaFixedExperimentSelection::Wm8FixedK2048R128 &&
            experimentSelection_ != CudaFixedExperimentSelection::Bk32FixedK2048R128 &&
            experimentSelection_ != CudaFixedExperimentSelection::Wm4ReuseFixedK2048R128 &&
            experimentSelection_ != CudaFixedExperimentSelection::TmaWm8FixedK2048R128 &&
            experimentSelection_ != CudaFixedExperimentSelection::TmaTraversal16x32 &&
            experimentSelection_ != CudaFixedExperimentSelection::TmaTraversal16x32Stage4) return true;
        return cudaWm8FixedEnvironmentValid(experimentSelection_, getenv("CANARY_PEARL_EXPERIMENT"),
            fixedWm8ConflictCaptured_,
            getenv("CANARY_PEARL_ATTEMPT_OVERLAP") || getenv("CANARY_PEARL_A_ROOT_FUSION") ||
            getenv("CANARY_PEARL_PROFILE_INTERVAL") || getenv("CANARY_PEARL_SHAPE") ||
            getenv("CANARY_PEARL_TILE") || getenv("CANARY_PEARL_HARD_TARGET") ||
            getenv("CANARY_PEARL_BENCH_ALL_HIT"));
    }
    bool rootFusionEnvironmentValid() const noexcept {
        return parseARootFusion(getenv("CANARY_PEARL_A_ROOT_FUSION")) == rootFusionMode_ &&
            (rootFusionMode_ != ARootFusionMode::Enabled ||
             (!rootFusionProfilingCaptured_ && !getenv("CANARY_PEARL_PROFILE_INTERVAL") &&
              parseCudaFixedExperiment(getenv("CANARY_PEARL_EXPERIMENT")) == experimentSelection_));
    }
    bool drainAttemptStreams() noexcept {
        // Always try both, including when the first reports an asynchronous
        // error. No event, host destination, or slot is freed before draining.
        const cudaError_t prep = prepStream_ ? cudaStreamSynchronize(prepStream_) : cudaSuccess;
        const cudaError_t compute = stream_ ? cudaStreamSynchronize(stream_) : cudaSuccess;
        const cudaError_t copy = copyStream_ ? cudaStreamSynchronize(copyStream_) : cudaSuccess;
        if (prep != cudaSuccess || compute != cudaSuccess || copy != cudaSuccess) publication_.poison();
        return prep == cudaSuccess && compute == cudaSuccess && copy == cudaSuccess;
    }

    bool allocateOverlap() {
        if (!attemptOverlapEnabled(overlapMode_) && !integratedEnabled()) return true;
        // Kept separate from the compute stream, and nonblocking with respect
        // to legacy default-stream host proof copies. No launch sizes change.
        const size_t bytes = static_cast<size_t>(shape_.m) * kK;
        bool ok = false;
        try {
            ok = alloc(&alternateA_.raw, bytes) && alloc(&alternateA_.noised, bytes) &&
                alloc(&alternateA_.root, 32) && alloc(&alternateA_.commit, 32) &&
                cudaStreamCreateWithFlags(&prepStream_, cudaStreamNonBlocking) == cudaSuccess &&
                cudaEventCreateWithFlags(&aReady_[0], cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&aReady_[1], cudaEventDisableTiming) == cudaSuccess;
        } catch (...) { ok = false; }
        if (!ok) publication_.poison(); // release owns every partial resource
        return ok;
    }

    bool lookaheadJobMatches(const Job &job) const noexcept {
        return publication_.matches(lookaheadPrepared_) && staging_ == lookaheadPrepared_ &&
            !std::memcmp(job.msg,lookaheadJob_.msg,sizeof(job.msg)) &&
            !std::memcmp(job.target,lookaheadJob_.target,sizeof(job.target)) &&
            job.epoch == lookaheadJob_.epoch && job.extra == lookaheadJob_.extra &&
            job.jobId == lookaheadJob_.jobId && job.valid == lookaheadJob_.valid &&
            parseAttemptOverlap(getenv("CANARY_PEARL_ATTEMPT_OVERLAP")) == overlapMode_ &&
            rootFusionEnvironmentValid();
    }
    bool discardPreparedLookahead() noexcept {
        const bool integratedDrained = discardIntegratedAttempt();
        if (executionAccountingEnabled() && lookahead_.ready) {
            if (!privateStats_.discardReady()) privateStats_.valid=false;
        }
        lookahead_.discard();
        if (!attemptOverlapEnabled(overlapMode_)) return integratedDrained;
        const bool drained = drainAttemptStreams();
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
        aRootEvidence_.invalidate();
#endif
        return drained && integratedDrained;
    }

    struct OverlapOps {
        PearlPow &p;
        const Job &job;
        std::vector<Solution> *out;
        bool prepare(unsigned slot, uint64_t nonce) {
            const unsigned threads = attemptOverlapThreads(p.overlapMode_);
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
            p.testPreparedThreads_[slot] = threads;
#endif
            return p.prepareA(p.aSlot(slot), nonce, p.prepStream_, false, threads) &&
                cudaEventRecord(p.aReady_[slot], p.prepStream_) == cudaSuccess;
        }
        bool launch(unsigned slot, uint64_t) {
            p.progress(PearlAlgorithmProgressEvent::AttemptStarted);
            return cudaStreamWaitEvent(p.stream_, p.aReady_[slot], 0) == cudaSuccess &&
                cudaMemsetAsync(p.dHitCount_, 0, sizeof(uint32_t), p.stream_) == cudaSuccess &&
                p.computeA(p.aSlot(slot), false);
        }
        bool retire(unsigned slot, uint64_t nonce) {
            p.hitReadback_ = 0;
            if (cudaMemcpyAsync(&p.hitReadback_, p.dHitCount_, sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, p.stream_) != cudaSuccess ||
                p.synchronizeStreamStatus() != cudaSuccess) return false;
            ++p.completedAttempts_;
            p.progress(PearlAlgorithmProgressEvent::AttemptCompleted);
            const HitWindow window = hitWindow(p.hitReadback_);
            p.progress(PearlAlgorithmProgressEvent::HitsObserved, window.observed);
            p.progress(PearlAlgorithmProgressEvent::HitsRetained, window.retained);
            if (window.saturated)
                p.progress(PearlAlgorithmProgressEvent::HitBufferSaturated,
                           window.observed - window.retained);
            if (!p.captureRetiredAttempt(nonce, p.aSlot(slot), true, slot)) return false;
            if (!window.retained) return true;
            p.progress(PearlAlgorithmProgressEvent::FinalizerStarted);
            return p.openWin(job, nonce, window, out, p.aSlot(slot));
        }
        bool drain() noexcept { return p.drainAttemptStreams(); }
    };

    bool searchOverlap(const Job &job, uint64_t nonceBase, uint64_t attempts,
                       std::vector<Solution> *out) noexcept {
        const size_t before = out->size();
        if (!attempts || attempts - 1 > UINT64_MAX - nonceBase ||
            attempts > UINT64_MAX - completedAttempts_) return failSearch();
        if (!prepStream_ || !aReady_[0] || !aReady_[1]) return failDeviceSearch();
        if (!overlapAnnounced_) {
            overlapAnnounced_ = true;
            // Actual path entry, not successful completion or share credit.
            // One per backend lifetime; correctness preflight precedes timing.
            fprintf(stderr,
                    "[pearl-overlap] path=%s plan=0x%016llx shape=%ux%u first_attempts=%llu\n",
                    attemptOverlapName(overlapMode_),
                    (unsigned long long)staging_.preparationPlanId,
                    shape_.m, shape_.n, (unsigned long long)attempts);
        }
        OverlapOps ops{*this, job, out};
        const bool hint = nextHintValid_;
        const uint64_t next = nextHintNonce_;
        nextHintValid_ = false; // one search owns each scheduler hint
        bool ok = false;
        try {
            if (attempts == 1) {
                if (lookahead_.ready && !lookaheadJobMatches(job)) {
                    if (!discardPreparedLookahead()) return failDeviceSearch();
                }
                // Copy potentially allocating job strings before enqueueing CUDA.
                lookaheadJob_ = job;
                lookaheadPrepared_ = staging_;
                const bool ownedReady=lookahead_.ready;
                privateSearchCacheHit_=ownedReady && lookahead_.nonce==nonceBase;
                ok = runSingleAttemptLookahead(nonceBase,hint,next,lookahead_,ops);
                if (executionAccountingEnabled()) {
                    if (!privateStats_.finishSearch(ownedReady,privateSearchCacheHit_,ok,lookahead_.ready)) ok=false;
                }
            } else {
                if (!discardPreparedLookahead()) return failDeviceSearch();
                ok = runAttemptOverlapBatch(nonceBase,attempts,ops);
            }
        } catch (...) {
            discardPreparedLookahead();
        }
        if (!ok) {
            // Bound proofs from earlier attempts are retired as one failed
            // search by the run-session ledger too; never expose partial output.
            out->resize(before);
            return failDeviceSearch();
        }
        return true;
    }

    bool captureRetiredAttempt(uint64_t nonce, const ASlot &a, bool overlap, unsigned slot) {
        if (executionAccountingEnabled()) {
            if (!privateStats_.valid || slot>1) return false;
            privateStats_.lastValid=true;
            privateStats_.nonce=nonce;
            privateStats_.slot=slot;
            privateStats_.cacheHit=overlap && privateSearchCacheHit_;
        }
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
        // Actual common retirement, including unarmed whole-search timing.
        // Counters cannot make wrapping evidence look like an empty window.
        bool fused = false;
        if (!aRootEvidence_.retire(slot, nonce, &fused) ||
            fused != (rootFusionMode_ == ARootFusionMode::Enabled)) {
            retirementStatsValid_ = false;
            return false;
        }
        uint64_t &rootCount = fused ? retirementStats_.aRootFusionRetirements
                                   : retirementStats_.aRootSeparateRetirements;
        uint64_t &pathCount = overlap ? retirementStats_.overlapRetirements
                                      : retirementStats_.serialRetirements;
        if (!retirementStatsValid_ || slot >= 2 || rootCount == UINT64_MAX || pathCount == UINT64_MAX ||
            retirementStats_.slotRetirements[slot] == UINT64_MAX ||
            (batchCaptures_ && retirementStats_.snapshotCaptures == UINT64_MAX)) {
            retirementStatsValid_ = false;
            return false;
        }
        ++pathCount;
        ++rootCount;
        retirementStats_.aRootFusion = fused;
        ++retirementStats_.slotRetirements[slot];
        retirementStats_.preparationPlanId = staging_.preparationPlanId;
        retirementStats_.preparationThreads = overlap ? testPreparedThreads_[slot] : 256;
        if (!batchCaptures_) return true;
        PearlTranscriptSnapshot next;
        next.overlapRetirement = overlap;
        next.aSlot = slot;
        next.preparationThreads = retirementStats_.preparationThreads;
        next.aRootFusion = fused;
        next.nonce = nonce;
        next.completedBefore = completedAttempts_ - 1;
        next.completedAfter = completedAttempts_;
        next.prepared = staging_;
        next.rawA.resize(static_cast<size_t>(shape_.m) * kK);
        next.rawBt.resize(static_cast<size_t>(shape_.n) * kK);
        next.transcripts.resize(static_cast<size_t>(tiles()) * kTranscriptWords);
        if (cudaMemcpy(next.rawA.data(), a.raw, next.rawA.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(next.rawBt.data(), dBt_, next.rawBt.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(next.transcripts.data(), dTranscripts_, next.transcripts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(next.roots.data(), a.root, 32, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(next.roots.data() + 32, dRoots_ + 32, 32, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(next.commitA.data(), a.commit, 32, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(next.commitB.data(), dCommitB_, 32, cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        next.valid = true;
        batchCaptures_->push_back(std::move(next));
        ++retirementStats_.snapshotCaptures;
#else
        (void)nonce; (void)a; (void)overlap; (void)slot;
#endif
        return true;
    }

    void freeTemporary(void *pointer) {
        if (pointer && cudaFree(pointer) != cudaSuccess)
            publication_.poison();
    }

    cudaError_t synchronizeStreamStatus() {
        const cudaError_t error = cudaStreamSynchronize(stream_);
        if (error != cudaSuccess) publication_.poison();
        return error;
    }

    bool validateRunJob(const Job &job, PearlJobContext *context) const {
        if (!configured_ || !experimentConfigurationValid_ || !context ||
            tma_fused::parse(getenv("CANARY_PEARL_TMA_DATAFLOW"))!=fusedMode_) return false;
        std::string error;
        if (!validatePearlJobContext(
                job, syntheticRun_ ? nullptr : &sourcePolicy_, context,
                &error)) {
            fprintf(stderr, "[pearl-pow] invalid canonical job: %s\n",
                    error.c_str());
            return false;
        }
        return (context->network == PearlNetwork::SyntheticBenchmark) ==
                   syntheticRun_ &&
               !getenv("CANARY_PEARL_BENCH_ALL_HIT") &&
               (syntheticRun_ || !getenv("CANARY_PEARL_HARD_TARGET"));
    }

    struct Win {
        uint64_t nonce = 0;
        bool verified = false;
        U256 bound;
        U256 digest;
        uint8_t jobMsg[32] = {};
        uint64_t jobTarget[4] = {};
        uint64_t jobEpoch = 0;
        std::string jobExtra;
        std::string jobId;
        bool jobValid = false;
        std::string proof;
    };

    static void bindWinToJob(Win *win, const Job &job) {
        memcpy(win->jobMsg, job.msg, sizeof(win->jobMsg));
        memcpy(win->jobTarget, job.target, sizeof(win->jobTarget));
        win->jobEpoch = job.epoch;
        win->jobExtra = job.extra;
        win->jobId = job.jobId;
        win->jobValid = job.valid;
    }

    static bool winMatchesJob(const Win &win, const Job &job) {
        return memcmp(win.jobMsg, job.msg, sizeof(win.jobMsg)) == 0 &&
               memcmp(win.jobTarget, job.target, sizeof(win.jobTarget)) == 0 &&
               win.jobEpoch == job.epoch && win.jobExtra == job.extra &&
               win.jobId == job.jobId && win.jobValid == job.valid;
    }

    bool failSearch() {
        if (executionAccountingEnabled()) privateStats_.valid=false;
        discardIntegratedAttempt();
        lookahead_.discard(); nextHintValid_ = false;
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
        aRootEvidence_.invalidate();
#endif
        profiler_.cancel();
        if (attemptOverlapEnabled(overlapMode_) || integratedEnabled()) drainAttemptStreams();
        // A false search result must authorize no partial proof, even if an
        // earlier attempt in the same batch had already opened a candidate.
        wins_.beginSearch();
        return false;
    }

    bool failDeviceSearch() {
        bReadbackValid_ = false;
        bCompactTreeValid_ = false;
        // No subsequent request may reuse possibly partial device state.
        drainAttemptStreams();
        publication_.poison();
        return failSearch();
    }

    void progress(PearlAlgorithmProgressEvent event, uint64_t amount = 1,
                  PearlProgressReason reason =
                      PearlProgressReason::None) noexcept {
        if (progressSink_)
            progressSink_->onPearlAlgorithmProgress(event, amount, reason);
    }

    uint32_t tiles() const {
        return (shape_.m / kTileSide) * (shape_.n / kTileSide);
    }

    size_t bytesFor(const Shape &s) const {
        const size_t k = kK, rank = kRank;
        const size_t aBytes = (size_t)s.m * k, bBytes = (size_t)s.n * k;
        const size_t chunks = (aBytes > bBytes ? aBytes : bBytes) / kChunkLen;
        const size_t tileCount = (size_t)(s.m / kTileSide) * (s.n / kTileSide);
        return ((attemptOverlapEnabled(overlapMode_) || integratedEnabled()) ? aBytes * 2 + 64 : 0)
               + (integratedEnabled() ? 4*kIntegratedOpeningBytes +
                    4*CompactProofPlan::maxSiblings*sizeof(uint64_t) : 0)
               + aBytes * 2                       // A, A_noised
               + bBytes * 2                     // B^t, B_noised
               + (size_t)s.m * rank             // E_AL
               + (size_t)s.n * rank * 2         // E_BR, flat and transposed
               + k * 8                          // permutation index pairs
               + chunks * 32 * 2                // merkle scratch
               + tileCount * 16 * sizeof(uint32_t)   // transcripts
               + kMaxHits * 36 + 4096
               + (tma_fused::fused(fusedMode_)?kMaxHits*64:0);
    }

    static uint64_t shapeKey(const Shape &s) {
        return (uint64_t{s.m} << 32) | uint64_t{s.n};
    }

    template <typename T>
    bool alloc(T **p, size_t bytes) {
        if (cudaMalloc(p, bytes) != cudaSuccess) return false;
        try { owned_.push_back(*p); }
        catch (...) {
            if (cudaFree(*p) != cudaSuccess) publication_.poison();
            *p = nullptr;
            throw;
        }
        return true;
    }

    /**
     * Measure 32 complete serial attempts at this shape: prepare, GEMM, scan
     * and the per-attempt hit-count readback/drain that search() runs. The
     * caller visits every surviving shape
     * once in each of seven deterministic rounds, with three
     * permutation/reverse pairs and a fresh final permutation.
     *
     * This exists because ranking shapes on the GEMM alone picks the wrong
     * one. The work beside it goes as m*k while candidates go as m*n, so the
     * overhead per candidate falls as 1/n and the widest shape that fits tends
     * to win. The GEMM timing cannot see any of that. Measured on a 4090, the
     * GEMM-only ranking chose 8192x16384 at 320 M candidates/s end to end
     * while 4096x65536 does 337, and it picked a DIFFERENT shape on one run in
     * four because it was ranking near-identical GEMM rates as noise.
     *
     * Buffers are filled before timing, and the target is zeroed so powScan
     * finds nothing and scans the lot either way. Its cost does not depend on
     * the data, but reading pages that were never written is not measuring the
     * same thing as reading real ones.
     */
    bool measureAttemptSample(const Shape &s, size_t cfg, size_t epoch, size_t round,
                              double *rateOut) {
        if (!rateOut || epoch >= pearl::tuner::kMaximumFullAttemptTuningEpochs ||
            round >= pearl::tuner::kSampleCount) return false;
        shape_ = s;
        tile_ = cfg;
        std::uint64_t completed = 0;
        float elapsedMs = 0.0f;
        pearl::tuner::RateSample rate;
        auto recordSample = [&](bool valid) {
            fprintf(stderr, "[pearl-pow] full-attempt sample epoch=%zu round=%zu "
                    "shape=%ux%u plan=0x%016llx candidates=%llu elapsed_ms=%.9g "
                    "rate_cps=%.17g valid=%d attempts=%llu completed=%llu tiles=%llu\n",
                    epoch + 1, round, s.m, s.n,
                    static_cast<unsigned long long>(tileConfig(cfg).preparationPlanId),
                    static_cast<unsigned long long>(tiles()) * pearl::tuner::kFullAttemptBatchCount,
                    static_cast<double>(elapsedMs), valid ? rate.candidatesPerSecond : 0.0,
                    valid ? 1 : 0,
                    static_cast<unsigned long long>(pearl::tuner::kFullAttemptBatchCount),
                    static_cast<unsigned long long>(completed),
                    static_cast<unsigned long long>(tiles()));
        };
        if (!allocate()) {
            publication_.poison();
            release();
            recordSample(false);
            return false;
        }

        // Initialize through the exact per-job path used by search(). This is
        // not optional setup: runAttempt() reads the host jobSalt_ and device
        // job key, commitments and salts/seeds. Hand-filling A/B and target
        // left those inputs uninitialized and made the timing irreproducible.
        Job fixture;
        fillBenchJob(&fixture);
        if (!fixture.valid || !beginJob(fixture, true)) {
            publication_.poison();
            release();
            recordSample(false);
            return false;
        }

        // Same ownership boundary as serial search(): copy count and drain
        // EACH attempt before reset/reuse. Do not benchmark a queued pipeline
        // that production does not run. Synthetic zero target must have no hit.
        auto completeAttempt = [&](std::uint64_t nonce) {
            hitReadback_ = 0;
            if (!runAttempt(nonce) ||
                cudaMemcpyAsync(&hitReadback_, dHitCount_, sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
                synchronizeStreamStatus() != cudaSuccess || hitReadback_ != 0) {
                publication_.poison();
                return false;
            }
            return true;
        };
        std::uint64_t warmupNonce = 0;
        if (!pearl::tuner::fullAttemptNonce(s.m, s.n, epoch, round, 0, &warmupNonce) ||
            !completeAttempt(warmupNonce)) {
            publication_.poison();
            release();
            recordSample(false);
            return false;
        }

        cudaEvent_t evA = nullptr, evB = nullptr;
        bool ok = true;
        auto eventCall = [&](cudaError_t eventErr, const char *stage) {
            if (eventErr == cudaSuccess) return true;
            publication_.poison();
            fprintf(stderr,
                    "[pearl-pow] CUDA event %s failed for %ux%u attempt "
                    "round %zu: %s\n",
                    stage, s.m, s.n, round, cudaGetErrorString(eventErr));
            ok = false;
            return false;
        };
        if (!eventCall(cudaEventCreate(&evA), "creation (start)")) {
            release();
            recordSample(false);
            return false;
        }
        if (!eventCall(cudaEventCreate(&evB), "creation (stop)")) {
            eventCall(cudaEventDestroy(evA), "destruction (start)");
            release();
            recordSample(false);
            return false;
        }

        if (eventCall(cudaEventRecord(evA, stream_), "recording (start)")) {
            if (!pearl::tuner::completeFullAttemptBatch(
                    s.m, s.n, epoch, round, completeAttempt, &completed)) {
                fprintf(stderr,
                        "[pearl-pow] complete full-attempt batch failed for %ux%u "
                        "round %zu after %llu drained attempts\n",
                        s.m, s.n, round, static_cast<unsigned long long>(completed));
                publication_.poison();
                ok = false;
            }
        }
        if (ok &&
            (!eventCall(cudaEventRecord(evB, stream_), "recording (stop)") ||
             !eventCall(cudaEventSynchronize(evB), "synchronization"))) {
            ok = false;
        }
        if (ok && !eventCall(cudaEventElapsedTime(&elapsedMs, evA, evB),
                             "elapsed-time query")) {
            ok = false;
        }
        if (ok) {
            rate = pearl::tuner::fullAttemptBatchRate(tiles(), completed, elapsedMs);
            if (!rate.valid) {
                fprintf(stderr,
                        "[pearl-pow] %ux%u full-attempt sample rejected in "
                        "round %zu: %s\n",
                        s.m, s.n, round,
                        pearl::tuner::rejectionReasonName(rate.rejection));
                ok = false;
                publication_.poison();
            }
        }
        eventCall(cudaEventDestroy(evA), "destruction (start)");
        eventCall(cudaEventDestroy(evB), "destruction (stop)");
        release();
        // Cleanup errors also invalidate the complete observation. Never use
        // partial candidates or include an earlier rejected epoch's samples.
        recordSample(ok && rate.valid && !publication_.poisoned());
        if (!ok || publication_.poisoned()) return false;
        *rateOut = rate.candidatesPerSecond;
        return true;
    }

    bool allocate() {
        if (!fusedScanSelectionValid()) return false;
        const size_t k = kK, rank = kRank;
        const size_t aBytes = (size_t)shape_.m * k;
        const size_t bBytes = (size_t)shape_.n * k;
        const size_t chunks = (aBytes > bBytes ? aBytes : bBytes) / kChunkLen;
        compactReadback_ = proof_timing::enabled(getenv("CANARY_PEARL_COMPACT_READBACK"));
        if (compactReadback_ && (chunks > CompactProofPlan::maxLeaves ||
            aBytes / kChunkLen < 2 || bBytes / kChunkLen < 2)) return false;
        return alloc(&dA_, aBytes) && alloc(&dAn_, aBytes) &&
               alloc(&dBt_, bBytes) &&
               // One layout, not both: whichever the tuner settled on.
               (tileConfig(tile_).nMajorB ? alloc(&dBnT_, bBytes)
                                            : alloc(&dBn_, bBytes)) &&
               alloc(&dEAL_, (size_t)shape_.m * rank) &&
               alloc(&dEBRflat_, (size_t)shape_.n * rank) &&
               alloc(&dEBR_, (size_t)shape_.n * rank) &&
               alloc(&dArF_, k * 2) && alloc(&dArS_, k * 2) &&
               alloc(&dBlF_, k * 2) && alloc(&dBlS_, k * 2) &&
               alloc(&dScratch_, chunks * 32 * 2) && alloc(&dRoots_, 64) &&
               alloc(&dCommitA_, 32) && alloc(&dCommitB_, 32) &&
               alloc(&dJobKey_, 32) && alloc(&dSaltA_, 32) &&
               alloc(&dSaltB_, 32) && alloc(&dSeedA_, 32) &&
               alloc(&dSeedB_, 32) && alloc(&dTarget_, 32) &&
               alloc(&dTranscripts_, (size_t)tiles() * 16 * sizeof(uint32_t)) &&
               alloc(&dHitIndex_, (kMaxHits + kFusedScanCanarySlots) * 4) &&
               alloc(&dHitDigest_, (kMaxHits + kFusedScanCanarySlots) * 32) && alloc(&dHitCount_, 4) &&
               initializeFusedScanCanaries() &&
               (!tma_fused::fused(fusedMode_) || alloc(&dHitTranscripts_,kMaxHits*64)) &&
               (!compactReadback_ ||
                (alloc(&dProofTreeA_, (aBytes / kChunkLen * 2) * 32) &&
                 alloc(&dProofTreeB_, (bBytes / kChunkLen * 2) * 32) &&
                 alloc(&dProofSiblings_, CompactProofPlan::maxSiblings * sizeof(uint64_t)) &&
                 alloc(&dProofOpening_, size_t(kTileSide) * kK + CompactProofPlan::maxSiblings * 32))) &&
               allocateOverlap() && allocateIntegrated();
    }

    /** Per-job work: everything that depends on the header but not on A. */
    bool beginJob(const Job &job, bool internalTuning = false) {
        if (publication_.poisoned()) return false;
        PearlJobContext context;
        if (internalTuning) {
            if (!validatePearlJobContext(job, nullptr, &context, nullptr) ||
                context.network != PearlNetwork::SyntheticBenchmark) {
                publication_.abandonRetryable();
                return false;
            }
        } else if (!validateRunJob(job, &context)) {
            publication_.abandonRetryable();
            return false;
        }
        PreparedJobRequest request;
        PreparedJobHost next;
        const PreparedJobMode mode =
            context.network == PearlNetwork::SyntheticBenchmark
                ? PreparedJobMode::SyntheticHardTarget
                : PreparedJobMode::Normal;
        if (!buildCudaPreparedJobRequest(tileConfig(tile_), shape_.m,
                shape_.n, arch10_, mode, &request) ||
            !buildPreparedJobHost(job,
                internalTuning || syntheticRun_ ? nullptr : &sourcePolicy_,
                request, &next)) {
            publication_.abandonRetryable();
            wins_.beginSearch();
            return false;
        }
        if (publication_.matches(next)) return true;
        stage_timing::JobScope<JobStageAdapter> jobTiming(
            !internalTuning && exactStageTimingActive(), shape_.m, shape_.n, kK, kRank,
            arch10_, JobStageAdapter{});
        if (!discardPreparedLookahead()) return false;
        if (!publication_.beginRewrite()) return false;
        bProofCache_.clear(); // Invalidate before any per-job B/key rewrite.
        bReadbackValid_ = false;
        bCompactTreeValid_ = false;
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
        aRootEvidence_.invalidate();
#endif
        wins_.beginSearch();
        // Persist upload sources until the stream has completed. Publication
        // remains Dirty while this staging object/device fixture is rewritten.
        staging_ = next;
        cert_ = static_cast<int>(next.certificateVersion);
        cfg_ = next.config;
        bound_ = next.resolvedBound;
        memcpy(jobKey_, next.jobKeyBytes.data(), sizeof(jobKey_));
        jobSalt_ = next.jobSalt;

        cudaError_t setupErr = cudaMemcpyAsync(
            dJobKey_, jobKey_, 32, cudaMemcpyHostToDevice, stream_);
        if (setupErr == cudaSuccess)
            setupErr = cudaMemcpyAsync(dSaltA_, seedSaltA(), 32,
                                       cudaMemcpyHostToDevice, stream_);
        if (setupErr == cudaSuccess)
            setupErr = cudaMemcpyAsync(dSaltB_, seedSaltB(), 32,
                                       cudaMemcpyHostToDevice, stream_);
        if (setupErr == cudaSuccess)
            setupErr = cudaMemcpyAsync(dSeedA_, staging_.seedABytes.data(), 32,
                                       cudaMemcpyHostToDevice, stream_);
        if (setupErr == cudaSuccess)
            setupErr = cudaMemcpyAsync(dSeedB_, staging_.seedBBytes.data(), 32,
                                       cudaMemcpyHostToDevice, stream_);
        if (setupErr == cudaSuccess)
            setupErr = cudaMemcpyAsync(dTarget_, staging_.targetBytesLE.data(), 32,
                                       cudaMemcpyHostToDevice, stream_);
        if (setupErr != cudaSuccess) {
            fprintf(stderr, "[pearl-pow] per-job fixture upload failed: %s\n",
                    cudaGetErrorString(setupErr));
            // Do not overwrite accepted transfers' host sources on a retry.
            // A failed device operation requires a fresh algorithm instance.
            cudaStreamSynchronize(stream_);
            publication_.poison();
            return false;
        }

        // B, once. Roll a fresh one per job so a long run does not reuse the
        // same columns forever.
        btSeed_ = next.btSeed;
        const size_t bBytes = (size_t)shape_.n * kK;
        genMatrix<<<(bBytes / 8 + 255) / 256, 256, 0, stream_>>>(dBt_, bBytes,
                                                                 btSeed_);
        merkleRoot((const uint8_t *)dBt_, (uint32_t)(bBytes / kChunkLen),
                   dRoots_ + 32);
        deriveCommitmentB<<<1, 1, 0, stream_>>>(dRoots_ + 32, dJobKey_, dSaltB_,
                                                shape_.n, cert_ >= 3, dCommitB_);

        const uint32_t brCount = shape_.n * kRank;
        noiseUniform<<<((brCount + 31) / 32 + 255) / 256, 256, 0, stream_>>>(
            dEBRflat_, brCount, dCommitB_, dSeedB_, kNoiseMask, kNoiseShift);
        transposeNoiseB<<<(brCount + 255) / 256, 256, 0, stream_>>>(
            dEBRflat_, dEBR_, shape_.n, kRank);
        noisePerm<<<((kK + 7) / 8 + 255) / 256, 256, 0, stream_>>>(
            dBlF_, dBlS_, kK, dCommitB_, dSeedB_, kRank);
        // Only the layout the SELECTED kernel reads. tile_ is known by now -
        // prepare() tunes before any job arrives - and building both cost a
        // whole extra noising pass and a second full-size buffer (128 MB at
        // the mining shape) that nothing would ever read. Per job, so this is
        // job-switch latency and memory headroom, not steady hashrate.
        const size_t bElems = (size_t)kK * shape_.n;
        if (tileConfig(tile_).nMajorB)
            applyNoiseBt<<<(bElems + 255) / 256, 256, 0, stream_>>>(
                dBt_, dEBR_, dBlF_, dBlS_, dBnT_, shape_.n, kK);
        else
            applyNoiseB<<<(bElems + 255) / 256, 256, 0, stream_>>>(
                dBt_, dEBR_, dBlF_, dBlS_, dBn_, shape_.n, kK);

        const cudaError_t launchErr = cudaGetLastError();
        const cudaError_t syncErr = cudaStreamSynchronize(stream_);
        if (launchErr != cudaSuccess || syncErr != cudaSuccess) {
            fprintf(stderr, "[pearl-pow] per-job setup failed: %s\n",
                    cudaGetErrorString(launchErr != cudaSuccess ? launchErr
                                                              : syncErr));
            publication_.poison();
            return false;
        }
        const bool published = publication_.publish(next);
        if (published) jobTiming.complete();
        return published;
    }

    /** Per-attempt work: A's chain, the GEMM, and the PoW scan. */
    bool runAttempt(uint64_t nonce) {
        const cudaError_t resetErr =
            cudaMemsetAsync(dHitCount_, 0, sizeof(uint32_t), stream_);
        if (resetErr != cudaSuccess) {
            fprintf(stderr, "[pearl-pow] hit counter reset failed: %s\n",
                    cudaGetErrorString(resetErr));
            return false;
        }
        return prepareA(primaryA(), nonce, stream_, true) &&
               computeA(primaryA(), true);
    }

    ASlot primaryA() const noexcept { return {dA_, dAn_, dRoots_, dCommitA_}; }
    ASlot aSlot(unsigned slot) const noexcept { return slot ? alternateA_ : primaryA(); }

    // Scratch/noise/permutation buffers are shared only between preparations:
    // the prep stream is serial, and GEMM reads only the slot's noised A.
    bool prepareA(const ASlot &a, uint64_t nonce, cudaStream_t stream, bool profile,
                  unsigned threads = 256) {
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
        const unsigned evidenceSlot = a.raw == dA_ ? 0 : a.raw == alternateA_.raw ? 1 : 2;
        if (!aRootEvidence_.invalidateSlot(evidenceSlot)) return false;
#endif
        if (threads != 64 && threads != 256) return false;
        const size_t aBytes = (size_t)shape_.m * kK;
        const bool requestedFusion = rootFusionMode_ == ARootFusionMode::Enabled;
        if (requestedFusion && (!rootFusionEnvironmentValid() || !kARootFusionCompiled ||
            !aRootFusionPreparationValid(arch10_, shape_.m, shape_.n,
                static_cast<uint32_t>(aBytes / kChunkLen), threads, static_cast<unsigned>(cert_)))) return false;
        bool fusedEnqueued = false;
        if (profile) profiler_.boundary(0, stream);
        genMatrix<<<(aBytes / 8 + threads - 1) / threads, threads, 0, stream>>>(
            a.raw, aBytes, matrixSeed(nonce ^ jobSalt_, false));
        if (profile) profiler_.boundary(1, stream);
        if (requestedFusion) {
            merkleRootOn((const uint8_t *)a.raw, (uint32_t)(aBytes / kChunkLen), a.root,
                         stream, threads, a.commit, &fusedEnqueued);
            if (!fusedEnqueued || cudaGetLastError() != cudaSuccess) return false;
        } else {
            merkleRootOn((const uint8_t *)a.raw, (uint32_t)(aBytes / kChunkLen), a.root, stream, threads);
            if (profile) profiler_.boundary(2, stream);
            deriveCommitmentA<<<1, 1, 0, stream>>>(a.root, dCommitB_, dSaltA_,
                                                    shape_.m, cert_ >= 3, a.commit);
        }
        if (profile) profiler_.boundary(3, stream);

        const uint32_t alCount = shape_.m * kRank;
        noiseUniform<<<((alCount + 31) / 32 + threads - 1) / threads, threads, 0, stream>>>(
            dEAL_, alCount, a.commit, dSeedA_, kNoiseMask, kNoiseShift);
        noisePerm<<<((kK + 7) / 8 + threads - 1) / threads, threads, 0, stream>>>(
            dArF_, dArS_, kK, a.commit, dSeedA_, kRank);
        if (profile) profiler_.boundary(4, stream);
        if(tma_fused::combined(fusedMode_)) {
            if(tma_fused::parse(getenv("CANARY_PEARL_TMA_DATAFLOW"))!=fusedMode_ ||
               shape_.m!=16384 || shape_.n!=131072 || kK!=8192 || kRank!=128 ||
               ((!shapeOwnedGuard_ || !shapeBounded_) && !productTmaCombinedAllowed())) return false;
            applyNoiseAPacked<<<dim3(kK/1024,shape_.m/8),256,0,stream>>>(
                a.raw,dEAL_,dArF_,dArS_,a.noised);
        } else applyNoiseA<<<(aBytes + threads - 1) / threads, threads, 0, stream>>>(
            a.raw, dEAL_, dArF_, dArS_, a.noised, shape_.m, kK, kRank);
        if (profile) profiler_.boundary(5, stream);
        if (cudaGetLastError() != cudaSuccess) return false;
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
        if (!aRootEvidence_.prepared(evidenceSlot, nonce, fusedEnqueued)) return false;
#endif
        return true;
    }

    bool productTmaCombinedAllowed() const noexcept {
        return tma_fused::productModeAllowed(fusedMode_,pearlProductFixedSelectionActive(),true,
            tile_==kTmaDeliveryTile && pearlProductFixedSelectionMatches(
                getenv("CANARY_PEARL_SHAPE"),getenv("CANARY_PEARL_TILE")));
    }

    bool computeA(const ASlot &a, bool profile, bool *gemmEnqueued = nullptr) {
        if (gemmEnqueued) *gemmEnqueued = false;
        // Whichever tile configuration prepare() measured fastest on this
        // card. See kTileConfigs for why that is not a constant.
        const TileConfig &tc = tileConfig(tile_);
        dim3 grid(shape_.n / tc.blockN, shape_.m / tc.blockM);
        if (cudaGetLastError() != cudaSuccess) return false;
        if(tma_fused::parse(getenv("CANARY_PEARL_TMA_DATAFLOW"))!=fusedMode_) return false;
        if(tma_fused::fused(fusedMode_)) {
            if(tile_!=kTmaDeliveryTile || arch10_!=120 || shape_.m!=16384 ||
               shape_.n!=131072 || kK!=8192 || kRank!=128 ||
               !dHitTranscripts_ || ((!shapeOwnedGuard_ || !shapeBounded_) &&
                                     !productTmaCombinedAllowed())) return false;
            tma_delivery::ScanOutput scan{a.commit,reinterpret_cast<const uint32_t*>(dTarget_),
                dHitIndex_,dHitDigest_,dHitCount_,dHitTranscripts_,kMaxHits};
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
            scan.allTranscripts=batchCaptures_?dTranscripts_:nullptr;
#endif
            const cudaError_t error=tma_fused::combined(fusedMode_)?
                launchTmaCombined(stream_,a.noised,dBnT_,int(shape_.m),int(shape_.n),scan):tma_fused::asymmetric(fusedMode_)?
                launchTmaFused<true,true>(grid,stream_,a.noised,dBnT_,int(shape_.m),int(shape_.n),scan):
                launchTmaFused<true,false>(grid,stream_,a.noised,dBnT_,int(shape_.m),int(shape_.n),scan);
            if(error!=cudaSuccess) {
                fprintf(stderr,"[pearl-tma-scan] launchTmaFused failed: %s (%d)\n",cudaGetErrorString(error),int(error));
                return false;
            }
            ++fusedAttempts_;
            if(tma_fused::early(fusedMode_)) ++earlyAttempts_;
            if(profile) { profiler_.boundary(6,stream_); profiler_.boundary(7,stream_); }
            return true;
        }
        if (tc.launch(grid, tc.threads, stream_, a.noised, tc.nMajorB ? dBnT_ : dBn_,
                  dTranscripts_,
                  (int)shape_.m, (int)shape_.n, (int)kK, kRank) != cudaSuccess)
            return false;
        if (gemmEnqueued) *gemmEnqueued = true;
        if (profile) profiler_.boundary(6, stream_);
        powScan<<<(tiles() + 255) / 256, 256, 0, stream_>>>(
            dTranscripts_, tiles(), a.commit, (const uint32_t *)dTarget_,
            dHitIndex_, dHitDigest_, dHitCount_, kMaxHits);
        if (profile) profiler_.boundary(7, stream_);
        return cudaGetLastError() == cudaSuccess;
    }

    bool recoverFusedOverflow(const ASlot &a) {
        if(!tma_fused::needsRecovery(fusedMode_,hitReadback_,kMaxHits)) return true;
        // Same immutable inputs before completion credit or A-slot reuse.
        const uint32_t originallyObserved=hitReadback_;
        dim3 grid(shape_.n/256,shape_.m/128);
        if(cudaMemsetAsync(dHitCount_,0,sizeof(uint32_t),stream_)!=cudaSuccess ||
           launchTmaDelivery(grid,512,stream_,a.noised,dBnT_,dTranscripts_,
               int(shape_.m),int(shape_.n),kK,kRank)!=cudaSuccess) return false;
        powScan<<<(tiles()+255)/256,256,0,stream_>>>(dTranscripts_,tiles(),a.commit,
            reinterpret_cast<const uint32_t*>(dTarget_),dHitIndex_,dHitDigest_,dHitCount_,kMaxHits);
        if(cudaGetLastError()!=cudaSuccess ||
           cudaMemcpyAsync(&hitReadback_,dHitCount_,sizeof(uint32_t),cudaMemcpyDeviceToHost,stream_)!=cudaSuccess ||
           synchronizeStreamStatus()!=cudaSuccess || hitReadback_!=originallyObserved) return false;
        gatherTmaRetainedTranscripts<<<4,256,0,stream_>>>(dTranscripts_,dHitIndex_,dHitTranscripts_,kMaxHits);
        if(cudaGetLastError()!=cudaSuccess || synchronizeStreamStatus()!=cudaSuccess) return false;
        ++fusedRecoveries_;
        fprintf(stderr,"[pearl-tma-scan] overflow-recovery observed=%u retained=%u extra_work_credit=0\n",hitReadback_,kMaxHits);
        return true;
    }

    /**
     * Invert the transcript index, open the tile, and build the submission.
     *
     * The transcript layout is not row-major over tiles: noisy_gemm.cuh writes
     * them block-major over rank-sized blocks and then (hi, wi) inside each,
     * to match what the Python reference emits. Getting this inversion wrong
     * produces a proof that opens the wrong sixteen rows, which verifies
     * locally against nothing and is rejected by the node.
     */
    bool openWin(const Job &job, uint64_t nonce, const HitWindow &window,
                 std::vector<Solution> *out, const ASlot &a) {
        PearlFinalizerProgressScope finalizer(progressSink_);
        proof_timing::NativeScope timing(proofTimingOption(), nonce);
        timing.shape(shape_.m, shape_.n, kK, kRank);
        const uint32_t hits = window.retained;
        std::vector<uint32_t> index(hits);
        std::vector<uint32_t> digest(hits * 8);
        // search() has synchronized the attempt. Synchronous readbacks keep
        // these instance-owned destinations alive through completion on every branch.
        cudaError_t copyErr = cudaMemcpy(
            index.data(), dHitIndex_, hits * 4, cudaMemcpyDeviceToHost);
        if (copyErr == cudaSuccess)
            copyErr = cudaMemcpy(digest.data(), dHitDigest_, hits * 32,
                                 cudaMemcpyDeviceToHost);

        if (copyErr != cudaSuccess) return failDeviceSearch();
        // Only the first win is submitted. A second tile from the same attempt
        // is a second solution to the SAME block, so it can never be accepted
        // once the first is, and building its proof would be wasted work.
        // WHICH win, when an attempt has several: the LOWEST tile index, not
        // the first one the scan happened to record.
        //
        // powScan appends hits through an atomic counter, so the order is
        // whatever the hardware produced - it differs between two runs on one
        // card and between CUDA and Vulkan on the same job and nonce. Any of
        // them is a valid solution to the same block, so nothing was wrong;
        // but it made the miner's output nondeterministic, which means two
        // runs of one job cannot be diffed and the two backends cannot be
        // compared at all. Taking the minimum costs a scan of at most 64
        // entries, once per win, and makes the share a function of the job and
        // the nonce while the queue is not saturated.  On saturation CUDA's
        // atomic append decides which 64 wins are retained, so this is only
        // the lowest RETAINED tile, not necessarily the global lowest tile.
        // Every retained tile remains a valid proof candidate; the saturation
        // warning above makes the loss of global determinism explicit.
        uint32_t sel = 0;
        if (!selectLowestRetainedHit(index.data(), index.size(), window, &sel)) {
            fprintf(stderr,
                    "[pearl-pow] retained hit window does not own a complete "
                    "selection - not submitted\n");
            return false;
        }
        progress(PearlAlgorithmProgressEvent::CandidateSelected);
        const uint32_t flat = index[sel];
        const uint32_t tilesPerSide = kRank / kTileSide;
        const uint32_t blocksPerRow = shape_.n / kRank;
        const uint32_t wi = flat % tilesPerSide;
        const uint32_t hi = (flat / tilesPerSide) % tilesPerSide;
        const uint32_t block = flat / (tilesPerSide * tilesPerSide);
        const uint32_t jIdx = block % blocksPerRow;
        const uint32_t iIdx = block / blocksPerRow;
        const uint32_t tRow = iIdx * kRank + hi * kTileSide;
        const uint32_t tCol = jIdx * kRank + wi * kTileSide;

        MerkleProof aProof, btProof;
        bool bCacheHit = false;
        const size_t aBytes = (size_t)shape_.m * kK;
        const size_t bBytes = (size_t)shape_.n * kK;
        // A changes each attempt. B is immutable within a published job.
        // Opt-in B reuse becomes valid only after independent host root checks,
        // and is invalidated BEFORE any B/key rewrite or device release.
        auto exactSize = [](std::vector<int8_t> &buffer, size_t bytes) {
            if (buffer.size() != bytes) {
                std::vector<int8_t>().swap(buffer);
                buffer.resize(bytes);
            }
        };
        const size_t rowBytes = size_t(kTileSide) * kK;
        exactSize(proofReadbackA_, compactReadback_ ? rowBytes : aBytes);
        exactSize(proofReadbackB_, compactReadback_ ? rowBytes : bBytes);
        auto &A = proofReadbackA_;
        auto &Bt = proofReadbackB_;
        if (compactReadback_) {
            if (integratedSearchActive_) {
                if (!readIntegratedOpenings(job,nonce,a,tRow,tCol,flat,&aProof,&btProof)) return false;
            } else if (!readCompactOpening(reinterpret_cast<const uint8_t *>(a.raw), aBytes,
                    tRow, dProofTreeA_, false, &aProof, nonce) ||
                !readCompactOpening(reinterpret_cast<const uint8_t *>(dBt_), bBytes,
                    tCol, dProofTreeB_, bCompactTreeValid_, &btProof, nonce)) return false;
            bCacheHit = bCompactTreeValid_;
            if (aProof.leafData.size() != A.size() || btProof.leafData.size() != Bt.size())
                return failSearch();
            memcpy(A.data(), aProof.leafData.data(), A.size());
            memcpy(Bt.data(), btProof.leafData.data(), Bt.size());
            if (proof_timing::enabled(proofTimingOption()))
                fprintf(stderr, "[pearl-compact-readback] bytes=%llu b_tree_hit=%d\n",
                    static_cast<unsigned long long>(A.size()+Bt.size()+aProof.siblings.size()+btProof.siblings.size()),
                    bCacheHit ? 1 : 0);
        } else {
            copyErr = cudaMemcpy(A.data(), a.raw, aBytes, cudaMemcpyDeviceToHost);
            const bool reuseB = proof_timing::enabled(getenv("CANARY_PEARL_CACHE_B_READBACK")) && bReadbackValid_;
            if (copyErr == cudaSuccess && !reuseB)
                copyErr = cudaMemcpy(Bt.data(), dBt_, bBytes, cudaMemcpyDeviceToHost);
            if (proof_timing::enabled(proofTimingOption()))
                fprintf(stderr, "[pearl-b-readback] hit=%d bytes=%llu\n", reuseB ? 1 : 0,
                        static_cast<unsigned long long>(reuseB ? 0 : bBytes));
        }
        uint8_t commitA[32], commitB[32];
        if (integratedSearchActive_) {
            const auto &capsule = integratedCapsules_[integratedCurrentSlot_];
            if (!capsule.valid) return failSearch();
            memcpy(commitA,capsule.host+kIntegratedMetaOffset,32);
            memcpy(commitB,capsule.host+kIntegratedMetaOffset+32,32);
        } else if (copyErr == cudaSuccess)
            copyErr = cudaMemcpy(commitA, a.commit, 32,
                                 cudaMemcpyDeviceToHost);
        if (!integratedSearchActive_ && copyErr == cudaSuccess)
            copyErr = cudaMemcpy(commitB, dCommitB_, 32,
                                 cudaMemcpyDeviceToHost);
        if (copyErr != cudaSuccess) {
            fprintf(stderr, "[pearl-pow] winning-attempt readback failed: %s\n",
                    cudaGetErrorString(copyErr));
            return failDeviceSearch();
        }
        timing.boundary(); // readback; compact mode also constructs the openings

        // Keep the old diagnostic boundary. Compact mode owns only winning
        // rows; the reference retains one host buffer for each full matrix.
        timing.boundary(); // no duplicate full matrix allocation or copy

        Win win;
        win.nonce = nonce;
        win.bound = bound_;
        bindWinToJob(&win, job);
        uint32_t hostTr[kTranscriptWords] = {};
        win.verified = recheck(A, Bt, commitA, commitB, tRow, tCol, hostTr,
                               &win.digest, compactReadback_);
        if(tma_fused::fused(fusedMode_) && win.verified) {
            uint32_t retained[kTranscriptWords]; uint8_t expectedDigest[32];
            win.digest.toBytesLE(expectedDigest);
            if(cudaMemcpy(retained,dHitTranscripts_+size_t(sel)*kTranscriptWords,
                    sizeof(retained),cudaMemcpyDeviceToHost)!=cudaSuccess)
                return failDeviceSearch();
            if(std::memcmp(retained,hostTr,sizeof(retained)) ||
               std::memcmp(expectedDigest,digest.data()+size_t(sel)*8,32)) {
                fprintf(stderr,"[pearl-tma-scan] retained index/transcript/digest association mismatch\n");
                return failDeviceSearch();
            }
        }
        if (!win.verified && getenv("CANARY_PEARL_DEBUG")) {
            // Is the GPU's transcript for this tile the same as the one the
            // host just rebuilt from A, B and the noise? If they match, the
            // disagreement is in the digest or the bound; if they differ, the
            // strips or the tile index are wrong. Anything else is guessing.
            uint32_t gpuTr[kTranscriptWords] = {};
            if (integratedSearchActive_)
                memcpy(gpuTr,integratedCapsules_[integratedCurrentSlot_].host+kIntegratedMetaOffset+128,sizeof(gpuTr));
            else if (cudaMemcpy(gpuTr,
                           tma_fused::fused(fusedMode_) ? dHitTranscripts_+size_t(sel)*kTranscriptWords :
                           dTranscripts_ + (size_t)flat * kTranscriptWords,
                           sizeof(gpuTr), cudaMemcpyDeviceToHost) != cudaSuccess)
                return failDeviceSearch();
            bool same = true;
            for (int w = 0; w < (int)kTranscriptWords; w++)
                if (gpuTr[w] != hostTr[w]) same = false;
            fprintf(stderr,
                    "[pearl-dbg] flat=%u tRow=%u tCol=%u shape=%ux%u "
                    "transcript %s\n", flat, tRow, tCol, shape_.m, shape_.n,
                    same ? "MATCHES (digest/bound differ)" : "DIFFERS");
            if (!same) {
                fprintf(stderr, "  gpu :");
                for (int w = 0; w < (int)kTranscriptWords; w++)
                    fprintf(stderr, " %08x", gpuTr[w]);
                fprintf(stderr, "\n  host:");
                for (int w = 0; w < (int)kTranscriptWords; w++)
                    fprintf(stderr, " %08x", hostTr[w]);
                fprintf(stderr, "\n");
            }
        }
        progress(win.verified ? PearlAlgorithmProgressEvent::HostValid
                              : PearlAlgorithmProgressEvent::HostInvalid);
        // No cap. run.cpp calls verify() once per solution returned by THIS
        // search(), so the list only has to cover this call - and search()
        // clears it. Capping it at 32 while a call can run 64 attempts meant
        // verify() could not find the nonce and reported a perfectly good
        // solution as a host-verification failure, which then went
        // unsubmitted. It scaled with hashrate, because a faster kernel gets
        // more attempts into the same batch: 50% of solutions thrown away at
        // 1024x16384, 8% at 4096x32768, none at all on the slowest kernel.
        if (!win.verified) {
            // HostInvalid is terminal. Disarm before any ledger/output
            // allocation so exception unwinding cannot publish a second drop.
            finalizer.close();
            // Reported, not submitted: the core prints it as a host rejection.
            wins_.record(win);
            Solution sol;
            sol.nonce = nonce;
            out->push_back(sol);
            return true;
        }
        timing.boundary(); // selection and host recheck

        if (!compactReadback_ && !buildProof(reinterpret_cast<const uint8_t *>(A.data()), A.size(), jobKey_,
                        leafIndicesFromRows(tRow, kTileSide, kK), &aProof, 4)) {
            finalizer.drop();
            return false;
        }
        timing.boundary(); // reference A tree (compact path built during readback)
        if (!compactReadback_ && !bProofCache_.build(reinterpret_cast<const uint8_t *>(Bt.data()), Bt.size(), jobKey_,
                        leafIndicesFromRows(tCol, kTileSide, kK), &btProof, 4, &bCacheHit)) {
            finalizer.drop();
            return false;
        }
        timing.boundary(); // reference B tree/cache (compact path built during readback)
        if (proofTimingOption())
            fprintf(stderr, "[pearl-b-cache] hit=%d\n", bCacheHit ? 1 : 0);
        progress(PearlAlgorithmProgressEvent::FinalizerProgress);

        // The one link in the chain nothing else can catch.
        //
        // recheck() and digestFromProof() both take commitA/commitB from the
        // DEVICE, so a wrong Merkle root makes the noise wrong, the transcript
        // wrong and the digest wrong all together - and every local check
        // agrees, because they are all downstream of the same bad value. The
        // pool is the first party to recompute the root from the opened leaves,
        // and when it disagrees the only thing it can say is "hash does not
        // meet difficulty target".
        //
        // Reference mode hashes the full matrices on the host (with B cache).
        // Compact mode rehashes opened chunks and reconstructs their paths on
        // the host; the device supplies only unopened sibling CVs. Both modes
        // compare reconstructed roots on every proof. The independent official
        // verifier remains a separate requirement before any pool transmission.
        uint8_t devRoots[64];
        if (integratedSearchActive_)
            memcpy(devRoots,integratedCapsules_[integratedCurrentSlot_].host+kIntegratedMetaOffset+64,sizeof(devRoots));
        else if (cudaMemcpy(devRoots, a.root, 32, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(devRoots + 32, dRoots_ + 32, 32,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            finalizer.drop();
            return failDeviceSearch();
        }
        const bool aRootOk = memcmp(devRoots, aProof.root, 32) == 0;
        const bool bRootOk = memcmp(devRoots + 32, btProof.root, 32) == 0;
        if (!aRootOk || !bRootOk) {
            bReadbackValid_ = false;
            bCompactTreeValid_ = false;
            bProofCache_.clear();
            fprintf(stderr,
                    "[pearl-pow] the device's Merkle root for %s disagrees with "
                    "the host's over the same bytes - the commitments this proof "
                    "was mined under are not the ones it opens, so it is not "
                    "submitted. This is a GPU-side defect, not a pool problem.\n",
                    !aRootOk ? (bRootOk ? "A" : "A and B") : "B");
            finalizer.drop(PearlProgressReason::CandidateOwnershipLost);
            return true;
        }
        if (!compactReadback_) bReadbackValid_ = true;
        progress(PearlAlgorithmProgressEvent::FinalizerProgress);

        // The second half of the same blind spot. Everything downstream takes
        // commitA/commitB FROM THE DEVICE - recheck() and digestFromProof()
        // both - so a wrong derivation is invisible to every local check and
        // visible to the pool, which recomputes it from the root the proof
        // opens. Two blake3 blocks on the host, once per win.
        uint8_t hostCommitA[32], hostCommitB[32];
        commitments(aProof.root, btProof.root, jobKey_, shape_.m, shape_.n,
                    cert_ >= 3, hostCommitA, hostCommitB);
        if (memcmp(hostCommitA, commitA, 32) != 0 ||
            memcmp(hostCommitB, commitB, 32) != 0) {
            fprintf(stderr,
                    "[pearl-pow] the device's commitments disagree with the "
                    "host's over the same Merkle roots - this proof is bound to "
                    "a key the pool will not derive, so it is not submitted. "
                    "This is a GPU-side defect, not a pool problem.\n");
            finalizer.drop(PearlProgressReason::CandidateOwnershipLost);
            return true;
        }
        progress(PearlAlgorithmProgressEvent::FinalizerProgress);

        if (compactReadback_) bCompactTreeValid_ = true;

        // The last thing that can still be wrong is the proof itself. recheck()
        // agreed with the device because it was handed the device's own
        // buffers; this reads back only what was serialised, which is what the
        // pool reconstructs from. If they disagree the proof does not describe
        // the tile that won, and submitting it wastes a share and earns
        // "hash does not meet difficulty target".
        U256 proofDigest;
        if (!digestFromProof(aProof, tRow, btProof, tCol, commitA, commitB,
                             shape_.m, shape_.n, kK, kRank, kTileSide,
                             &proofDigest)) {
            fprintf(stderr,
                    "[pearl-pow] the proof does not open the rows this tile "
                    "needs - not submitted\n");
            finalizer.drop(PearlProgressReason::ProofSerializationFailed);
            return true;
        }
        if (memcmp(proofDigest.v, win.digest.v, sizeof(proofDigest.v)) != 0) {
            fprintf(stderr,
                    "[pearl-pow] proof digest disagrees with the mined digest "
                    "at tile (%u,%u) - not submitted\n",
                    tRow, tCol);
            finalizer.drop(PearlProgressReason::ProofSerializationFailed);
            return true;
        }
        progress(PearlAlgorithmProgressEvent::FinalizerProgress);

        Solution sol;
        sol.nonce = nonce;
        // The digest of the tile we SELECTED, not of hit 0. These were the
        // same expression while the selection was always hit 0, and taking the
        // lowest index without moving this made the miner report one tile's
        // digest for another tile's proof. verify() caught it immediately,
        // which is exactly what it is for.
        for (int i = 0; i < 4; i++)
            sol.hit[i] = (uint64_t)digest[sel * 8 + i * 2] |
                         ((uint64_t)digest[sel * 8 + i * 2 + 1] << 32);
        DenseProofEncodeResult encoded = encodePlainProofChecked(
            shape_.m, shape_.n, kK, kRank, aProof, tRow, kTileSide,
            btProof, tCol, kTileSide);
        if (!encoded.ok()) {
            fprintf(stderr,
                    "[pearl-pow] checked proof serialization refused the "
                    "candidate at tile (%u,%u): %s - not submitted\n",
                    tRow, tCol, denseProofEncodeErrorName(encoded.error));
            finalizer.drop(PearlProgressReason::ProofSerializationFailed);
            return true;
        }
        try {
            sol.extra = base64(encoded.bytes);
            win.proof = sol.extra;
            // Do output growth before binding. A later ledger-record failure
            // still fails the whole search; the run owner invalidates any
            // binding already made, never retroactively publishes its proof.
            if (out->size() == out->max_size()) {
                finalizer.drop(PearlProgressReason::ProofSerializationFailed);
                return failSearch();
            }
            out->reserve(out->size() + 1);
            if (proofBindingSink_ &&
                (!proofBindingSink_->bindSerializedProof(job, &sol) ||
                 sol.pearlProofToken == 0 || sol.nonce != nonce ||
                 sol.extra != win.proof ||
                 memcmp(sol.hit, win.digest.v, sizeof(sol.hit)) != 0)) {
                finalizer.drop(PearlProgressReason::CandidateOwnershipLost);
                return failSearch();
            }
            // Only records paired with an emitted Solution authorize verify.
            timing.proofToken(sol.pearlProofToken);
            wins_.record(win);
            out->push_back(std::move(sol));
        } catch (...) {
            finalizer.drop(PearlProgressReason::ProofSerializationFailed);
            return failSearch();
        }
        progress(PearlAlgorithmProgressEvent::ProofSerialized);
        finalizer.close();
        timing.boundary(); // root/commitment/proof checks, encoding and binding
        timing.completeOnExit();
        return true;
    }

    /** The node's own check, on the host, from the two matrices. */
    bool recheck(const std::vector<int8_t> &A, const std::vector<int8_t> &Bt,
                 const uint8_t commitA[32], const uint8_t commitB[32],
                 uint32_t tRow, uint32_t tCol,
                 uint32_t *trOut = nullptr, U256 *digestOut = nullptr,
                 bool selectedRows = false) const {
        SelectedNoise noise;
        if (!noise.generate(commitA, commitB, shape_.m, shape_.n, kK, kRank,
                            tRow, tCol, kTileSide)) return false;

        // Only the winning strips, not the whole product: sixteen rows of
        // A_noised and sixteen columns of B_noised.
        std::vector<int8_t> aStrip((size_t)kTileSide * kK);
        std::vector<int8_t> bStrip((size_t)kK * kTileSide);
        for (int i = 0; i < kTileSide; i++)
            for (size_t col = 0; col < kK; col++) {
                const int32_t e =
                    (int32_t)noise.aRows[(size_t)i * kRank + noise.ar.first[col]] -
                    (int32_t)noise.aRows[(size_t)i * kRank + noise.ar.second[col]];
                aStrip[(size_t)i * kK + col] =
                    (int8_t)(A[(size_t)((selectedRows ? 0 : tRow) + i) * kK + col] + e);
            }
        for (size_t p = 0; p < kK; p++)
            for (int j = 0; j < kTileSide; j++) {
                const int32_t e =
                    (int32_t)noise.btRows[(size_t)j * kRank + noise.bl.first[p]] -
                    (int32_t)noise.btRows[(size_t)j * kRank + noise.bl.second[p]];
                bStrip[p * kTileSide + j] =
                    (int8_t)(Bt[(size_t)((selectedRows ? 0 : tCol) + j) * kK + p] + e);
            }

        uint32_t transcript[kTranscriptWords];
        tileTranscript(aStrip.data(), bStrip.data(), kK, kTileSide, kRank, 0, 0,
                       kTileSide, kTileSide, transcript);
        if (trOut)
            for (uint32_t w = 0; w < kTranscriptWords; w++) trOut[w] = transcript[w];
        uint8_t d[32];
        powDigest(transcript, commitA, d);
        const U256 digest = U256::fromBytesLE(d);
        if (digestOut) *digestOut = digest;
        return digest.le(bound_);
    }

    // Executed by the existing serial device owner after the winning attempt
    // completes. Separate trees cannot alias next-A preparation scratch or the
    // alternate A slot. B reuse is published only after host root/commit checks.
    bool readCompactOpening(const uint8_t *data, size_t bytes, uint32_t firstRow,
                            uint8_t *tree, bool reuseTree, MerkleProof *out, uint64_t nonce) {
        const proof_timing::SystemAdapter clock;
        const auto started = clock.now();
        const auto leaves = leafIndicesFromRows(firstRow, kTileSide, kK);
        CompactProofPlan plan;
        if (leaves.empty() || !compactProofPlan(bytes / kChunkLen, leaves.front(), leaves.size(), &plan) ||
            plan.leafBytes() != size_t(kTileSide)*kK) return failSearch();
        std::vector<uint8_t> opening(plan.openingBytes());
        cudaError_t error = cudaSuccess;
        const uint32_t chunks = static_cast<uint32_t>(plan.total);
        if (!reuseTree) {
            chunkCvs<<<(chunks+255)/256,256,0,stream_>>>(data,chunks,
                reinterpret_cast<const uint32_t *>(dJobKey_),tree);
            error=cudaGetLastError();
            size_t offset=0;
            for (uint32_t width=chunks; error==cudaSuccess && width>2; width/=2) {
                reduceProofLevel<<<((width/2)+255)/256,256,0,stream_>>>(tree+offset*32,width/2,
                    reinterpret_cast<const uint32_t *>(dJobKey_),tree+(offset+width)*32);
                error=cudaGetLastError();offset+=width;
            }
        }
        if (error==cudaSuccess && !plan.siblingNodes.empty())
            error=cudaMemcpy(dProofSiblings_,plan.siblingNodes.data(),plan.siblingNodes.size()*sizeof(uint64_t),cudaMemcpyHostToDevice);
        if (error==cudaSuccess) {
            gatherProofOpening<<<(plan.openingBytes()+255)/256,256,0,stream_>>>(data,tree,
                plan.first*kChunkLen,static_cast<uint32_t>(plan.leafBytes()),dProofSiblings_,
                static_cast<uint32_t>(plan.siblingNodes.size()),dProofOpening_);
            error=cudaGetLastError();
        }
        // Drain even on launch/copy failure before any temporary host data dies.
        const cudaError_t drained=cudaStreamSynchronize(stream_);
        if (error==cudaSuccess) error=drained;
        const auto prepared = clock.now();
        if (error==cudaSuccess)
            error=cudaMemcpy(opening.data(),dProofOpening_,opening.size(),cudaMemcpyDeviceToHost);
        if (error!=cudaSuccess) {
            fprintf(stderr,"[pearl-pow] compact opening failed: %s\n",cudaGetErrorString(error));
            return failDeviceSearch();
        }
        const auto copied = clock.now();
        if (!reconstructCompactProof(plan,opening.data(),opening.size(),jobKey_,out))
            return failSearch();
        const auto checked = clock.now();
        if (proof_timing::enabled(proofTimingOption()))
            fprintf(stderr,"[pearl-compact-stage] v=1 nonce=%016llx matrix=%c tree_hit=%d bytes=%llu "
                "prepare_ns=%llu copy_ns=%llu host_paths_ns=%llu total_ns=%llu\n",
                static_cast<unsigned long long>(nonce),data==reinterpret_cast<const uint8_t *>(dBt_) ? 'B' : 'A',reuseTree ? 1 : 0,
                static_cast<unsigned long long>(opening.size()),static_cast<unsigned long long>(prepared-started),
                static_cast<unsigned long long>(copied-prepared),static_cast<unsigned long long>(checked-copied),
                static_cast<unsigned long long>(checked-started));
        return true;
    }

    /** Chunk CVs, then as many tree levels per launch as a block can hold. */
    void merkleRoot(const uint8_t *data, uint32_t chunks, uint8_t *root) {
        merkleRootOn(data, chunks, root, stream_);
    }
    void merkleRootOn(const uint8_t *data, uint32_t chunks, uint8_t *root,
                      cudaStream_t stream, unsigned cap = 256,
                      uint8_t *aCommit = nullptr, bool *fusedEnqueued = nullptr) {
        chunkCvs<<<(chunks + cap - 1) / cap, cap, 0, stream>>>(
            data, chunks, (const uint32_t *)dJobKey_, dScratch_);
        uint32_t count = chunks;
        const uint8_t *in = dScratch_;
        uint8_t *outBuf = dScratch_ + (size_t)chunks * 32;
        while (count > 2) {
            const uint32_t threads = attemptMerkleReductionThreads(count, cap);
            const uint32_t blocks = count / (2 * threads);
            const size_t shared = (size_t)threads * 2 * 8 * sizeof(uint32_t);
            reduceTree<<<blocks, threads, shared, stream>>>(
                in, count, (const uint32_t *)dJobKey_, outBuf);
            count = blocks;
            const uint8_t *next = outBuf;
            outBuf = const_cast<uint8_t *>(in);
            in = next;
        }
#if defined(CANARY_PEARL_ENABLE_A_ROOT_FUSION)
        if (aCommit) {
            if (!fusedEnqueued || count != 2) return;
            rootCommitmentAFusedTest<<<1, 1, 0, stream>>>(in, (const uint32_t *)dJobKey_,
                dCommitB_, dSaltA_, shape_.m, static_cast<uint32_t>(cert_), root, aCommit);
            *fusedEnqueued = true; // Actual launch branch; retirement requires successful completion.
            return;
        }
#else
        (void)aCommit; (void)fusedEnqueued;
#endif
        rootCv<<<1, 1, 0, stream>>>(in, (const uint32_t *)dJobKey_, root);
    }

    static constexpr uint32_t kK = kCudaCommonDim;
    static constexpr int kRank = 128;
    static constexpr int kNoiseMask = 63, kNoiseShift = 32;

    Shape shape_ = kShapes[sizeof(kShapes) / sizeof(kShapes[0]) - 1];
    int arch10_ = 0;           // compute capability x10, 0 until queried
    PreparationCancellation preparationCancellation_;
    std::string tuningKey_;    // exact selected CUDA device UUID bytes
    bool exactPinnedConfiguration_ = false;
    FormalBenchmarkRequest formalRequest_;
    bool executionAccountingEnabled() const noexcept {return pearlPrivateEnabled(privateExecution_) || pearlPublicEnabled(publicExecution_);}
    PearlPrivateExecution executionAccountingMode() const noexcept {return pearlPublicEnabled(publicExecution_)?pearlPublicAccountingMode(publicExecution_):privateExecution_;}
    const PearlPublicExecution publicExecution_=pearlParsePublicExecution(getenv("CANARY_PEARL_NATIVE_EVIDENCE"),getenv("CANARY_PEARL_PUBLIC_EXECUTION"));
    const PearlPrivateExecution privateExecution_=pearlParsePrivateExecution(
        getenv("CANARY_PEARL_PRIVATE_EXECUTION"),getenv("CANARY_PEARL_PRIVATE_EXECUTION_MODE"));
    const tma_fused::Mode fusedMode_=tma_fused::parse(getenv("CANARY_PEARL_TMA_DATAFLOW"));
    uint64_t earlyAttempts_=0,fusedAttempts_=0,fusedRecoveries_=0;
    uint32_t* dHitTranscripts_=nullptr;
    PearlPrivateExecutionStats privateStats_;
    bool privateSearchCacheHit_=false;
    const CudaFixedExperimentSelection experimentSelection_ =
        parseCudaFixedExperiment(getenv("CANARY_PEARL_EXPERIMENT"));
    const AttemptOverlapMode overlapMode_ =
        parseAttemptOverlap(getenv("CANARY_PEARL_ATTEMPT_OVERLAP"));
    const ARootFusionMode rootFusionMode_ = parseARootFusion(getenv("CANARY_PEARL_A_ROOT_FUSION"));
    const bool rootFusionProfilingCaptured_ = getenv("CANARY_PEARL_PROFILE_INTERVAL") != nullptr;
    const bool fixedWm8ConflictCaptured_ =
        (experimentSelection_ == CudaFixedExperimentSelection::Wm8FixedK2048R128 ||
         experimentSelection_ == CudaFixedExperimentSelection::Bk32FixedK2048R128 ||
         experimentSelection_ == CudaFixedExperimentSelection::Wm4ReuseFixedK2048R128 ||
         experimentSelection_ == CudaFixedExperimentSelection::TmaWm8FixedK2048R128 ||
         experimentSelection_ == CudaFixedExperimentSelection::TmaTraversal16x32 ||
         experimentSelection_ == CudaFixedExperimentSelection::TmaTraversal16x32Stage4) &&
        (getenv("CANARY_PEARL_ATTEMPT_OVERLAP") || getenv("CANARY_PEARL_A_ROOT_FUSION") ||
        getenv("CANARY_PEARL_PROFILE_INTERVAL") || getenv("CANARY_PEARL_SHAPE") ||
        getenv("CANARY_PEARL_TILE") || getenv("CANARY_PEARL_HARD_TARGET") ||
        getenv("CANARY_PEARL_BENCH_ALL_HIT"));
    bool experimentConfigurationValid_ = false;
    BenchmarkIdentity identity_;
    mutable PearlNativeSelectionState nativeSelection_;
    mutable PearlShapeExperimentBinding shapeExperiment_{
        getenv("CANARY_PEARL_SHAPE_EXPERIMENT"), getenv("CANARY_PEARL_SHAPE"), getenv("CANARY_PEARL_TILE")};
    bool shapeConfigurationValid_ = false, shapeBounded_ = false, shapeOwnedGuard_ = false;
    size_t formalTile_ = 0;
    const char *tileFilter_ = getenv("CANARY_PEARL_TILE");
    size_t tile_ = 0;          // index into kTileConfigs, chosen by measurement
    bool allocated_ = false;
    bool configured_ = false;
    bool syntheticRun_ = false;
    PearlSourcePolicy sourcePolicy_;
    JobPublication publication_;
    PreparedJobHost staging_;
    // Owned by this device instance; release() discards both host allocations.
    std::vector<int8_t> proofReadbackA_, proofReadbackB_;
    MerkleProofCache bProofCache_;
    bool bReadbackValid_ = false;
    bool compactReadback_ = false, bCompactTreeValid_ = false;
    uint8_t *dProofTreeA_ = nullptr, *dProofTreeB_ = nullptr, *dProofOpening_ = nullptr;
    uint64_t *dProofSiblings_ = nullptr;
    int cert_ = 3;
    MiningConfig cfg_;
    U256 bound_;
    uint8_t jobKey_[32] = {};
    uint64_t jobSalt_ = 0, btSeed_ = 0;
    uint64_t completedAttempts_ = 0;
    uint32_t hitReadback_ = 0;
    PearlAlgorithmProgressSink *progressSink_ = nullptr;
    PearlProofBindingSink *proofBindingSink_ = nullptr;
    ProofOwnershipLedger<Win> wins_;
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
    std::vector<PearlTranscriptSnapshot> *batchCaptures_ = nullptr;
    bool singleCaptureActive_ = false;
    PearlRetirementStats retirementStats_;
    bool retirementStatsValid_ = true;
    unsigned testPreparedThreads_[2] = {};
    ARootPreparationEvidence aRootEvidence_;
#endif

    cudaStream_t stream_ = nullptr;
    cudaStream_t prepStream_ = nullptr;
    cudaEvent_t aReady_[2] = {nullptr, nullptr};
    ASlot alternateA_;
    bool overlapAnnounced_ = false;
    PreparedAttemptLookahead lookahead_;
    PreparedJobHost lookaheadPrepared_;
    Job lookaheadJob_;
    bool nextHintValid_ = false;
    uint64_t nextHintNonce_ = 0;
    stage_timing::Binding exactStageTiming_{getenv("CANARY_PEARL_STAGE_TIMING"),
                                           getenv("CANARY_PEARL_PROFILE_INTERVAL")};
    AttemptProfiler profiler_{exactStageTiming_.interval(getenv("CANARY_PEARL_PROFILE_INTERVAL")),
                              CudaProfileAdapter{exactStageTiming_.enabled()}};
    std::vector<void *> owned_;

    int8_t *dA_ = nullptr, *dAn_ = nullptr, *dBt_ = nullptr, *dBn_ = nullptr;
    int8_t *dBnT_ = nullptr;   // the same noised B, stored n-major
    int8_t *dEAL_ = nullptr, *dEBRflat_ = nullptr, *dEBR_ = nullptr;
    uint16_t *dArF_ = nullptr, *dArS_ = nullptr, *dBlF_ = nullptr, *dBlS_ = nullptr;
    uint8_t *dScratch_ = nullptr, *dRoots_ = nullptr;
    uint8_t *dCommitA_ = nullptr, *dCommitB_ = nullptr, *dJobKey_ = nullptr;
    uint8_t *dSaltA_ = nullptr, *dSaltB_ = nullptr;
    uint8_t *dSeedA_ = nullptr, *dSeedB_ = nullptr, *dTarget_ = nullptr;
    uint32_t *dTranscripts_ = nullptr, *dHitIndex_ = nullptr;
    uint32_t *dHitDigest_ = nullptr, *dHitCount_ = nullptr;
};

}  // namespace

Algorithm *makePearlPow() { return new PearlPow(); }
#if defined(CANARY_PEARL_MIXED_SCHEDULE_PROBE) && CANARY_PEARL_MIXED_SCHEDULE_PROBE == 1
int pearlMixedScheduleProbe() { return PearlPow::mixedScheduleProbe(); }
#endif
bool pearlCudaShapeExperimentCompiled() noexcept {
#if defined(CANARY_PEARL_ENABLE_SHAPE_EXPERIMENT) && CANARY_PEARL_ENABLE_SHAPE_EXPERIMENT == 1 && \
    defined(CANARY_PEARL_ONLY_BUILD) && CANARY_PEARL_ONLY_BUILD == 1
    return true;
#else
    return false;
#endif
}
bool pearlCudaFixedExperimentCompiled() noexcept { return kFixedExperimentCompiled; }
bool pearlCudaAttemptOverlapCompiled() noexcept { return kAttemptOverlapCompiled; }
bool pearlCudaIntegratedPipelineCompiled() noexcept { return kIntegratedPipelineCompiled; }
bool pearlCudaARootFusionCompiled() noexcept { return kARootFusionCompiled; }
bool pearlCudaExperimentCompiled(const char *selection) noexcept {
    return experimentCompiled(parseCudaFixedExperiment(selection));
}
#ifdef CANARY_PEARL_TEST_TRANSCRIPT_SNAPSHOT
bool pearlTestFusedScanSnapshot(Algorithm *algorithm,PearlFusedScanSnapshot *out) noexcept {
    auto *backend=dynamic_cast<PearlPow *>(algorithm);
    return backend && backend->testFusedScanSnapshot(out);
}
bool pearlTestFusedScanTargetProbe(Algorithm *algorithm,const std::array<uint32_t,8> &target,
        PearlFusedScanSnapshot *out) noexcept {
    auto *backend=dynamic_cast<PearlPow *>(algorithm);
    return backend && backend->testFusedScanTargetProbe(target,out);
}
bool pearlTestIntegratedPipelineStats(Algorithm *algorithm, PearlSpeculativeAttemptCounters *out) noexcept {
    auto *backend = dynamic_cast<PearlPow *>(algorithm);
    return backend && backend->testIntegratedPipelineStats(out);
}
bool pearlTestIntegratedPipelineTimings(Algorithm *algorithm, pearl::IntegratedPipelineHostTimings *out) noexcept {
    auto *backend = dynamic_cast<PearlPow *>(algorithm);
    return backend && backend->testIntegratedPipelineTimings(out);
}
Algorithm *pearlTestCreateSerialReplay(Algorithm *ordinaryOverlap,const Job &job) noexcept {
    auto *backend=dynamic_cast<PearlPow *>(ordinaryOverlap);
    return backend?backend->testCreateSerialReplay(job):nullptr;
}

bool pearlTestRetirementStats(Algorithm *algorithm, PearlRetirementStats *out) noexcept {
    if (!out) return false;
    *out = PearlRetirementStats{};
    auto *backend = dynamic_cast<PearlPow *>(algorithm);
    return backend && backend->testRetirementStats(out);
}

bool pearlTestSearchBatchAndCapture(Algorithm *algorithm, const Job &job,
                    uint64_t nonce, uint64_t attempts,
                    std::vector<PearlTranscriptSnapshot> *captures,
                    std::vector<Solution> *solutions) noexcept {
    auto *backend = dynamic_cast<PearlPow *>(algorithm);
    if (!backend) return false;
    return backend->testSearchBatchAndCapture(job, nonce, attempts, captures, solutions);
}

bool pearlTestSearchAndCapture(Algorithm *algorithm, const Job &job,
                              uint64_t nonce,
                              PearlTranscriptSnapshot *out) noexcept {
    auto *backend = dynamic_cast<PearlPow *>(algorithm);
    if (!backend) {
        if (out) *out = PearlTranscriptSnapshot{};
        return false;
    }
    return backend->testSearchAndCapture(job, nonce, out);
}
#endif

}  // namespace om
