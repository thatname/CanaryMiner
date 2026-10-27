// Isolated SM120-TMA-DELIVERY-01: retained K8192 TMA transport, current WM4/WN4.
// Sources: NVIDIA CUDA tensor-map API / PTX TMA and mbarrier documentation.
// No Akoya source is incorporated. Not production dispatch; fixed K8192/R128.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "noisy_gemm.cuh"
#include "blake3.cuh"
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1200
#if !defined(CANARY_PEARL_AMPERE_BUILD)
#error "Canary TMA WM8 experiment requires native SM120"
#endif
// Ampere build: the TMA kernels exist for the host dispatch, which refuses any device but 12.0 before launch. Their
// device bodies trap instead of carrying TMA/mbarrier PTX the target cannot assemble.
#define CANARY_TMA_DELIVERY_DEVICE_UNAVAILABLE 1
#endif
namespace om::pearl::tma_delivery {
constexpr int CommonDim=8192, Rank=128, Blocks=CommonDim/64;
static_assert(Blocks==128 && CommonDim/Rank==64, "exact K8192 fold schedule");
constexpr int BlockM=128, BlockN=256;
constexpr int StageBytes=64*(BlockM+BlockN), Stages=3;
constexpr int SharedBytes=Stages*StageBytes+3*8;
constexpr int AStageBytes=BlockM*64, BPairBytes=BlockN*128;
constexpr int AsymABytes=3*AStageBytes, AsymPayloadBytes=AsymABytes+2*BPairBytes;
constexpr int AsymSharedBytes=AsymPayloadBytes+5*8;
static_assert(AsymSharedBytes==90152 && AsymABytes%1024==0,"asymmetric layout budget");
// Optional diagnostic pointers are null in scored execution. The same fused
// kernel instantiation is exercised with and without them.
struct ScanOutput {
    const uint8_t* key=nullptr;
    const uint32_t* target=nullptr;
    uint32_t* indices=nullptr;
    uint32_t* digests=nullptr;
    uint32_t* count=nullptr;
    uint32_t* retainedTranscripts=nullptr;
    uint32_t capacity=0;
    uint32_t* allTranscripts=nullptr;
    uint32_t* allDigests=nullptr;
    uint32_t* allHits=nullptr;
    uint32_t* visits=nullptr;
};
__device__ __forceinline__ bool digestMeets(const uint32_t* digest,const uint32_t* target) {
    #pragma unroll
    for(int w=7;w>=0;--w) {
        if(digest[w]<target[w]) return true;
        if(digest[w]>target[w]) return false;
    }
    return true;
}
__device__ __forceinline__ void fusedScanTile(const uint32_t* words,uint32_t flat,ScanOutput out) {
    uint32_t block[16],cv[8];
    #pragma unroll
    for(int w=0;w<16;++w) block[w]=words[w];
    b3d::loadCv(out.key,cv);
    b3d::compress(cv,block,0,64u,
        b3d::kKeyedHash|b3d::kChunkStart|b3d::kChunkEnd|b3d::kRoot);
    const bool hit=digestMeets(cv,out.target);
    if(out.allTranscripts) {
        #pragma unroll
        for(int w=0;w<16;++w) out.allTranscripts[size_t(flat)*16+w]=block[w];
    }
    if(out.allDigests) {
        #pragma unroll
        for(int w=0;w<8;++w) out.allDigests[size_t(flat)*8+w]=cv[w];
    }
    if(out.allHits) out.allHits[flat]=hit?1u:0u;
    if(out.visits) atomicAdd(out.visits+flat,1u);
    if(!hit) return;
    const uint32_t slot=atomicAdd(out.count,1u);
    if(slot>=out.capacity) return; // host must recover overflow before credit
    out.indices[slot]=flat;
    #pragma unroll
    for(int w=0;w<8;++w) out.digests[size_t(slot)*8+w]=cv[w];
    #pragma unroll
    for(int w=0;w<16;++w) out.retainedTranscripts[size_t(slot)*16+w]=block[w];
}
__device__ __forceinline__ unsigned sharedAddress(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__host__ __device__ constexpr int swizzle64(int offset) {
    return offset ^ ((offset >> 3) & 0x30);
}
__host__ __device__ constexpr int swizzle128(int offset) {
    return offset ^ ((offset >> 3) & 0x70);
}
__device__ __forceinline__ void issueAsym(const CUtensorMap* map,int8_t* shared,
    uint64_t* barrier,int bytes,int k,int row) {
    unsigned bar=sharedAddress(barrier),dst=sharedAddress(shared);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        ::"r"(bar),"r"(bytes):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(dst),"l"(map),"r"(k),"r"(row),"r"(bar):"memory");
}
__device__ __forceinline__ void issue(const CUtensorMap* a,const CUtensorMap* b,
    int8_t* shared,uint64_t* barrier,int stage,int kb,int row,int col) {
    unsigned bar=sharedAddress(barrier+stage);
    unsigned sa=sharedAddress(shared+stage*StageBytes);
    unsigned sb=sa+BlockM*64;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        ::"r"(bar),"r"(StageBytes):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(sa),"l"(a),"r"(kb*64),"r"(row),"r"(bar):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(sb),"l"(b),"r"(kb*64),"r"(col),"r"(bar):"memory");
}
__device__ __forceinline__ void wait(uint64_t* barrier,int phase) {
    unsigned done=0,bar=sharedAddress(barrier);
    do {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
            :"=r"(done):"r"(bar),"r"(phase):"memory");
    } while(!done);
}
template<bool Fused=false,bool EarlyReuse=false,bool AsymmetricB=false>
__global__ __launch_bounds__(512,1) void kernel(
    const __grid_constant__ CUtensorMap aMap,const __grid_constant__ CUtensorMap bMap,
    int32_t* c,uint32_t* transcript,int m,int n,ScanOutput scan={}) {
#if defined(CANARY_TMA_DELIVERY_DEVICE_UNAVAILABLE)
    __trap();
#else
    extern __shared__ __align__(1024) int8_t storage[];
    static_assert(!AsymmetricB || (Fused && EarlyReuse),"candidate keeps v15 early/fused path");
    auto* barriers=reinterpret_cast<uint64_t*>(storage+(AsymmetricB?AsymPayloadBytes:Stages*StageBytes));
    int blocksN=n/BlockN,blocksM=m/BlockM,linear=blockIdx.x+blockIdx.y*blocksN;
    int first=(linear/(8*blocksN))*8,groupRows=min(8,blocksM-first);
    int inside=linear%(8*blocksN);
    int rowBase=(first+inside%groupRows)*BlockM,colBase=(inside/groupRows)*BlockN;
    if(threadIdx.x==0) {
        for(int s=0;s<(AsymmetricB?5:Stages);++s) {
            unsigned p=sharedAddress(barriers+s);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(p):"memory");
        }
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
    }
    __syncthreads();
    if(threadIdx.x==0) {
        if constexpr(AsymmetricB) {
            for(int s=0;s<3;++s)
                issueAsym(&aMap,storage+s*AStageBytes,barriers+s,AStageBytes,s*64,rowBase);
            for(int s=0;s<2;++s)
                issueAsym(&bMap,storage+AsymABytes+s*BPairBytes,barriers+3+s,BPairBytes,s*128,colBase);
        } else for(int s=0;s<Stages;++s)
            issue(&aMap,&bMap,storage,barriers,s,s,rowBase,colBase);
    }
    int warp=threadIdx.x/32,lane=threadIdx.x%32,wm=warp/4,wn=warp%4;
    uint32_t acc[2][4][2][4]={},slots[2][4]={};
    int reduction=0;
    #pragma unroll 1
    for(int block=0;block<Blocks;++block) {
        int stage=block%Stages;
        wait(barriers+stage,(block/Stages)&1);
        if constexpr(AsymmetricB) {
            // The second half uses the same completed pair; no phase advance.
            if((block&1)==0) wait(barriers+3+((block/2)%2),(block/4)&1);
        }
        int8_t* sa=storage+stage*(AsymmetricB?AStageBytes:StageBytes);
        int8_t* sb=AsymmetricB?storage+AsymABytes+((block/2)%2)*BPairBytes:sa+BlockM*64;
        int ldrow=(lane&7)+((lane>>3)&1)*8,ldoff=(lane>>4)*16;
        #pragma unroll
        for(int sub=0;sub<2;++sub) {
            uint32_t ar[2][4],br[4][2][2];
            #pragma unroll
            for(int i=0;i<2;++i) {
                int row=(wm*2+i)*16+ldrow;
                ldmatrixX4(ar[i],sa+swizzle64(row*64+sub*32+ldoff));
            }
            #pragma unroll
            for(int j=0;j<4;++j) {
                int col=(wn*4+j)*16+ldrow;
                uint32_t tmp[4];
                if constexpr(AsymmetricB)
                    ldmatrixX4(tmp,sb+swizzle128(col*128+(block&1)*64+sub*32+ldoff));
                else ldmatrixX4(tmp,sb+swizzle64(col*64+sub*32+ldoff));
                br[j][0][0]=tmp[0];br[j][1][0]=tmp[1];
                br[j][0][1]=tmp[2];br[j][1][1]=tmp[3];
            }
            if constexpr(EarlyReuse) {
                if(sub==1) {
                    // Every CTA consumer has loaded its final shared operands.
                    // The following MMA and rank fold use registers only.
                    __syncthreads();
                    if(threadIdx.x==0) {
                        if constexpr(AsymmetricB) {
                            if(block+3<Blocks)
                                issueAsym(&aMap,storage+stage*AStageBytes,barriers+stage,AStageBytes,(block+3)*64,rowBase);
                            // All readers of BOTH halves have loaded their last
                            // shared operands; the next pair may now overwrite.
                            if((block&1)==1 && block+3<Blocks)
                                issueAsym(&bMap,storage+AsymABytes+((block/2)%2)*BPairBytes,
                                    barriers+3+((block/2)%2),BPairBytes,(block/2+2)*128,colBase);
                        } else if(block+Stages<Blocks)
                            issue(&aMap,&bMap,storage,barriers,stage,block+Stages,rowBase,colBase);
                    }
                }
            }
            #pragma unroll
            for(int i=0;i<2;++i)
                #pragma unroll
                for(int j=0;j<4;++j)
                    #pragma unroll
                    for(int h=0;h<2;++h) mmaS8(acc[i][j][h],ar[i],br[j][h],acc[i][j][h]);
        }
        if((block&1)==1) {
            #pragma unroll
            for(int i=0;i<2;++i)
                #pragma unroll
                for(int j=0;j<4;++j) {
                    uint32_t v=warpXor(xor3(
                        xor3(acc[i][j][0][0],acc[i][j][0][1],acc[i][j][0][2]),
                        xor3(acc[i][j][0][3],acc[i][j][1][0],acc[i][j][1][1]),
                        acc[i][j][1][2]^acc[i][j][1][3]));
                    if(lane==(reduction&15))
                        slots[i][j]=rotl32(slots[i][j],13)^v;
                }
            ++reduction;
        }
        // All consumers have finished the stage before its next TMA overwrite.
        if constexpr(!EarlyReuse) {
            __syncthreads();
            if(threadIdx.x==0 && block+Stages<Blocks)
                issue(&aMap,&bMap,storage,barriers,stage,block+Stages,rowBase,colBase);
        }
    }
    #pragma unroll
    for(int i=0;i<2;++i)
        #pragma unroll
        for(int j=0;j<4;++j) {
            int row=rowBase+(wm*2+i)*16,col=colBase+(wn*4+j)*16;
            if(c) {
                #pragma unroll
                for(int h=0;h<2;++h) {
                    int rr=row+(lane>>2),cc=col+h*8+(lane&3)*2;
                    c[size_t(rr)*n+cc]=int32_t(acc[i][j][h][0]);
                    c[size_t(rr)*n+cc+1]=int32_t(acc[i][j][h][1]);
                    c[size_t(rr+8)*n+cc]=int32_t(acc[i][j][h][2]);
                    c[size_t(rr+8)*n+cc+1]=int32_t(acc[i][j][h][3]);
                }
            }
            if constexpr (!Fused) { if(lane<16) {
                int flat=(((row/128)*(n/128)+col/128)*8+(row%128)/16)*8+(col%128)/16;
                transcript[size_t(flat)*16+lane]=slots[i][j];
            }}
        }
    if constexpr (Fused) {
        // Async drain is proved by completion waits, NOT a CTA barrier alone:
        // every issued block is <Blocks and is consumed after wait(phase).
        // Blocks125/126/127 consume the last versions of A stages2/0/1; none
        // issues a successor. Asymmetric B pairs62/63 complete at blocks124/126
        // and their last readers are blocks125/127; neither issues another pair.
        // The final last-reader CTA barrier therefore
        // follows all three terminal copy completions in both schedules.
        // Fold and slots are register-only. Each warp now owns 512 scratch
        // bytes; all its transcript writers precede the syncwarp below.
        // Scratch never overlaps live mbarriers at Stages*StageBytes.
        static_assert(16*8*16*sizeof(uint32_t)<=Stages*StageBytes,"scratch excludes barriers");
        auto* scratch=reinterpret_cast<uint32_t*>(storage)+warp*8*16;
        if(lane<16) {
            #pragma unroll
            for(int i=0;i<2;++i)
                #pragma unroll
                for(int j=0;j<4;++j) scratch[(i*4+j)*16+lane]=slots[i][j];
        }
        __syncwarp();
        if(lane<8) {
            const int i=lane/4,j=lane%4;
            const int row=rowBase+(wm*2+i)*16,col=colBase+(wn*4+j)*16;
            const uint32_t flat=(((row/128)*(n/128)+col/128)*8+(row%128)/16)*8+(col%128)/16;
            fusedScanTile(scratch+lane*16,flat,scan);
        }
    }
#endif
}
inline CUresult encode(CUtensorMap* map,void* source,int rows,int tileRows,int tileK=64) {
    if(!map||!source||rows<=0||(reinterpret_cast<uintptr_t>(source)&15) ||
       (tileRows!=128&&tileRows!=256) || (tileK!=64 && !(tileK==128&&tileRows==256))) return CUDA_ERROR_INVALID_VALUE;
    cuuint64_t dims[2]={CommonDim,static_cast<cuuint64_t>(rows)},strides[1]={CommonDim};
    cuuint32_t box[2]={static_cast<cuuint32_t>(tileK),static_cast<cuuint32_t>(tileRows)},step[2]={1,1};
    void* entry=nullptr;
    cudaDriverEntryPointQueryResult status;
    if(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",&entry,12000,0,&status)!=cudaSuccess ||
       status!=cudaDriverEntryPointSuccess || !entry) return CUDA_ERROR_NOT_SUPPORTED;
    auto encodeTiled=reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(entry);
    return encodeTiled(map,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,source,
        dims,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,
        tileK==128?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
} // namespace om::pearl::tma_delivery
