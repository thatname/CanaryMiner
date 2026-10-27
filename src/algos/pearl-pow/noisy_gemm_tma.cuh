// Default-off independent TMA loader around Canary's MIT-licensed WM8 arithmetic.
// Sources: NVIDIA CUDA tensor-map API / PTX TMA and mbarrier documentation.
// No Akoya source is incorporated. Not production dispatch; fixed K2048/R128.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "noisy_gemm.cuh"
#include "cta_traversal.h"
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1200
#if !defined(CANARY_PEARL_AMPERE_BUILD)
#error "Canary TMA WM8 experiment requires native SM120"
#endif
#define CANARY_TMA_WM8_DEVICE_UNAVAILABLE 1
#endif
namespace om::pearl::tma_wm8 {
constexpr int StageBytes=64*(256+128), Stages=3;
constexpr int SharedBytes=Stages*StageBytes+3*8;
__device__ __forceinline__ unsigned sharedAddress(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__host__ __device__ constexpr int swizzle64(int offset) {
    return offset ^ ((offset >> 3) & 0x30);
}
__device__ __forceinline__ void issue(const CUtensorMap* a,const CUtensorMap* b,
    int8_t* shared,uint64_t* barrier,int stage,int kb,int row,int col) {
    unsigned bar=sharedAddress(barrier+stage);
    unsigned sa=sharedAddress(shared+stage*StageBytes);
    unsigned sb=sa+256*64;
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
template<bool Traversal16x32=false>
__global__ __launch_bounds__(512,1) void kernel(
    const __grid_constant__ CUtensorMap aMap,const __grid_constant__ CUtensorMap bMap,
    int32_t* c,uint32_t* transcript,int m,int n) {
#if defined(CANARY_TMA_WM8_DEVICE_UNAVAILABLE)
    __trap();
#else
    extern __shared__ __align__(1024) int8_t storage[];
    auto* barriers=reinterpret_cast<uint64_t*>(storage+Stages*StageBytes);
    int blocksN=n/128,blocksM=m/256,linear=blockIdx.x+blockIdx.y*blocksN;
    int rowBase,colBase;
    if constexpr(Traversal16x32) {
        cta_traversal::Tile tile;
        if(!cta_traversal::supertile<16,32>(linear,blocksM,blocksN,&tile)) return;
        rowBase=int(tile.row)*256;colBase=int(tile.column)*128;
    } else {
        int first=(linear/(8*blocksN))*8,groupRows=min(8,blocksM-first);
        int inside=linear%(8*blocksN);
        rowBase=(first+inside%groupRows)*256;colBase=(inside/groupRows)*128;
    }
    if(threadIdx.x==0) {
        for(int s=0;s<Stages;++s) {
            unsigned p=sharedAddress(barriers+s);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(p):"memory");
        }
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
    }
    __syncthreads();
    if(threadIdx.x==0) for(int s=0;s<Stages;++s)
        issue(&aMap,&bMap,storage,barriers,s,s,rowBase,colBase);
    int warp=threadIdx.x/32,lane=threadIdx.x%32,wm=warp/2,wn=warp%2;
    uint32_t acc[2][4][2][4]={},slots[2][4]={};
    int reduction=0;
    #pragma unroll 1
    for(int block=0;block<32;++block) {
        int stage=block%Stages;
        wait(barriers+stage,(block/Stages)&1);
        int8_t* sa=storage+stage*StageBytes;
        int8_t* sb=sa+256*64;
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
                ldmatrixX4(tmp,sb+swizzle64(col*64+sub*32+ldoff));
                br[j][0][0]=tmp[0];br[j][1][0]=tmp[1];
                br[j][0][1]=tmp[2];br[j][1][1]=tmp[3];
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
                    if(lane==reduction) slots[i][j]=v;
                }
            ++reduction;
        }
        // All consumers have finished the stage before its next TMA overwrite.
        __syncthreads();
        if(threadIdx.x==0 && block+Stages<32)
            issue(&aMap,&bMap,storage,barriers,stage,block+Stages,rowBase,colBase);
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
            if(lane<16) {
                int flat=(((row/128)*(n/128)+col/128)*8+(row%128)/16)*8+(col%128)/16;
                transcript[size_t(flat)*16+lane]=slots[i][j];
            }
        }
#endif
}
inline CUresult encode(CUtensorMap* map,void* source,int rows,int tileRows) {
    if(!map||!source||rows<=0||(reinterpret_cast<uintptr_t>(source)&15) ||
       (tileRows!=128&&tileRows!=256)) return CUDA_ERROR_INVALID_VALUE;
    cuuint64_t dims[2]={2048,static_cast<cuuint64_t>(rows)},strides[1]={2048};
    cuuint32_t box[2]={64,static_cast<cuuint32_t>(tileRows)},step[2]={1,1};
    return cuTensorMapEncodeTiled(map,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,source,
        dims,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
} // namespace om::pearl::tma_wm8
