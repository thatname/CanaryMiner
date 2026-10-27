// Isolated combined-v1; the v16 control kernel remains byte-for-byte unchanged.
// Production K-loop uses B remapping and empty mbarriers between stages.
#pragma once
#include "noisy_gemm_tma_delivery.cuh"
namespace om::pearl::tma_delivery {
constexpr int CombinedBarrierBytes=8*10;
constexpr int CombinedScratchOffset=(AsymPayloadBytes+CombinedBarrierBytes+15)&~15;
constexpr int CombinedPitch=17, CombinedMessages=128;
constexpr int CombinedSharedBytes=CombinedScratchOffset+CombinedMessages*CombinedPitch*4;
static_assert(CombinedSharedBytes==98896,"operand/barrier/scratch budget");
__device__ __forceinline__ void combinedHash(const uint32_t* scratch,int rowBase,int colBase,int n,ScanOutput out) {
    // Hash lane ownership differs from MMA ownership: all lanes in four warps.
    const int message=threadIdx.x,owner=message/8,i=(message%8)/4,j=message%4;
    const int row=rowBase+((owner/4)*2+i)*16,col=colBase+((owner%4)*4+j)*16;
    const uint32_t flat=(((row/128)*(n/128)+col/128)*8+(row%128)/16)*8+(col%128)/16;
    fusedScanTile(scratch+message*CombinedPitch,flat,out);
}
__device__ __forceinline__ void mbarArrive(uint64_t* barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"(sharedAddress(barrier)):"memory");
}
__global__ __launch_bounds__(512,1) void combinedKernel(
    const __grid_constant__ CUtensorMap aMap,const __grid_constant__ CUtensorMap bMap,
    int m,int n,ScanOutput scan) {
#if defined(CANARY_TMA_DELIVERY_DEVICE_UNAVAILABLE)
    __trap();
#else
    extern __shared__ __align__(1024) int8_t storage[];
    auto* barriers=reinterpret_cast<uint64_t*>(storage+AsymPayloadBytes);
    uint64_t* emptyA=barriers+5;
    uint64_t* emptyB=barriers+8;
    auto* scratch=reinterpret_cast<uint32_t*>(storage+CombinedScratchOffset);
    const int blocksN=n/BlockN,blocksM=m/BlockM;
    const int warp=threadIdx.x/32,lane=threadIdx.x%32,wm=warp/4,wn=warp%4;
    int previousRow=0,previousCol=0;
    #pragma unroll 1
    for(int tile=0;tile<2;++tile) {
        // Flatten against ORIGINAL blocksN, never the half-sized launch grid.
        const int linear=int(blockIdx.x)*2+tile;
        const int first=(linear/(16*blocksN))*16,groupRows=min(16,blocksM-first);
        const int inside=linear%(16*blocksN);
        const int rowBase=(first+inside%groupRows)*BlockM,colBase=(inside/groupRows)*BlockN;
        if(threadIdx.x==128) {
            for(int s=0;s<5;++s) {
                const unsigned p=sharedAddress(barriers+s);
                // The previous tile's final waits consumed every issued copy;
                // publication below joins ALL consumers before invalidation.
                if(tile) asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(p):"memory");
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(p):"memory");
            }
            for(int s=5;s<10;++s) {
                const unsigned p=sharedAddress(barriers+s);
                if(tile) asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(p):"memory");
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 16;"::"r"(p):"memory");
            }
            asm volatile("fence.proxy.async.shared::cta;":::"memory");
        }
        __syncthreads();
        if(threadIdx.x==128) {
            for(int s=0;s<3;++s)
                issueAsym(&aMap,storage+s*AStageBytes,barriers+s,AStageBytes,s*64,rowBase);
            for(int s=0;s<2;++s)
                issueAsym(&bMap,storage+AsymABytes+s*BPairBytes,barriers+3+s,BPairBytes,s*128,colBase);
        }
        // Prefill is issued before splitting hash / compute. Scratch is wholly
        // disjoint from operands and barriers throughout the next tile.
        __syncthreads();
        // Earlier transcript workers enter K loop after releasing the stage reads to its producer.
        if(tile && threadIdx.x<CombinedMessages) combinedHash(scratch,previousRow,previousCol,n,scan);
        uint32_t acc[2][4][2][4]={},slots[2][4]={};
        int reduction=0;
        #pragma unroll 1
        for(int block=0;block<Blocks;++block) {
            const int stage=block%Stages;
            wait(barriers+stage,(block/Stages)&1);
            if((block&1)==0) wait(barriers+3+((block/2)%2),(block/4)&1);
            int8_t* sa=storage+stage*AStageBytes;
            int8_t* sb=storage+AsymABytes+((block/2)%2)*BPairBytes;
            const int ldrow=(lane&7)+((lane>>3)&1)*8,ldoff=(lane>>4)*16;
            const int ldrowB=(lane&7)+((lane>>4)&1)*8,ldoffB=((lane>>3)&1)*16;
            #pragma unroll
            for(int sub=0;sub<2;++sub) {
                uint32_t ar[2][4],br[4][2][2];
                #pragma unroll
                for(int i=0;i<2;++i) {
                    const int row=(wm*2+i)*16+ldrow;
                    ldmatrixX4(ar[i],sa+swizzle64(row*64+sub*32+ldoff));
                }
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const int col=(wn*4+j)*16+ldrowB;
                    uint32_t tmp[4];
                    ldmatrixX4(tmp,sb+swizzle128(col*128+(block&1)*64+sub*32+ldoffB));
                    br[j][0][0]=tmp[0];br[j][0][1]=tmp[1];
                    br[j][1][0]=tmp[2];br[j][1][1]=tmp[3];
                }
                if(sub==1) {
                    __syncwarp();
                    if(lane==0) {
                        mbarArrive(emptyA+stage);
                        if((block&1)==1) mbarArrive(emptyB+((block/2)%2));
                    }
                    if(threadIdx.x==128 && block+3<Blocks) {
                        wait(emptyA+stage,(block/Stages)&1);
                        issueAsym(&aMap,storage+stage*AStageBytes,barriers+stage,AStageBytes,(block+3)*64,rowBase);
                        if((block&1)==1) {
                            wait(emptyB+((block/2)%2),(block/4)&1);
                            issueAsym(&bMap,storage+AsymABytes+((block/2)%2)*BPairBytes,
                                barriers+3+((block/2)%2),BPairBytes,(block/2+2)*128,colBase);
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
                        const uint32_t v=warpXor(xor3(
                            xor3(acc[i][j][0][0],acc[i][j][0][1],acc[i][j][0][2]),
                            xor3(acc[i][j][0][3],acc[i][j][1][0],acc[i][j][1][1]),
                            acc[i][j][1][2]^acc[i][j][1][3]));
                        if(lane==(reduction&15)) slots[i][j]=rotl32(slots[i][j],13)^v;
                    }
                ++reduction;
            }
        }
        // Last-reader barrier at block127 follows all terminal TMA waits.
        // All hashes of the previous tile joined at block0's reader barrier.
        if(lane<16) {
            #pragma unroll
            for(int i=0;i<2;++i)
                #pragma unroll
                for(int j=0;j<4;++j)
                    scratch[(warp*8+i*4+j)*CombinedPitch+lane]=slots[i][j];
        }
        __syncthreads(); // publish all 128 messages and finish the old tile
        previousRow=rowBase;previousCol=colBase;
        if(tile==1 && threadIdx.x<CombinedMessages) combinedHash(scratch,rowBase,colBase,n,scan);
    }
#endif
}
} // namespace om::pearl::tma_delivery
