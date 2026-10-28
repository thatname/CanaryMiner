#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace om::pearl {
// Exact rank128, eight rows per block; no raw/committed data mutation.
// Each thread owns four columns and reuses their indices across eight rows.
// kK is the row stride: K8192 for the SM120 combined path, K2048 on SM86.
template<int kK=8192>
__global__ void applyNoiseAPacked(const int8_t* raw,const int8_t* eAL,
    const uint16_t* first,const uint16_t* second,int8_t* output) {
    static_assert(kK%1024==0,"one block covers 1024 columns");
    __shared__ uint32_t noiseWords[8*128/4];
    const int rowBase=blockIdx.y*8,col=blockIdx.x*1024+threadIdx.x*4;
    noiseWords[threadIdx.x]=reinterpret_cast<const uint32_t*>(eAL+rowBase*128)[threadIdx.x];
    const ushort4 f=reinterpret_cast<const ushort4*>(first+col)[0];
    const ushort4 s=reinterpret_cast<const ushort4*>(second+col)[0];
    const uint16_t fi[4]={f.x,f.y,f.z,f.w},si[4]={s.x,s.y,s.z,s.w};
    __syncthreads();
    const auto* noise=reinterpret_cast<const int8_t*>(noiseWords);
    #pragma unroll
    for(int row=0;row<8;++row) {
        const size_t offset=size_t(rowBase+row)*kK+col;
        const uint32_t packed=reinterpret_cast<const uint32_t*>(raw+offset)[0];
        uint32_t result=0;
        #pragma unroll
        for(int byte=0;byte<4;++byte) {
            const int value=int(int8_t((packed>>(byte*8))&255))+
                int(noise[row*128+fi[byte]])-int(noise[row*128+si[byte]]);
            result|=uint32_t(uint8_t(value))<<(byte*8);
        }
        reinterpret_cast<uint32_t*>(output+offset)[0]=result;
    }
}
} // namespace om::pearl
