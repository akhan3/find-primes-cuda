#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include <defs.h>

__device__ __constant__ uint32  d_primeFactors[3401];   // pi(sqrt_ulimit) = 3401

__global__ void primeKernel(    const uint32 ulimit,
                                byte* g_all_primes,
                                const uint32 primeFactors_count,
                                const byte cook             )
{
    if(cook) return;

    // #ifdef __DEVICE_EMULATION__
    // if(blockIdx.x == 0 && threadIdx.x == 0)
        // for(uint32 i = 0; i < primeFactors_count; i++)
            // printf("%d: %d\n", i, d_primeFactors[i]);
    // __syncthreads();
    // #endif

    __shared__ uint32 primeFactors_index;
    __shared__ uint32 primeFactors_indexlast;
    //__shared__ uint32 thisFactor;
    uint32 thisFactor;

    if(threadIdx.x == 0) 
    {
        // initial 6 prime factors are not needed {2,3,5,7,11,13}
        uint32 chunk_size = (uint32) ceil((float)(primeFactors_count-6) / gridDim.x);
        // 6 is the index of 17 in d_primeFactors array
        primeFactors_index = 6 + blockIdx.x * chunk_size;
        primeFactors_indexlast = primeFactors_index + chunk_size - 1;
        if(primeFactors_indexlast >= primeFactors_count)    // take care of last chunk
            primeFactors_indexlast = primeFactors_count-1;
        if(primeFactors_index >= primeFactors_count)    // take care of redundant chunk
            primeFactors_index = primeFactors_count-1;
        #ifdef __DEVICE_EMULATION__
        if(blockIdx.x == 0 && threadIdx.x == 0) {
            printf("primeFactors_count=%d\n", primeFactors_count);
            printf("chunk_size=%d\n", chunk_size);
        }
        // printf("\nBlock%d: (%d, %d) => ", blockIdx.x, primeFactors_index, primeFactors_indexlast);
        printf("\nBlock%d: [", blockIdx.x);
        for(uint32 i = primeFactors_index; i <= primeFactors_indexlast; i++)
            printf("%d, ", d_primeFactors[i]);
        printf("]\n");
        #endif
        
    }
    __syncthreads();

    // main loop
    while(primeFactors_index <= primeFactors_indexlast) {
        thisFactor = d_primeFactors[primeFactors_index];
        #ifdef __DEVICE_EMULATION__
        if(threadIdx.x == 0)
            printf("  thisFactor = %u\n", thisFactor);
        __syncthreads();
        #endif

        uint32 thisMark = 2*thisFactor + threadIdx.x * thisFactor;
        while(thisMark <= ulimit) {
            // #ifdef __DEVICE_EMULATION__
            // printf("    thisFactor=%d, tid=%d: thisMark = %u\n", thisFactor, threadIdx.x, thisMark);
            // #endif
            CLR_BYTE(g_all_primes, thisMark-1);
            if(thisMark == 0) return;
            thisMark += thisFactor*blockDim.x;
        }
        __syncthreads();

        if(threadIdx.x == 0)
            ++primeFactors_index;
        __syncthreads();
    }
}


#endif // #ifndef _TEMPLATE_KERNEL_H_
