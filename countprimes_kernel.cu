#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include <defs.h>

__device__ __constant__ byte    d_precomputed_primes[65536];

__global__ void primeKernel(    const uint32 ulimit,
                                byte* g_all_primes,
                                const uint32 sqrt_ulimit,
                                const byte cook             )
{
    if(cook) return;

    __shared__ uint32 thisFactor;
    __shared__ uint32 lastFactor;

    if(threadIdx.x == 0) 
    {
        uint32 block_chunk = (uint32) ceil((float)(sqrt_ulimit-17+1) / gridDim.x);
        if(blockIdx.x == 0)     // for 0th block, use first unused factor
            thisFactor = 17;
        else
        {
            thisFactor = 17 + block_chunk * blockIdx.x;
            while(GET_BYTE(d_precomputed_primes, thisFactor-1) == 0)
                thisFactor++;   // Search for the next prime divisor in precomputed_primes
        }
        lastFactor = (17 + block_chunk * blockIdx.x) + block_chunk;
    }
    __syncthreads();

    // main loop
    //int c = 0;
    while(thisFactor < lastFactor) {
        //#ifdef __DEVICE_EMULATION__
        //if(threadIdx.x == 0)
            //printf("  iteration%d: thisFactor = %u\n", c++, thisFactor);
        //__syncthreads();
        //#endif

        uint32 thisMark = thisFactor * (1 + threadIdx.x);
        while(thisMark <= ulimit) {
            //#ifdef __DEVICE_EMULATION__
            //if(threadIdx.x == 0)
                //printf("    thisMark = %u\n", thisMark);
            //#endif
            CLR_BYTE(g_all_primes, thisMark-1);
            thisMark += thisFactor*blockDim.x;
        }
        __syncthreads();

        if(threadIdx.x == 0) {
            do  // Search for the next prime divisor in precomputed_primes
                thisFactor++;
            while(GET_BYTE(d_precomputed_primes, thisFactor-1) == 0);
        }
        __syncthreads();
    }
}


#endif // #ifndef _TEMPLATE_KERNEL_H_
