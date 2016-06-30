#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include <defs.h>

__device__ __constant__ unsigned char d_precomputed_primes[65536];

__global__ void primeKernel(    const uint64 llimit,
                                const uint64 ulimit,
                                byte* g_all_primes,
                                uint32 firstFactor,
                                const uint32 num_bytes_pre,
                                const uint32 sqrt_ulimit,
                                const byte cook             )
{
    if(cook) return;

    __shared__ uint32 thisFactor;
    __shared__ uint64 firstMultiple;
    uint32 lastFactor;

    if(blockIdx.x != 0 && threadIdx.x == 0) {
        firstFactor = (uint32)ceil(num_bytes_pre*8.0f / gridDim.x) * blockIdx.x;
        while(GET_BIT(d_precomputed_primes, firstFactor-1) == 0)
            firstFactor++;   // Search for the next prime divisor in precomputed_primes
    }
    if(threadIdx.x == 0)
        thisFactor = firstFactor;
    __syncthreads();
    lastFactor = (uint32)ceil(num_bytes_pre*8.0f / gridDim.x) * (blockIdx.x+1) - 1;

    while(thisFactor <= lastFactor) {
        if(threadIdx.x == 0) {
            firstMultiple = llimit;
            while(firstMultiple % thisFactor)
                firstMultiple++;
        }
        __syncthreads();

        uint64 thisMark = firstMultiple + thisFactor*threadIdx.x;
        while(thisMark <= ulimit) {
            CLR_BYTE(g_all_primes, thisMark-llimit);
            thisMark += thisFactor*blockDim.x;
        }
        __syncthreads();

        if(threadIdx.x == 0) {
            do  // Search for the next prime divisor in precomputed_primes
                thisFactor++;
            while(GET_BIT(d_precomputed_primes, thisFactor-1) == 0);
        }
        __syncthreads();
    }
}


#endif // #ifndef _TEMPLATE_KERNEL_H_
