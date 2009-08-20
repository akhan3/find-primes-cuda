#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include <defs.h>

#define SDATA(index)      cutilBankChecker(sdata, index)
// #define SDATA(index)      sdata[index]


__device__ __constant__ unsigned char d_precomputed_primes[65536];

__global__ void primeKernel(    const uint64 llimit, const uint64 ulimit,
                                byte* g_all_primes,
                                uint32 firstFactor,
                                const uint32 num_bytes_pre,
                                const uint32 sqrt_ulimit,
                                const byte cook   )
{
    if(cook) return;

// shared memory
    //extern  __shared__  float sdata[];

    __shared__ uint32 thisFactor;
    __shared__ uint64 first_multiple;
    uint32 lastFactor;
    uint64 thisMark;

    if(blockIdx.x != 0 && threadIdx.x == 0) {
        firstFactor = (uint32)ceil(num_bytes_pre*8.0f / gridDim.x) * blockIdx.x;
        while(GET_BIT(d_precomputed_primes, firstFactor-1) == 0)
            firstFactor++;   // Search for the next prime divisor in precomputed_primes
    }
    __syncthreads();
    if(threadIdx.x == 0)
        thisFactor = firstFactor;
    __syncthreads();
    lastFactor = (uint32)ceil(num_bytes_pre*8.0f / gridDim.x) * (blockIdx.x+1) - 1;

    while(thisFactor <= lastFactor) {
        if(threadIdx.x == 0) {
            first_multiple = llimit;
            while(first_multiple % thisFactor)
                first_multiple++;
        }
        __syncthreads();

        thisMark = first_multiple + thisFactor*threadIdx.x;
//         printf("KERNEL%u.%u: thisFactor=%u, first_multiple=%llu, thisMark=%llu\n", blockIdx.x, threadIdx.x, thisFactor, first_multiple, thisMark);
        while(thisMark <= ulimit) {
            CLR_BYTE(g_all_primes, thisMark-llimit);
            thisMark += thisFactor*blockDim.x;
        }
        __syncthreads();
//         printf("KERNEL%u.%u: thisFactor=%u, first_multiple=%llu, thisMark=%llu\n", blockIdx.x, threadIdx.x, thisFactor, first_multiple, thisMark);

        if(threadIdx.x == 0) {
            do  // Search for the next prime divisor in precomputed_primes
                thisFactor++;
            while(GET_BIT(d_precomputed_primes, thisFactor-1) == 0);
        }
        __syncthreads();
    }
}


#endif // #ifndef _TEMPLATE_KERNEL_H_
