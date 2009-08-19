#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include <defs.h>

#define SDATA(index)      cutilBankChecker(sdata, index)
// #define SDATA(index)      sdata[index]


__device__ __constant__ unsigned char d_precomputed_primes[65536];

__global__ void primeKernel(uint64 llimit, uint64 ulimit, byte* g_all_primes, uint32 firstFactor, uint16 num_threads, byte cook)
{
    if(cook) return;

    // shared memory
//     extern  __shared__  float sdata[];

//     const uint32 offset = threadIdx.x;

    uint32 sqrt_ulimit = (uint32)ceil(sqrt((float)ulimit));
    __shared__ uint32 thisFactor;
    __shared__ uint64 first_multiple;
    uint64 mark;

    if(threadIdx.x == 0)
        thisFactor = firstFactor;
    __syncthreads();

    while(thisFactor <= sqrt_ulimit) {
        if(threadIdx.x == 0) {
            first_multiple = llimit;
            while(first_multiple % thisFactor)
                first_multiple++;
        }
        __syncthreads();

        mark = first_multiple + thisFactor*threadIdx.x;
//         printf("  thisFactor = %llu\n", thisFactor);
        while(mark <= ulimit) {
            g_all_primes[mark-llimit] = 0;
//             CLR_BYTE(g_all_primes, mark-llimit);
//             printf("    thisFactor=%llu, mark=%llu, (mark-llimit)=%llu, (mark-llimit)>>3=%llu\n", thisFactor, mark, mark-llimit, (mark-llimit)>>3);
            mark += thisFactor*blockDim.x;
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
