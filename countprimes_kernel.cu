#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include <defs.h>

#define SDATA(index)      cutilBankChecker(sdata, index)
// #define SDATA(index)      sdata[index]


__device__ __constant__ unsigned char d_precomputed_primes[65536];

__global__ void primeKernel_1(uint64 llimit, uint64 ulimit, byte* all_primes, byte cook)
{
    if(cook) return;

    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

//     printf("\n");
//     for(uint64 i = llimit; i <= ulimit; i++)
//         printf("%2llu : %u\n", i%120120, GET_BIT(d_precomputed_pattern, i-llimit));


    uint64 thisFactor = 17;
    uint64 mark;
    uint64 first_multiple;
    while(thisFactor*thisFactor <= ulimit) {
        first_multiple = llimit;
        while(first_multiple % thisFactor)
            first_multiple++;
        mark = first_multiple;
//         printf("  thisFactor = %llu\n", thisFactor);
        while(mark <= ulimit) {
            CLR_BIT(all_primes, mark-llimit);
//             printf("    thisFactor=%llu, mark=%llu, (mark-llimit)=%llu, (mark-llimit)>>3=%llu\n", thisFactor, mark, mark-llimit, (mark-llimit)>>3);
            mark += thisFactor;
        }
        do  // Search for the next prime divisor in precomputed_primes
            thisFactor++;
        while(GET_BIT(d_precomputed_primes, thisFactor-1) == 0);
    }


//     uint64 prime_counter = 0;
//     for(uint64 i = llimit; i <= ulimit; i++)
//         if(GET_BIT(all_primes, i-llimit))
//             prime_counter++;

//     printf("DEVICE: %llu primes found between [%llu, %llu]\n", prime_counter, llimit, ulimit);




    // access thread id
//     const unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    // copy the global data to shared memory
    __syncthreads();

//     if(threadIdx.x == 0 || threadIdx.x == blockDim.x-1)
//         for(unsigned int i = 0; i < N; i++)
//             printf("gtid:%d.%d SDATA(%d) = %f\n", blockIdx.x, threadIdx.x, i, SDATA(i));
//     __syncthreads();

}


#endif // #ifndef _TEMPLATE_KERNEL_H_
