#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <limits.h>

#include <cutil_inline.h>
#include <countprimes_kernel.cu>
#include <defs.h>


void findPrimes(uint64 ulimit, byte* array) {
    CLR_BIT(array, 1-1);
    for(uint64 i = 2; i <= ulimit; i++)
        SET_BIT(array, i-1);
    uint64 thisFactor = 2;
    uint64 mark;
    while(thisFactor * thisFactor <= ulimit) {
        mark = thisFactor + thisFactor;
        while(mark <= ulimit) {
            CLR_BIT(array, mark-1);
            mark += thisFactor;
        }
        // Search for the next prime divisor
        do thisFactor++; while(GET_BIT(array, thisFactor-1) == 0);
        assert(thisFactor <= ulimit);
    }
}

void countPrimes_range(uint64 llimit, uint64 ulimit, byte* precomputed_primes, byte* array) {
    if(llimit == 0) {
        CLR_BYTE(array, 0);
        CLR_BYTE(array, 1);
    }
    else if(llimit == 1) {
        CLR_BYTE(array, 0);
    }
    uint64 thisFactor = 17;
    uint64 mark;
    uint64 first_multiple;
    while(thisFactor*thisFactor <= ulimit) {
        first_multiple = llimit;
        while(first_multiple % thisFactor)
            first_multiple++;
        mark = first_multiple;
        while(mark <= ulimit) {
            if(GET_BYTE(array, mark-llimit))
                CLR_BYTE(array, mark-llimit);
            mark += thisFactor;
        }
        do  // Search for the next prime divisor
            thisFactor++;
        while(GET_BIT(precomputed_primes, thisFactor-1) == 0);
        assert(thisFactor <= ulimit);
    }
}

void markPrimesPattern(uint64 llimit, uint64 ulimit, uint64 top_divisor, byte* precomputed_primes, byte* array) {
    for(uint64 i = llimit; i <= ulimit; i++)
        SET_BYTE(array, i-llimit);
    uint64 thisFactor = 2;
    uint64 mark;
    while(thisFactor <= top_divisor && thisFactor*thisFactor <= ulimit) {
        mark = llimit - 1 + thisFactor;
//         printf("thisFactor = %llu\n", thisFactor);
        while(mark <= ulimit) {
            CLR_BYTE(array, mark-llimit);
//             printf("  thisFactor=%llu, mark=%llu, (mark-llimit)=%llu, (mark-llimit)>>3=%llu\n", thisFactor, mark, mark-llimit, (mark-llimit)>>3);
            mark += thisFactor;
        }
        do  // Search for the next prime divisor in precomputed_primes
            thisFactor++;
        while(GET_BYTE(precomputed_primes, thisFactor-1) == 0);
        assert(thisFactor <= ulimit);
    }
}


int main(int argc, char *argv[]) {
    uint64 llimit, ulimit;  // upper and lower limits, both inclusive
    float ll_float, ul_float;
    assert(argc == 3);
    sscanf(argv[1], "%f", &ll_float);
    sscanf(argv[2], "%f", &ul_float);
    llimit = (uint64)ll_float;
    ulimit = (uint64)ul_float;
    assert(llimit <= ulimit);
    assert(ulimit <= 274877906944);
    printf("Counting primes in the interval [%llu, %llu]...\n", llimit, ulimit);

    uint64 precomputed_top = SIXTYFOUR_KB * 8;    // 524,288
    byte precomputed_primes[SIXTYFOUR_KB];
    findPrimes(precomputed_top, precomputed_primes);

    uint64 prime_counter = 0;

    if(ulimit <= precomputed_top) {
        printf("No need to use GPU...\n");
        prime_counter = 0;
        for(uint64 i = llimit; i <= ulimit; i++)
            if(GET_BIT(precomputed_primes, i-1))
                prime_counter++;
        printf("%llu primes found between [%llu, %llu]\n", prime_counter, llimit, ulimit);
        return 0;
    }

    if(llimit <= precomputed_top) {
        printf("counting some primes from the precomputed list...\n");
        prime_counter = 0;
        for(uint64 i = llimit; i <= precomputed_top; i++)
            if(GET_BIT(precomputed_primes, i-1))
                prime_counter++;
        printf("%llu primes found between [%llu, %llu]\n", prime_counter, llimit, precomputed_top);
    }

    byte* precomputed_pattern = NULL;    // pattern of marked non-primes which are multiples of (2,3,5,7,11,13)
    precomputed_pattern = (byte*)malloc(120120);    // byte-wise array
    markPrimesPattern(1, 120120, 13, precomputed_primes, precomputed_pattern);

// memory allocation
    uint64 num_bytes = ulimit-llimit+1;
    printf("num_bytes = %llu, %.2fMB\n", num_bytes, num_bytes/1024.0/1024.0);
    byte* all_primes = NULL;
    all_primes = (byte*)malloc(num_bytes);
    for (uint32 i = 0; i < num_bytes; i += 120120)
        memcpy(all_primes+i, precomputed_pattern, 120120);
//         printf("i=%6u, i+120120=%6u\n", i, i+120120);


// now using GPU...
    // allocate device memory
    cutilSafeCall(cudaMemcpyToSymbol(d_precomputed_primes, precomputed_primes, SIXTYFOUR_KB));
    byte* d_all_primes = NULL;
    cutilSafeCall(cudaMalloc(&d_all_primes, num_bytes));
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_all_primes, all_primes, num_bytes, cudaMemcpyHostToDevice));

    uint32 timer_cpu = 0; cutilCheckError(cutCreateTimer(&timer_cpu));
    uint32 timer_gpu = 0; cutilCheckError(cutCreateTimer(&timer_gpu));

    // cook the kernel
    primeKernel_1<<<1, 1, 0>>>(llimit, ulimit, d_all_primes, 1);
    cutilCheckError(cutStartTimer(timer_gpu));
    // launch the kernel
    {
        primeKernel_1<<<1, 1, 0>>>(llimit, ulimit, d_all_primes, 0);
        cutilCheckMsg("Kernel execution failed");
        cudaThreadSynchronize();
    }
    cutilCheckError(cutStopTimer(timer_gpu));
    float time_gpu = cutGetTimerValue(timer_gpu);
    cutilCheckError(cutDeleteTimer(timer_gpu));

    byte* all_primes_gpu_result = NULL;
    all_primes_gpu_result = (byte*)malloc(num_bytes);
    cutilSafeCall(cudaMemcpy(all_primes_gpu_result, d_all_primes, num_bytes, cudaMemcpyDeviceToHost));

    // counting primes
    prime_counter = 0;
    for(uint64 i = llimit; i <= ulimit; i++)
        if(GET_BYTE(all_primes_gpu_result, i-llimit))
            prime_counter++;
    printf("GPU: %llu primes found between [%llu, %llu] in %f ms\n", prime_counter, llimit, ulimit, time_gpu);
    free(all_primes_gpu_result);


// reference solution by CPU
    cutilCheckError(cutStartTimer(timer_cpu));
    {
        countPrimes_range(llimit, ulimit, precomputed_primes, all_primes);
    }
    cutilCheckError(cutStopTimer(timer_cpu));
    float time_cpu = cutGetTimerValue(timer_cpu);
    cutilCheckError(cutDeleteTimer(timer_cpu));

    // counting primes
    prime_counter = 0;
    for(uint64 i = llimit; i <= ulimit; i++)
        if(GET_BYTE(all_primes, i-llimit))
            prime_counter++;
    printf("CPU: %llu primes found between [%llu, %llu] in %f ms\n", prime_counter, llimit, ulimit, time_cpu);
    free(all_primes);

    return 0;
}
