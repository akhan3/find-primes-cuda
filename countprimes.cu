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
    uint64 thisMark;
    const uint32 sqrt_ulimit = (uint32)floor(sqrt((double)ulimit));
    while(thisFactor <= sqrt_ulimit) {
        thisMark = thisFactor + thisFactor;
        while(thisMark <= ulimit) {
            CLR_BIT(array, thisMark-1);
            thisMark += thisFactor;
        }
        // Search for the next prime divisor
        do thisFactor++; while(GET_BIT(array, thisFactor-1) == 0);
    }
}

void markPrimesPattern(uint64 llimit, uint64 ulimit, uint64 lastFactor, byte* precomputed_primes, byte* array) {
    assert(llimit == 1);
    for(uint64 i = llimit; i <= ulimit; i++)
        SET_BYTE(array, i-llimit);
    uint64 thisFactor = 2;
    uint64 thisMark;
    const uint32 sqrt_ulimit = (uint32)floor(sqrt((double)ulimit));
    while(thisFactor <= lastFactor && thisFactor <= sqrt_ulimit) {
        thisMark = llimit - 1 + thisFactor;
        while(thisMark <= ulimit) {
            CLR_BYTE(array, thisMark-llimit);
            thisMark += thisFactor;
        }
        do  // Search for the next prime divisor in precomputed_primes
            thisFactor++;
        while(GET_BIT(precomputed_primes, thisFactor-1) == 0);
    }
}

void countPrimes_range(     const uint64 llimit,
                            const uint64 ulimit,
                            const byte* precomputed_primes,
                            const uint32 firstFactor,
                            const uint32 precomputed_top,
                            byte* array     )
{
//     printf("GOLDEN: [%llu, %llu] %u %u\n", llimit, ulimit, firstFactor, precomputed_top);
    assert(llimit >= 2);
    uint32 thisFactor = firstFactor;
    const uint32 sqrt_ulimit = (uint32)floor(sqrt((double)ulimit));
    while(thisFactor <= sqrt_ulimit) {
//         printf("GOLDEN: thisFactor=%u, precomputed_top=%u, sqrt_ulimit=%u\n", thisFactor, precomputed_top, sqrt_ulimit);
        uint64 thisMark = llimit;
        while(thisMark % thisFactor)
            thisMark++;
        while(thisMark <= ulimit) {
            CLR_BYTE(array, thisMark-llimit);
            thisMark += thisFactor;
        }
        do  // Search for the next prime divisor in precomputed_primes
            thisFactor++;
        while(GET_BIT(precomputed_primes, thisFactor-1) == 0);
        assert(thisFactor <= precomputed_top);
    }
}


int main(int argc, char *argv[]) {
    const uint16 num_mp = 64;
    const uint16 num_threads = 64;
    const uint32 lastFactor_pre = 13;
    const uint32 firstFactor_sieve = 17;
    // LCM(2,3,5,...initial primes, 8, 64) (to align with 64-byte boundary of device memory)
    const uint32 num_bytes_pattern = 3*5*7*11*13*64;

    double ll_double, ul_double;   // upper and lower limits, both inclusive
    assert(argc >= 3);
    sscanf(argv[1], "%lf", &ll_double);
    sscanf(argv[2], "%lf", &ul_double);
    const uint64 llimit = (ll_double < 2.0) ? (uint64)2 : (uint64)ll_double;
    const uint64 ulimit = (uint64)ul_double;
    const uint32 sqrt_ulimit = (uint32)floor(sqrt((double)ulimit));
    assert(llimit <= ulimit);
    assert(ulimit - llimit >= (uint64)1.9e5);
    assert(ulimit <= (CONSTANT_MEM_SIZE*8)*(CONSTANT_MEM_SIZE*8)); // 2^38 = 274877906944
    printf("Counting primes in the interval [%llu, %llu]...\n", llimit, ulimit);

// precomputing primes upto sqrt(ulimit)
    uint32 precomputed_top = (uint32)(ceil(floor(sqrt((double)ulimit))/(num_mp*8.0))*(num_mp*8));
    uint32 num_bytes_pre = (uint32)ceil(precomputed_top/8.0);   // 8 numbers per byte
    printf("num_bytes_pre = %u Bytes, %.2fKB\n", num_bytes_pre, num_bytes_pre/1024.0);
    byte* precomputed_primes = NULL;
    precomputed_primes = (byte*)malloc(num_bytes_pre);          // bit-wise array
    assert(precomputed_primes != NULL);
    uint32 timer_pre = 0;
    cutilCheckError(cutCreateTimer(&timer_pre));
    cutilCheckError(cutStartTimer(timer_pre));
    findPrimes(precomputed_top, precomputed_primes);    // call the function
    cutilCheckError(cutStopTimer(timer_pre));

// counting some primes from the precomputed primes list
    uint64 prime_precounter = 0;
    if(llimit <= precomputed_top) {
        for(uint64 i = llimit; i <= precomputed_top; i++)
            if(GET_BIT(precomputed_primes, i-1))
                prime_precounter++;
        printf("  %llu primes found from the precomputed list [%llu, %u]\n", prime_precounter, llimit, precomputed_top);
    }

// precomputing pattern of non-primes which are multiples of some intial primes
    byte* precomputed_pattern = NULL;           // byte-wise array
    precomputed_pattern = (byte*)malloc(num_bytes_pattern);
    assert(precomputed_pattern != NULL);
    cutilCheckError(cutStartTimer(timer_pre));
    markPrimesPattern(1, num_bytes_pattern, lastFactor_pre, precomputed_primes, precomputed_pattern);   // call the function
    cutilCheckError(cutStopTimer(timer_pre));
    float time_pre = cutGetTimerValue(timer_pre);
    printf("CPU: %fms and %u bytes taken to precompute primes between [1, %u] and marking the pattern [1, %u] \n", time_pre, num_bytes_pre, precomputed_top, num_bytes_pattern);

// CPU memory allocation and filling it with precomputed_pattern
    uint64 num_bytes = ulimit-llimit+1;
    printf("num_bytes = %llu, %.2fMB\n", num_bytes, num_bytes/1024.0/1024.0);
    byte* all_primes = NULL;
    all_primes = (byte*)malloc(num_bytes);
    assert(all_primes != NULL);
    int patternboundray_2_llimit = ((llimit-1) % num_bytes_pattern);
    int llimit_2_patternboundray = num_bytes_pattern - patternboundray_2_llimit;
    uint32 start_address = 0;
    if(patternboundray_2_llimit != 0) {
        memcpy(all_primes, precomputed_pattern+patternboundray_2_llimit, llimit_2_patternboundray);
        start_address = llimit_2_patternboundray;
    }
    for(uint32 i = start_address; i < num_bytes; i += num_bytes_pattern) {
        if(i+num_bytes_pattern > num_bytes)
            memcpy(all_primes+i, precomputed_pattern, num_bytes-i);
        else
            memcpy(all_primes+i, precomputed_pattern, num_bytes_pattern);
    }

// now using GPU...
    // copy precomputed_primes in device constant memory
    cutilSafeCall(cudaMemcpyToSymbol(d_precomputed_primes, precomputed_primes, num_bytes_pre));
    byte* d_all_primes = NULL;
    cutilSafeCall(cudaMalloc(&d_all_primes, num_bytes));
    assert(d_all_primes != NULL);
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_all_primes, all_primes, num_bytes, cudaMemcpyHostToDevice));
    uint32 timer_gpu = 0; cutilCheckError(cutCreateTimer(&timer_gpu));
    // cook the kernel
    printf("launching kernel with %u blocks and %u threads...\n", num_mp, num_threads);
    primeKernel<<<num_mp, num_threads>>>(llimit, ulimit, d_all_primes, firstFactor_sieve, num_bytes_pre, sqrt_ulimit, 1);
    cutilCheckError(cutStartTimer(timer_gpu));
    // launch the kernel
    {
        primeKernel<<<num_mp, num_threads>>>(llimit, ulimit, d_all_primes, firstFactor_sieve, num_bytes_pre, sqrt_ulimit, 0);
        cutilCheckMsg("Kernel execution failed");
        cudaThreadSynchronize();
    }
    cutilCheckError(cutStopTimer(timer_gpu));
    float time_gpu = cutGetTimerValue(timer_gpu);

    byte* all_primes_gpu_result = NULL;
    all_primes_gpu_result = (byte*)malloc(num_bytes);
    assert(all_primes_gpu_result != NULL);
    cutilSafeCall(cudaMemcpy(all_primes_gpu_result, d_all_primes, num_bytes, cudaMemcpyDeviceToHost));
    // counting primes
    uint64 prime_counter_gpu;
    prime_counter_gpu = prime_precounter;
    for(uint64 i = llimit; i <= ulimit; i++)
        if(GET_BYTE(all_primes_gpu_result, i-llimit))
            prime_counter_gpu++;
    printf("GPU: %llu primes found between [%llu, %llu] in %.3f ms\n", prime_counter_gpu, llimit, ulimit, time_gpu);
    free(all_primes_gpu_result);

#if 1
// reference solution by CPU
    uint32 timer_cpu = 0;
    cutilCheckError(cutCreateTimer(&timer_cpu));
    cutilCheckError(cutStartTimer(timer_cpu));
    {
        countPrimes_range(llimit, ulimit, precomputed_primes, firstFactor_sieve, precomputed_top, all_primes);
    }
    cutilCheckError(cutStopTimer(timer_cpu));
    float time_cpu = cutGetTimerValue(timer_cpu);
    // counting primes
    uint64 prime_counter = prime_precounter;
    for(uint64 i = llimit; i <= ulimit; i++)
        if(GET_BYTE(all_primes, i-llimit))
            prime_counter++;
    printf("CPU: %llu primes found between [%llu, %llu] in %.3f ms\n", prime_counter, llimit, ulimit, time_cpu);
    if(prime_counter - prime_counter_gpu != 0)
        printf("ERROR: CPU crossed out %llu less non-primes than GPU\n", prime_counter - prime_counter_gpu);
    free(all_primes);
#endif

    return 0;
}
