#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <limits.h>

#include <cutil_inline.h>
#include <countprimes_kernel.cu>
#include <defs.h>


uint32 findPrimes(uint32 ulimit, byte *array, uint32 *array_out) {
    CLR_BYTE(array, 1-1);
    for(uint32 i = 2; i <= ulimit; i++)
        SET_BYTE(array, i-1);
    uint32 thisFactor = 2;
    uint32 thisMark;
    const uint32 sqrt_ulimit = (uint32)floor(sqrt((double)ulimit));
    while(thisFactor <= sqrt_ulimit) {
        thisMark = thisFactor + thisFactor;
        while(thisMark <= ulimit) {
            CLR_BYTE(array, thisMark-1);
            thisMark += thisFactor;
        }
        do  // Search for the next prime divisor
            thisFactor++; 
        while(GET_BYTE(array, thisFactor-1) == 0);
    }
    // counting and saving primes in integer array
    uint32 prime_count = 0;
    for(uint32 i = 1; i <= ulimit; i++)
        if(GET_BYTE(array, i-1)) {
            array_out[prime_count] = i;
            //printf("%d: %d\n", prime_count, array_out[prime_count]);
            prime_count++;
    }
    return prime_count;
}

void markPrimesPattern(uint32 ulimit, uint32 lastFactor, byte* precomputed_primes, byte* array) {
    for(uint32 i = 1; i <= ulimit; i++)
        SET_BYTE(array, i-1);
    uint32 thisFactor = 2;
    uint32 thisMark;
    const uint32 sqrt_ulimit = (uint32)floor(sqrt((double)ulimit));
    while(thisFactor <= lastFactor && thisFactor <= sqrt_ulimit) {
        thisMark = thisFactor;
        while(thisMark <= ulimit) {
            CLR_BYTE(array, thisMark-1);
            thisMark += thisFactor;
        }
        do  // Search for the next prime divisor in precomputed_primes
            thisFactor++;
        while(GET_BYTE(precomputed_primes, thisFactor-1) == 0);
    }
}

void countPrimes(   const uint32 ulimit,  
                    const uint32 *primeFactors,
                    const uint32 primeFactors_count,
                    const uint32 firstFactor_index,
                    byte* array     )
{
    uint32 primeFactors_index = firstFactor_index;
    uint32 thisFactor = primeFactors[primeFactors_index];
    while(primeFactors_index < primeFactors_count) {
        uint32 thisMark = thisFactor * thisFactor;
        while(thisMark % thisFactor)
            thisMark++;
        while(thisMark <= ulimit) {
            CLR_BYTE(array, thisMark-1);
            thisMark += thisFactor;
        }
        primeFactors_index++;
        thisFactor = primeFactors[primeFactors_index];
    }
}


int main(int argc, char *argv[]) {

// GPU specific information
    uint16 num_threads = 128;
    uint16 num_blocks = 30*512;
    
// read command line arguments and set up upper limit
    assert(argc == 2 || argc == 3);
    double ul_double;   // upper limit, inclusive
    sscanf(argv[1], "%lf", &ul_double);
    if(argc == 3)
        sscanf(argv[2], "%hu", &num_blocks);

    #ifdef __DEVICE_EMULATION__
    num_threads = 4;
    num_blocks = 10;
    #endif

    const uint32 llimit = 1;
    const uint32 ulimit = (uint32)ul_double;
    const uint32 sqrt_ulimit = (uint32)floor(sqrt((double)ulimit));
    assert(llimit <= ulimit);
    assert(ulimit <= 1000000000);   // 1e9
    assert(ulimit <= CONSTANT_MEM_SIZE*CONSTANT_MEM_SIZE); // 2^32 = 4294967296 = 4.29e10
    printf("Counting primes less than %u...\n", ulimit);
    printf("sqrt(%u) = %u\n", ulimit, sqrt_ulimit);

// setting up precomputed primes
    const uint32 lastFactor_pre = 13;
    // const uint32 firstFactor_sieve = 17;
    // LCM(2,3,5,7,11,13, 64) (to align with 64-byte boundary of device memory)
    const uint32 num_bytes_pattern = 960960;

// precomputing primes upto sqrt(ulimit)
    byte *precomputed_primes = NULL;
    uint32 *primeFactors = NULL;
    precomputed_primes = (byte*)malloc(sqrt_ulimit);    // byte-wise array
    primeFactors = (uint32*)malloc(3401 * sizeof(uint32));    // pi(sqrt_ulimit) = 3401
    assert(precomputed_primes != NULL && primeFactors != NULL);
    uint32 timer_pre = 0;
    cutilCheckError(cutCreateTimer(&timer_pre));
    cutilCheckError(cutStartTimer(timer_pre));
    uint32 prime_precounter = findPrimes(sqrt_ulimit, precomputed_primes, primeFactors);    // call the function
    cutilCheckError(cutStopTimer(timer_pre));
    printf("%u primes found from the precomputed list [1, %u]\n", prime_precounter, sqrt_ulimit);

// precomputing pattern of non-primes which are multiples of some intial primes
    byte* precomputed_pattern = NULL;           
    precomputed_pattern = (byte*)malloc(num_bytes_pattern);     // byte-wise array
    assert(precomputed_pattern != NULL);
    cutilCheckError(cutStartTimer(timer_pre));
    markPrimesPattern(num_bytes_pattern, lastFactor_pre, precomputed_primes, precomputed_pattern);   // call the function
    cutilCheckError(cutStopTimer(timer_pre));
    float time_pre = cutGetTimerValue(timer_pre);
    printf("CPU: %fms and %u bytes taken to precompute primes between [1, %u] and marking the pattern [1, %u] \n", 
                    time_pre, sqrt_ulimit, sqrt_ulimit, num_bytes_pattern);

// CPU memory allocation and filling it with precomputed_pattern
    uint32 num_bytes = ulimit-llimit+1;
    // printf("num_bytes = %u, %.2fMB\n", num_bytes, num_bytes/1024.0/1024.0);
    byte* all_primes = NULL;
    all_primes = (byte*)malloc(num_bytes);
    assert(all_primes != NULL);    
    for(uint32 i = 0; i < num_bytes; i += num_bytes_pattern) {
        if(i+num_bytes_pattern > num_bytes)
            memcpy(all_primes+i, precomputed_pattern, num_bytes-i);
        else
            memcpy(all_primes+i, precomputed_pattern, num_bytes_pattern);
    }
    // mark 1 as non-prime (special case)
    CLR_BYTE(all_primes, 1-1);
    // mark primes under sqrt_ulimit
    for(int i = 0; i < prime_precounter; i++)
        SET_BYTE(all_primes, primeFactors[i]-1);
    

    // for(int i = 1; i <= ulimit; i++)
        // printf("%s%d%s\n", all_primes[i-1] ? "      " : "------", i, 
                           // all_primes[i-1] ? "      " : "------");
    // return EXIT_SUCCESS;
    

// now using GPU...
    // copy primeFactors in device constant memory
    cutilSafeCall(cudaMemcpyToSymbol(d_primeFactors, primeFactors, prime_precounter*sizeof(uint32)));
    //cutilSafeCall(cudaMemcpyToSymbol((char*)&ulimit_constmem, (char*)&ulimit, sizeof(ulimit)));
    byte* d_all_primes = NULL;
    cutilSafeCall(cudaMalloc(&d_all_primes, num_bytes));
    assert(d_all_primes != NULL);
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_all_primes, all_primes, num_bytes, cudaMemcpyHostToDevice));
    uint32 timer_gpu_kernel = 0; cutilCheckError(cutCreateTimer(&timer_gpu_kernel));
    // cook the kernel
    printf("launching kernel with %u blocks and %u threads...\n", num_blocks, num_threads);
    primeKernel<<<num_blocks, num_threads>>>(ulimit, d_all_primes, prime_precounter, 1);
    cutilCheckError(cutStartTimer(timer_gpu_kernel));
    // launch the kernel
    {
        primeKernel<<<num_blocks, num_threads>>>(ulimit, d_all_primes, prime_precounter, 0);
        cutilCheckMsg("Kernel execution failed");
        cudaThreadSynchronize();
        //byte dummy_ptr;
        //cutilSafeCall(cudaMemcpy(&dummy_ptr, d_all_primes, 1, cudaMemcpyDeviceToHost));
    }
    cutilCheckError(cutStopTimer(timer_gpu_kernel));
    float time_gpu_kernel = cutGetTimerValue(timer_gpu_kernel);
    byte* all_primes_gpu_result = NULL;
    all_primes_gpu_result = (byte*)malloc(num_bytes);
    assert(all_primes_gpu_result != NULL);
    cutilSafeCall(cudaMemcpy(all_primes_gpu_result, d_all_primes, num_bytes, cudaMemcpyDeviceToHost));
    // counting primes
    uint32 prime_counter_gpu;
    prime_counter_gpu = 0;
    for(uint32 i = llimit; i <= ulimit; i++)
        if(GET_BYTE(all_primes_gpu_result, i-llimit))
            prime_counter_gpu++;
    printf("GPU: %u primes found between [%u, %u] in %.3f ms\n", prime_counter_gpu, llimit, ulimit, time_gpu_kernel);
    free(all_primes_gpu_result);

#if 0
// reference solution by CPU
    uint32 timer_cpu_kernel = 0;
    cutilCheckError(cutCreateTimer(&timer_cpu_kernel));
    cutilCheckError(cutStartTimer(timer_cpu_kernel));
    {
        countPrimes(ulimit, primeFactors, prime_precounter, 6, all_primes);
    }
    cutilCheckError(cutStopTimer(timer_cpu_kernel));
    float time_cpu_kernel = cutGetTimerValue(timer_cpu_kernel);
    // counting primes
    uint32 prime_counter = 0;
    for(uint32 i = llimit; i <= ulimit; i++)
        if(GET_BYTE(all_primes, i-llimit)) 
            prime_counter++;
    printf("CPU: %u primes found between [%u, %u] in %.3f ms\n", prime_counter, llimit, ulimit, time_cpu_kernel);
    if(prime_counter != prime_counter_gpu)
        printf("ERROR: CPU crossed out %u less non-primes than GPU\n", prime_counter - prime_counter_gpu);
    else
        printf("SUCCESS!!\n");
        
    free(all_primes);
#else
    float time_cpu_kernel = 0;
#endif

    fprintf(stderr, "%u %f %f\n", ulimit, time_cpu_kernel, time_gpu_kernel);
    NEWLINE;
    
    return 0;
}
