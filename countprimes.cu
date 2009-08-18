#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <cutil_inline.h>
#include <countprimes_kernel.cu>
#include <defs.h>


void findPrimes (uint64 ulimit, byte* array) {
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

void markPrimesPattern(uint64 llimit, uint64 ulimit, uint64 top_divisor, byte* precomputed_primes, byte* array) {
    for(uint64 i = llimit; i <= ulimit; i++)
        SET_BIT(array, i-llimit);
    uint64 thisFactor = 2;
    uint64 mark;
    while(thisFactor <= top_divisor && thisFactor*thisFactor <= ulimit) {
        mark = llimit - 1 + thisFactor;
//         printf("thisFactor = %llu\n", thisFactor);
        while(mark <= ulimit) {
            CLR_BIT(array, mark-llimit);
//             printf("  thisFactor=%llu, mark=%llu, (mark-llimit)=%llu, (mark-llimit)>>3=%llu\n", thisFactor, mark, mark-llimit, (mark-llimit)>>3);
            mark += thisFactor;
        }
        do  // Search for the next prime divisor in precomputed_primes
            thisFactor++;
        while(GET_BIT(precomputed_primes, thisFactor-1) == 0);
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

    const uint64 SIXTYFOUR_KB = 65536;
    uint64 precomputed_top = SIXTYFOUR_KB * 8;    // 524,288
    byte precomputed_primes[SIXTYFOUR_KB];
    findPrimes(precomputed_top, &precomputed_primes[0]);

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

    byte* precomputed_pattern = 0;    // pattern of marked non-primes which are multiples of (2,3,5,7,11,13)
    precomputed_pattern = (byte*)malloc(15015);
    markPrimesPattern(30031, 30031+120120-1, 13, &precomputed_primes[0], &precomputed_pattern[0]);
//     for(uint64 i = 1; i <= 120120; i++)
//         printf("%2llu : %u\n", i, GET_BIT(precomputed_pattern, i-1));

    // now using GPU...
//     uint64 ll_gpu = precomputed_top+1;
//     uint64 ul_gpu = ulimit;
    cutilSafeCall(cudaMemcpyToSymbol(d_precomputed_primes, precomputed_primes, SIXTYFOUR_KB));
    // allocate device memory
    byte* d_all_primes = 0;
    unsigned int num_bytes = (ulimit-llimit+1)/8;
    cutilSafeCall(cudaMalloc(&d_all_primes, num_bytes));
    // copy host memory to device
    for (unsigned int i = 0; i < num_bytes; i += 15015) {
        cutilSafeCall(cudaMemcpy(d_all_primes+i, precomputed_pattern, 15015, cudaMemcpyHostToDevice));
//         printf("i=%6u, i+15015=%6u\n", i, i+15015);
    }

    primeKernel_1<<<1, 1, 0>>>(llimit, ulimit, d_all_primes);

    byte* all_primes = 0;
    all_primes = (byte*)malloc(num_bytes);
    cutilSafeCall(cudaMemcpy(all_primes, d_all_primes, num_bytes, cudaMemcpyDeviceToHost));

    prime_counter = 0;
    for(uint64 i = llimit; i <= ulimit; i++)
        if(GET_BIT(all_primes, i-llimit))
            prime_counter++;
    printf("HOST: %llu primes found between [%llu, %llu]\n", prime_counter, llimit, ulimit);

    return 0;
}
