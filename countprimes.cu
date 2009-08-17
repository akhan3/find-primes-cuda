#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// #include <cutil_inline.h>
#include <countprimes_kernel.cu>


#define SET_BIT(array, i) array[i>>3] |= 1<<(i&0x07)
#define CLR_BIT(array, i) array[i>>3] &= ~(1<<(i&0x07))
#define GET_BIT(array, i) (array[i>>3] & 1<<(i&0x07)) >> (i&0x07)

typedef  unsigned long long uint64;
typedef  unsigned char byte;

void findPrimes (uint64 topCandidate, byte* array) {
    CLR_BIT(array, 0);
    CLR_BIT(array, 1);
    for(uint64 i = 2; i < topCandidate+1; i++)
        SET_BIT(array, i);
    uint64 thisFactor = 2;
    while(thisFactor * thisFactor <= topCandidate) {
        uint64 mark = thisFactor + thisFactor;
        while(mark <= topCandidate) {
            CLR_BIT(array, mark);
            mark += thisFactor;
        }
        // Search for the next prime divisor
        do thisFactor++; while(GET_BIT(array, thisFactor) == 0);
        assert(thisFactor <= topCandidate);
    }
}

int main(int argc, char *argv[]) {
    uint64 topCandidate = 100;
    if(argc > 1) {
        float N_float;
        sscanf(argv[1], "%f", &N_float);
        topCandidate  = (uint64)N_float;
    }

    topCandidate = 65536*8-1;     // 524287
    byte precomputed_primes[65536 * sizeof(byte)];
    findPrimes(topCandidate, precomputed_primes);

    unsigned int prime_counter = 0;
    for(uint64 i = 0; i <= topCandidate; i++) {
        if(GET_BIT(precomputed_primes, i)) {
            prime_counter++;
//             printf("%llu\n", i);
        }
    }
    printf("%u primes found between 0 and %llu\n", prime_counter, topCandidate);

//     cutilSafeCall(cudaMemcpyToSymbol(d_bitarray, precomputed_primes, 65536));




    return 0;
}
