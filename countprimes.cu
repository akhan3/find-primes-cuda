#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <cutil_inline.h>
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

// void findPrimes (uint64 ll, uint64 ul, byte* precomputed_primes, uint64 precomputed_top, byte* array) {
//     for(uint64 i = ll; i <= ul; i++)
//         SET_BIT(array, i);
//     for(uint64 i = 1; i <= precomputed_top; i++)
//         SET_BIT(array, i);
//
//     uint64 thisFactor = 2;
//     while(thisFactor * thisFactor <= topCandidate) {
//         uint64 mark = thisFactor + thisFactor;
//         while(mark <= topCandidate) {
//             CLR_BIT(array, mark);
//             mark += thisFactor;
//         }
//         // Search for the next prime divisor
//         do thisFactor++; while(GET_BIT(array, thisFactor) == 0);
//         assert(thisFactor <= topCandidate);
//     }
// }

int main(int argc, char *argv[]) {
    uint64 ll, ul;  // upper and lower limits, both inclusive
    float ll_float, ul_float;
    assert(argc == 3);
    sscanf(argv[1], "%f", &ll_float);
    sscanf(argv[2], "%f", &ul_float);
    ll = (uint64)ll_float;
    ul = (uint64)ul_float;
    assert(ll <= ul);
    assert(ul <= 274877906944);
    printf("Counting primes in the interval [%llu, %llu]...\n", ll, ul);

    uint64 precomputed_top = 65536*8-1;    // 524287
    byte precomputed_primes[65536 * sizeof(byte)];
    findPrimes(precomputed_top, &precomputed_primes[0]);

    uint64 prime_counter = 0;

    if(ul <= precomputed_top) {
        printf("No need to use GPU...\n");
        prime_counter = 0;
        for(uint64 i = ll; i <= ul; i++)
            if(GET_BIT(precomputed_primes, i))
                prime_counter++;
        printf("%llu primes found between [%llu, %llu]\n", prime_counter, ll, ul);
        return 0;
    }

    if(ll <= precomputed_top) {
        printf("counting some primes from the precomputed list...\n");
        prime_counter = 0;
        for(uint64 i = ll; i <= precomputed_top; i++)
            if(GET_BIT(precomputed_primes, i))
                prime_counter++;
        printf("%llu primes found between [%llu, %llu]\n", prime_counter, ll, precomputed_top);
    }

    // now using GPU...
    uint64 ll_gpu = precomputed_top+1;
    uint64 ul_gpu = ul;
    cutilSafeCall(cudaMemcpyToSymbol(d_bitarray, precomputed_primes, 65536));



    return 0;
}
