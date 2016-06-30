#ifndef _MYDEFS_H_
#define _MYDEFS_H_

#define SET_BIT(array, i) (array)[(i)>>3] |= 1<<((i)&0x07)
#define CLR_BIT(array, i) (array)[(i)>>3] &= ~(1<<((i)&0x07))
#define GET_BIT(array, i) ((array)[(i)>>3] & 1<<((i)&0x07)) >> ((i)&0x07)

#define SET_BYTE(array, i) array[i] = 1
#define CLR_BYTE(array, i) array[i] = 0
#define GET_BYTE(array, i) (byte)array[i]

#define CONSTANT_MEM_SIZE (uint64)65536

typedef  unsigned long long uint64;
typedef  unsigned int uint32;
typedef  unsigned short uint16;
typedef  unsigned char byte;


#endif // #ifndef _MYDEFS_H_
