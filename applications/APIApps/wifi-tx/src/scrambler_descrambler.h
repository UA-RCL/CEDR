#ifndef __SCRAM_DESCRAM_H__
#define __SCRAM_DESCRAM_H__

/* function prototypes */
void scrambler(int inlen, unsigned char ibuf[], unsigned char obuf[]);
void descrambler(int inlen, unsigned char ibuf[], unsigned char obuf[]);

#endif 
