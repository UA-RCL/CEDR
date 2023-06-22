#ifndef __QPSK_MOD_DEMOD_H__
#define __QPSK_MOD_DEMOD_H__

int  MOD_QPSK(int bitlen, unsigned char *bitstream, double *QPSK_real, double *QPSK_img, float *obuf);

int DeMOD_QPSK(int n, comp_t *ibuf, signed char *out);

#endif
