#ifndef __QPSK_MOD_DEMOD_H__
#define __QPSK_MOD_DEMOD_H__

/**
 * Necessary structure for pthreads
 * when sending arguments to the 
 * child thread
 */
struct args_qpsk {
	int bitlen;
	unsigned char *bitstream;
	double *QPSK_real;
	double *QPSK_img;
	float *obuf;
};

struct args_qpsk_demod {
    int n;
    comp_t *ibuf;
    signed char *out;
};

#ifndef THREAD_PER_TASK
int  MOD_QPSK(int bitlen, unsigned char *bitstream, double *QPSK_real, double *QPSK_img, float *obuf);
#else
void * MOD_QPSK(void *input);
#endif

#ifndef THREAD_PER_TASK
int DeMOD_QPSK(int n, comp_t *ibuf, signed char *out);
#else
void * DeMOD_QPSK(void *input);
#endif

#endif
