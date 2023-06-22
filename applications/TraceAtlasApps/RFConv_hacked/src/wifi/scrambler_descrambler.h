#ifndef __SCRAM_DESCRAM_H__
#define __SCRAM_DESCRAM_H__

/**
 * Necessary structure for pthreads
 * when sending arguments to the
 * child thread
 */
struct args_scrambler {
	int inlen;
	unsigned char *ibuf;
	unsigned char *obuf;
};

struct args_descrambler {
	int inlen;
	unsigned char *ibuf;
	unsigned char *obuf;
};

/* function prototypes */
#ifndef THREAD_PER_TASK
void descrambler(int inlen, unsigned char ibuf[], unsigned char obuf[]);
#else
void *descrambler(void *input);
#endif

#ifndef THREAD_PER_TASK
void scrambler(int inlen, unsigned char ibuf[], unsigned char obuf[]);
#else
void *scrambler(void *input);
#endif

#endif
