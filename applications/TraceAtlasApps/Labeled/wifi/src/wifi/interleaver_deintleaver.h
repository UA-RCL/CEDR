#ifndef __INTER_DEINTER_H__
#define __INTER_DEINTER_H__

/**
 * Necessary structure for pthreads
 * when sending arguments to the 
 * child thread
 */
struct args_interleaver {
	int N;
	unsigned char *datain;
	unsigned char *top1;
};

struct args_deinterleaver {
    signed char *datain;
    int N;
    unsigned char *top2;
};

/* function prototypes */

#ifndef THREAD_PER_TASK
void interleaver(signed char datain[],int N, unsigned char top1[]);
#else
void * interleaver(void *input);
#endif

#ifndef THREAD_PER_TASK
int deinterleaver(signed char data[], int N, signed char top2[]);
#else
void* deinterleaver(void *input);
#endif

#endif

