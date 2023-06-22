#ifndef __FFT_HS__

#define __FFT_HS__
#include "baseband_lib.h"

/**
 * Necessary structure for pthreads
 * when sending arguments to the
 * child thread
 */
struct args_ifft {
	int fft_id;
	int n;
	int hw_fft_busy;
	comp_t *fdata;
};

struct args_fft {
	int fft_id;
	comp_t *fdata;
	int n;
	int hw_fft_busy;
};

#ifndef THREAD_PER_TASK
void ifft_hs(int fft_id, comp_t fdata[], int n, int hw_fft_busy);
#else
void *ifft_hs(void *input);
#endif

#ifndef THREAD_PER_TASK
void fft_hs(int fft_id, comp_t fdata[], int n, int hw_fft_busy);
#else
void *fft_hs(void *input);
#endif

#endif
