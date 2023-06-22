#ifndef __FFT_HS__

#define __FFT_HS__
#include "baseband_lib.h"

void fft_hs (int fft_id, comp_t fdata[], int n, int hw_fft_busy);
void ifft_hs(int fft_id, comp_t idata[], int n, int hw_fft_busy);

#endif
