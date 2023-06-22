#ifndef __FFT_HS__

#define __FFT_HS__
#include "baseband_lib.h"

void fft_hs_PD(int fft_id, float* fft_in, float* fft_out, int n);
void ifft_hs_PD(int fft_id, float* fft_in, float* fft_out, int n);

#endif
