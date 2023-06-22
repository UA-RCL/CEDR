#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

void waveform_gen(double *, double, double, double *, size_t);

void waveform_gen(double *time, double B, double T, double *lfm_waveform, size_t n_samples) {
	for (size_t i = 0; i < 2 * n_samples; i += 2) {
		lfm_waveform[i] = creal(cexp(I * M_PI * B / T * pow(time[i / 2], 2)));
		lfm_waveform[i + 1] = cimag(cexp(I * M_PI * B / T * pow(time[i / 2], 2)));
	}
}
