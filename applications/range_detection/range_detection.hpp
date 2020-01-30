#pragma once

#include <fftw3.h>
//Enforce inclusion after fftw3 to guide how types are defined for fftw_complex
#include <complex.h>

struct range_detect_fields{
	size_t n_samples;
	size_t time_n_samples;
	double T;
	double B;
	float sampling_rate;
	double *time;
	float *received;
	float *lfm_waveform;
	fftwf_complex *in_xcorr1, *out_xcorr1, *in_xcorr2, *out_xcorr2, *in_xcorr3, *out_xcorr3;
	fftwf_plan p1, p2, p3;
	float *X1, *X2;
	float *corr_freq;
	float *corr;
	float lag;
	float max_corr;
	float index;
};

extern "C" void range_detect_nop(task_nodes *task);
extern "C" void range_detect_LFM(task_nodes *task);
extern "C" void range_detect_FFT_0(task_nodes *task);
extern "C" void range_detect_FFT_1(task_nodes *task);
extern "C" void range_detect_MUL(task_nodes *task);
extern "C" void range_detect_IFFT(task_nodes *task);
extern "C" void range_detect_MAX(task_nodes *task);
extern "C" void range_detection_init(dag_app *lag_detect);