#pragma once

#include <fftw3.h>
//Enforce inclusion after fftw3 to guide how types are defined for fftw_complex
#include <complex.h>

struct pulse_doppler_fields {
    size_t m ;                               // number of pulses
    size_t n_samples ;
    float PRI;
    float *mf;
    float *p;
    float *pulse;
    float *corr;

    fftwf_complex *in_xcorr1, *out_xcorr1, *in_xcorr2, *out_xcorr2, *in_xcorr3, *out_xcorr3;
    fftwf_plan *p1, *p2, *p3;
    float *q;
    float *r;
    float *f;
    float *max, *a, *b;
    fftwf_complex *in_fft, *out_fft;
    fftwf_plan *p4;

    float *X1, *X2;
    float *corr_freq;
};

extern "C" void pulse_doppler_nop(task_nodes *task);
extern "C" void pulse_doppler_FFT_0(task_nodes *task);
extern "C" void pulse_doppler_FFT_1(task_nodes *task);
extern "C" void pulse_doppler_MUL(task_nodes *task);
extern "C" void pulse_doppler_IFFT(task_nodes *task);
extern "C" void pulse_doppler_REALIGN_MAT(task_nodes *task);
extern "C" void pulse_doppler_FFT_2(task_nodes *task);
extern "C" void pulse_doppler_AMPLITUDE(task_nodes *task);
extern "C" void pulse_doppler_FFTSHIFT(task_nodes *task);
extern "C" void pulse_doppler_MAX0(task_nodes *task);
extern "C" void pulse_doppler_FINAL_MAX(task_nodes *task);
extern "C" void pulse_doppler_init(dag_app *pulse_doppler);