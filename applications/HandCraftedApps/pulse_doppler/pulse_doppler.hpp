#pragma once

#include <fftw3.h>
//Enforce inclusion after fftw3 to guide how types are defined for fftw_complex
#include <complex.h>

extern "C" void pulse_doppler_nop(void);
extern "C" void pulse_doppler_FFT_0_cpu(void);
extern "C" void pulse_doppler_FFT_0_accel(void);
extern "C" void pulse_doppler_FFT_1_cpu(void);
extern "C" void pulse_doppler_FFT_1_accel(void);
extern "C" void pulse_doppler_MUL(void);
extern "C" void pulse_doppler_IFFT_cpu(void);
extern "C" void pulse_doppler_IFFT_accel(void);
extern "C" void pulse_doppler_REALIGN_MAT(void);
extern "C" void pulse_doppler_FFT_2_cpu(void);
extern "C" void pulse_doppler_FFT_2_accel(void);
extern "C" void pulse_doppler_AMPLITUDE(void);
extern "C" void pulse_doppler_FFTSHIFT(void);
extern "C" void pulse_doppler_MAX0(void);
extern "C" void pulse_doppler_FINAL_MAX(void);
extern "C" void pulse_doppler_init(void);