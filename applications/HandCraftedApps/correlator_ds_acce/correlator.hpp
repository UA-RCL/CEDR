#pragma once

//extern "C" void fftwf_fft(float *input_array, fftwf_complex *in, fftwf_complex *out, float *output_array, size_t n_elements, fftwf_plan p );
extern "C" void RD_head_node(void);
extern "C" void RD_LFM(void);
extern "C" void RD_FFT0(void);
extern "C" void RD_FFT1(void);
extern "C" void RD_MUL(void);
extern "C" void RD_IFFT(void);
extern "C" void RD_MAX(void);




