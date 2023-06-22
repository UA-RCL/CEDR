#include "dash.h"

// This file includes all convolution implementation making use of different accelerators.

// Naive serial convolution
void conv_serial(dash_re_flt_type *input, dash_re_flt_type *filter, dash_re_flt_type *out, int height, int width, int kernel_size);

// ZIP based implementation
void conv_zip(dash_re_flt_type *input, dash_re_flt_type *filter, dash_re_flt_type *out, int height, int width, int kernel_size);

// GEMM based implementation
void conv_gemm(dash_re_flt_type *input, dash_re_flt_type *filter, dash_re_flt_type *out, int height, int width, int kernel_size);

// Below is for FFT based convolution
// Padding operation to match DASH_FFT's size constraint.
void conv_fft_pad(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width, int h, int w);

// Making input complex
void conv_fft_complex(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width);

// FFT2D operation
void conv_fft_fft2D(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width);

// Row by row conjugate op
void conv_fft_conj(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width, int mode);

// Element-wise multiplication
void conv_fft_mult(dash_re_flt_type *first_arr, dash_re_flt_type *second_arr, dash_re_flt_type *out, int height, int width);

// IFFT2D operation
void conv_fft_ifft2D(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width);

// Making input real
void conv_fft_real(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width);

// Crop op 
void conv_fft_crop(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width, int init_height, int init_width);

// Conv wrapper
void conv_fft(dash_re_flt_type *input, dash_re_flt_type *filter, dash_re_flt_type *out, int height, int width, int kernel_size);
