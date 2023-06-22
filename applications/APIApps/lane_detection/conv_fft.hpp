// This implementation only works with grayscale images.
// Later add 3 dimensional extension.

// Transpose operation to convert image from (height, width, depth) to (depth, height, width)
//void conv_fft_transpose_hwd_dhw(double *in, double *out, int height, int width);

//


// Padding operation to match DASH_FFT's size constraint.
void conv_fft_pad_hw(double *in, double *out, int height, int width, int h, int w);

// Making input complex
void conv_fft_complex_hw(double *in, double *out, int height, int width);

// FFT2D operation
void conv_fft_fft2D_hw(double *in, double *out, int height, int width);

// Row by row conjugate op
void conv_fft_conj_hw(double *in, double *out, int height, int width, int mode);

// Element-wise multiplication
void conv_fft_mult(double *first_arr, double *second_arr, double *out, int height, int width);

// IFFT2D operation
void conv_fft_ifft2D_hw(double *in, double *out, int height, int width);

// Making input real
void conv_fft_real_hw(double *in, double *out, int height, int width);

// Crop op 
void conv_fft_crop_hw(double *in, double *out, int height, int width, int init_height, int init_width);

// Filter Conversion
void filter_conversion(double *filter, int height, int width, double *out);

// Conv wrapper
void conv(double *input, double *g_filter_1, double *g_filter_2, double *gx, double *gy, double *gx_out, double *gy_out, int height, int width, int g_size_1, int g_size_2, int g_size_3);
