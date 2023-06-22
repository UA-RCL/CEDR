#include "conv.hpp"
//#include "dash.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

void conv_serial(dash_re_flt_type *in, dash_re_flt_type *filter, dash_re_flt_type *out, int height, int width, int kernel_size){
    int x, y, z;
    z = kernel_size / 2;
	for(int i = 0; i < height; i++){
	    for(int j = 0; j < width ; j++){
		    dash_re_flt_type sum = 0.0;
			for(int k = 0; k < kernel_size; k++){
				for(int m = 0; m < kernel_size; m++){
					x = i + k - z;
					y = j + m - z;
					if ((x >= 0 && x < height) && (y >= 0 && y < width)) sum += in[x * width + y] * filter[k * kernel_size + m];
				}
			}   
		    out[i * width + j] = sum;
		}
	}
}

void conv_zip(dash_re_flt_type *in, dash_re_flt_type *filter, dash_re_flt_type *out, int height, int width, int kernel_size){
    int x, y, z, index;
    z = kernel_size / 2;
    dash_re_flt_type temp_img_holder[kernel_size * kernel_size];
    dash_re_flt_type temp_out_holder[kernel_size * kernel_size];

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width ; j++){
            index = 0;
            for(int k = 0; k < kernel_size; k++){
                for(int m = 0; m < kernel_size; m++){
                    x = i + k - z;
                    y = j + m - z;
                    if ((x >= 0 && x < height) && (y >= 0 && y < width))temp_img_holder[index++] = in[x * width + y];
                    else temp_img_holder[index++] = 0;
                }
            }
            DASH_ZIP_flt((dash_cmplx_flt_type *)temp_img_holder, (dash_cmplx_flt_type *)filter, (dash_cmplx_flt_type *)temp_out_holder, kernel_size * kernel_size, ZIP_MULT);
            dash_re_flt_type sum = 0;
            for(int s = 0; s < kernel_size * kernel_size; s++){
                sum += temp_out_holder[s];
            }
            out[i * width + j] = sum;
        }
    }
}

void conv_gemm(dash_re_flt_type *input, dash_re_flt_type *filter, dash_re_flt_type *out, int height, int width, int kernel_size){
    dash_re_flt_type *input_im = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width * kernel_size * kernel_size));
    dash_re_flt_type *filter_im = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * kernel_size * kernel_size);

    dash_re_flt_type *out_re = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));
    dash_re_flt_type *out_im = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    dash_re_flt_type *input_gemm = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width * kernel_size * kernel_size * 2));
    dash_re_flt_type *filter_gemm = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (kernel_size * kernel_size * 2));
    dash_re_flt_type *out_gemm = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width * 2));

    for(int i = 0; i < height * width * kernel_size * kernel_size; i++)input_im[i] = 0;

    int x, y, z, index;
    z = kernel_size / 2;
    index = 0;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width ; j++){
            for(int k = 0; k < kernel_size; k++){
                for(int m = 0; m < kernel_size * 2; m++){
                    x = i + k - z;
                    y = j + m - z;
                    if(index % 2 == 0){
                        if ((x >= 0 && x < height) && (y >= 0 && y < width)){
                            input_gemm[index++] = input[x * width + y];
                        }else{
                            input_gemm[index++] = 0;
                        } 
                    }else{
                        input_gemm[index++] = 0;
                    }
                }
            }
       }
    }

    for(int i = 0; i < (kernel_size * kernel_size * 2); i++){
        if(i % 2 == 0){
            filter_gemm[i] = filter[i / 2];
        }else{
            filter_gemm[i] = 0;
        }
    }

//    DASH_GEMM_flt(input_re, input_im, filter, filter_im, out, out_im, height * width, kernel_size * kernel_size, 1);
    DASH_GEMM_flt((dash_cmplx_flt_type *)input_gemm, (dash_cmplx_flt_type *)filter_gemm, (dash_cmplx_flt_type *)out_gemm, height * width, kernel_size * kernel_size, 1);

    for(int i = 0; i < (height * width * 2); i++){
        if(i % 2 == 0){
            out[i / 2] = out_gemm[i];
        }
    }

    free(input_gemm);
    free(input_im);
    free(filter_im);
    free(out_re);
    free(out_im);
}

// Padding both input and filter to FFT compatible size
void conv_fft_pad(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width, int h, int w){    
    for(int row = 0; row < h; row++){
        for(int col = 0; col < w; col++){
            out[row * w + col] = 0;
            if(row < height && col < width){
                out[row * w + col] = in[row * width + col];
            }
        }
    }
}

void conv_fft_pad_filter(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width, int h, int w){    
    for(int row = 0; row < h; row++){
        for(int col = 0; col < w; col++){
            out[row * w + col] = 0;
            if(row < height && col < width){
                out[row * w + col] = in[row * width + col];
            }
        }
    }
}

// Making input and filter complex by adding 0 at n%2==1 indexes 
void conv_fft_complex(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width){
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            out[(row * width +  col) * 2] = in[row * width +  col];
            out[(row * width +  col) * 2 + 1] = 0;
        }
    }
}

// 2D forward FFT
void conv_fft_fft2D(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width){
    
    int count = 0;

    dash_re_flt_type row_temp[width];
    dash_re_flt_type row_output[width];
    dash_re_flt_type *row_row_fft_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    

    // Row-by-row FFT
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            row_temp[col] = in[row * width + col];
        }
        DASH_FFT_flt((dash_cmplx_flt_type *)row_temp, (dash_cmplx_flt_type *)row_output, width / 2, true);
        count++;
        for(int col = 0; col < width; col++){
            row_row_fft_output[row * width + col] = row_output[col];
        }
    }

    dash_re_flt_type *row_col_swap = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    // Moving complex part of the pixels from right to below for col-by-col FFT
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            if(col % 2 == 0){
                row_col_swap[(2 * row) * (width / 2) + (col / 2)] = row_row_fft_output[row * width + col];
            }
            else{
                row_col_swap[(2 * row + 1) * (width / 2) + (col / 2)] = row_row_fft_output[row * width + col];
            } 
        }
    }

    dash_re_flt_type col_temp[width];
    dash_re_flt_type col_output[width];
    dash_re_flt_type *col_col_fft_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    // Col-by-col FFT
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            col_temp[col] = row_col_swap[col * height + row];
        }
        DASH_FFT_flt((dash_cmplx_flt_type *)col_temp, (dash_cmplx_flt_type *)col_output, width / 2, true);
        count++;
        for(int col = 0; col < width; col++){
            col_col_fft_output[col * height + row] = col_output[col];
        }
    }

    
    // Moving complex part of the pixels from below to right
    for(int row = 0; row < width; row++){
        for(int col = 0; col < height; col++){
            if(row % 2 == 0){
                out[(row / 2) * (height * 2) + (2 * col)] = col_col_fft_output[row * height + col];
	    }
            else{
                out[(row / 2) * (height * 2) + (2 * col + 1)] = col_col_fft_output[row * height + col];
	    }
        }
    }
    free(row_row_fft_output);
    free(row_col_swap);
    free(col_col_fft_output);
}

// Conjugation operation, complex parts are multiplied by - 1
void conv_fft_conj(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width, int mode){    
    if(mode == 0){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                if((i * width + j) % 2 == 1 && in[i * width + j] != 0){
                    out[i * width + j] = -in[i * width + j];
		}
                else{
                    out[i * width + j] = in[i * width + j];
		}
            }
        }
    }else{
        for(int i = 0; i < width; i++){
            for(int j = 0; j < height / 2; j++){
                out[(2 * j) * width + i] = in[(2 * j) * width + i];
                out[(2 * j + 1) * width + i] = -in[(2 * j + 1) * width + i];
            }
        }
    }
}

// Multiplication of input and filter in frequency domain
void conv_fft_mult(dash_re_flt_type *first_arr, dash_re_flt_type *second_arr, dash_re_flt_type *out, int height, int width){        
    
    DASH_ZIP_flt((dash_cmplx_flt_type *)first_arr,(dash_cmplx_flt_type *)second_arr,(dash_cmplx_flt_type *)out,height*width/2,ZIP_MULT);
    /*for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col+=2){
            out[row * width + col] = first_arr[row * width + col] * second_arr[row * width + col] - first_arr[row * width + col + 1] * second_arr[row * width + col + 1];
            out[row * width + col + 1] = first_arr[row * width + col] * second_arr[row * width + col + 1] + first_arr[row * width + col + 1] * second_arr[row * width + col];
        }
    }*/
}


void conv_fft_ifft2D(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width){
    int count = 0;
    
    dash_re_flt_type row_temp[width];
    dash_re_flt_type row_output[width];
    dash_re_flt_type *row_row_fft_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    // Row-by-row IFFT
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            row_temp[col] = in[row * width + col];
        }
        count++;
        DASH_FFT_flt((dash_cmplx_flt_type *)row_temp, (dash_cmplx_flt_type *)row_output, width / 2, false);
        for(int col = 0; col < width; col++){
            row_row_fft_output[row * width + col] = row_output[col];
        }
    }

    // Moving complex part of the pixels from right to below for col-by-col IFFT
    dash_re_flt_type *row_col_swap = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            if(col % 2 == 0){
                row_col_swap[(2 * row) * (width / 2) + (col / 2)] = row_row_fft_output[row * width + col];
            }
            else{
                row_col_swap[(2 * row + 1) * (width / 2) + (col / 2)] = row_row_fft_output[row * width + col];
            } 
        }
    }

    dash_re_flt_type col_temp[height];
    dash_re_flt_type col_output[height];
    dash_re_flt_type *col_col_fft_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    // Col-by-col IFFT
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            col_temp[col] = row_col_swap[col * height + row];
        }
        DASH_FFT_flt((dash_cmplx_flt_type *)col_temp, (dash_cmplx_flt_type *)col_output, width / 2, false);
        count++;
        for(int col = 0; col < width; col++){
            col_col_fft_output[col * height + row] = col_output[col];
        }
    }

    // Moving complex part of the pixels from below to right
    for(int row = 0; row < width; row++){
        for(int col = 0; col < height; col++){
            if(row % 2 == 0)out[(row / 2) * (height * 2) + (2 * col)] = col_col_fft_output[row * height + col];
            else out[(row / 2) * (height * 2) + (2 * col + 1)] = col_col_fft_output[row * height + col];
        }
    }
    free(row_row_fft_output);
    free(row_col_swap);
    free(col_col_fft_output);
}

// Moving complex filters back to real 
void conv_fft_real(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width){
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width / 2; j++){
            out[i * (width / 2) + j] = in[(i * width + j * 2)];
        }
    }
}

// Cropping result back to original size
void conv_fft_crop(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width, int init_height, int init_width){
    for(int i = 0; i < init_height; i++){
        for(int j = 0; j < init_width; j++){
            out[i * init_width + j] = in[i * width + j];
        }
    }
}

// FFT based convolution
void conv_fft(dash_re_flt_type *input, dash_re_flt_type *filter, dash_re_flt_type *out, int height, int width, int kernel_size){
    int pad = kernel_size / 2;
    int temp_height = height + (pad * 2);
    int temp_width = width + (pad * 2);
    dash_re_flt_type *temp_input = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (temp_height * temp_width));

    for(int i = 0; i < temp_height; i++){
        for(int j = 0; j < temp_width; j++){
            if(i < pad || j < pad || i >= height + pad || j >= width + pad)temp_input[i * temp_width + j] = 0;
            else temp_input[i * temp_width + j] = input[((i - pad) * width + (j - pad))];
        }
    }

    int pad_h = pow(2, ceil(log2(temp_height)));
    int pad_w = pow(2, ceil(log2(temp_width)));
    dash_re_flt_type *padded_filter = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_h * pad_w));
    dash_re_flt_type *padded_input = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_h * pad_w));
    conv_fft_pad_filter(filter, padded_filter, kernel_size, kernel_size, pad_h, pad_w);
    conv_fft_pad(temp_input, padded_input, temp_height, temp_width, pad_h, pad_w);


    int complex_w = pad_w * 2;
    dash_re_flt_type *complex_filter = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_h * complex_w));
    dash_re_flt_type *complex_input = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_h * complex_w));
    conv_fft_complex(padded_filter, complex_filter, pad_h, pad_w);
    conv_fft_complex(padded_input, complex_input, pad_h, pad_w);

    dash_re_flt_type *fft_filter = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_h * complex_w));
    dash_re_flt_type *fft_input = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_h * complex_w));
    conv_fft_fft2D(complex_filter, fft_filter, pad_h, complex_w);
    conv_fft_fft2D(complex_input, fft_input, pad_h, complex_w);

    dash_re_flt_type *conj_filter = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_h * complex_w));
    conv_fft_conj(fft_filter, conj_filter, pad_h, complex_w, 0);

    dash_re_flt_type *mult_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_h * complex_w));
    conv_fft_mult(fft_input, conj_filter, mult_output, pad_h, complex_w);

    dash_re_flt_type *ifft_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_h * complex_w));
    conv_fft_ifft2D(mult_output, ifft_output, pad_h, complex_w);

    dash_re_flt_type *real_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (pad_h * (complex_w / 2)));
    conv_fft_real(ifft_output, real_output, pad_h, complex_w);

    conv_fft_crop(real_output, out, pad_h, complex_w / 2, height, width);

    free(padded_filter);
    free(padded_input);
    free(complex_filter);
    free(complex_input);
    free(fft_filter);
    free(fft_input);
    free(conj_filter);
    free(mult_output);
    free(ifft_output);
    free(real_output);
}

