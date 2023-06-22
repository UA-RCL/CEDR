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
    dash_re_flt_type *input_re = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width * kernel_size * kernel_size));
    dash_re_flt_type *input_im = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width * kernel_size * kernel_size));
    dash_re_flt_type *filter_im = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * kernel_size * kernel_size);

    dash_re_flt_type *out_re = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));
    dash_re_flt_type *out_im = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    for(int i = 0; i < height * width * kernel_size * kernel_size; i++)input_im[i] = 0;

    int x, y, z, index;
    z = kernel_size / 2;
    index = 0;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width ; j++){
            for(int k = 0; k < kernel_size; k++){
                for(int m = 0; m < kernel_size; m++){
                    x = i + k - z;
                    y = j + m - z;
                    if ((x >= 0 && x < height) && (y >= 0 && y < width)){
                        input_re[index++] = input[x * width + y];
                    }
                    else input_re[index++] = 0;
                }
            }
        }
    }

//    DASH_GEMM_flt(input_re, input_im, filter, filter_im, out, out_im, height * width, kernel_size * kernel_size, 1);
    DASH_GEMM_flt((dash_cmplx_flt_type *)input_re, (dash_cmplx_flt_type *)filter, (dash_cmplx_flt_type *)out, height * width, kernel_size * kernel_size, 1);

    free(input_re);
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
    size_t half_width = width/2;
    bool forwardTrans = true;

    dash_re_flt_type *row_row_fft_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    dash_cmplx_flt_type **fft_in_0; // height x (width/2)
    dash_cmplx_flt_type **fft_out_0; // height x (width/2)
    fft_in_0 = ((dash_cmplx_flt_type**) calloc(height, sizeof(dash_cmplx_flt_type*)));
    fft_out_0 = ((dash_cmplx_flt_type**) calloc(height, sizeof(dash_cmplx_flt_type*)));

    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    pthread_mutex_lock(barrier.mutex);

    // Row-by-row FFT
    for(int row = 0; row < height; row++){
	fft_in_0[row] = (dash_cmplx_flt_type *) &(in[row*width]);
	fft_out_0[row] = (dash_cmplx_flt_type *) &(row_row_fft_output[row*width]);
        DASH_FFT_flt_nb(&fft_in_0[row], &fft_out_0[row], &half_width, &forwardTrans, &barrier);
    }

    while (completion_ctr != height) {
        pthread_cond_wait(barrier.cond, barrier.mutex);
    }
    pthread_mutex_unlock(barrier.mutex);
    
    dash_re_flt_type *row_col_swap = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    // Moving complex part of the pixels from right to below for col-by-col FFT
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col+=2){
	    row_col_swap[(col/2)*height*2 + (row*2)] = row_row_fft_output[row*width + col];
	    row_col_swap[(col/2)*height*2 + (row*2) + 1] = row_row_fft_output[row*width + col + 1];
	}
    }

    dash_re_flt_type col_temp[width];
    dash_re_flt_type col_output[width];
    dash_re_flt_type *col_col_fft_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    cond = PTHREAD_COND_INITIALIZER;
    mutex = PTHREAD_MUTEX_INITIALIZER;
    completion_ctr = 0;
    barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    pthread_mutex_lock(barrier.mutex);

    // Col-by-col FFT
    for(int row = 0; row < height; row++){
        fft_in_0[row] = (dash_cmplx_flt_type *) &(row_col_swap[row*width]);
        fft_out_0[row] = (dash_cmplx_flt_type *) &(col_col_fft_output[row*width]);
        DASH_FFT_flt_nb(&fft_in_0[row], &fft_out_0[row], &half_width, &forwardTrans, &barrier);
    }

    while (completion_ctr != height) {
      pthread_cond_wait(barrier.cond, barrier.mutex);
    }
    pthread_mutex_unlock(barrier.mutex);
    
    // Moving complex part of the pixels from below to right
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col+=2){
	    out[(col/2)*height*2 + (row*2)] = col_col_fft_output[row*width + col];
	    out[(col/2)*height*2 + (row*2) + 1] = col_col_fft_output[row*width + col + 1];
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
// TODO: Use smaller ZIP sizes in non-blocking manner!   
    DASH_ZIP_flt((dash_cmplx_flt_type *)first_arr,(dash_cmplx_flt_type *)second_arr,(dash_cmplx_flt_type *)out,height*width/2,ZIP_MULT);
    /*for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col+=2){
            out[row * width + col] = first_arr[row * width + col] * second_arr[row * width + col] - first_arr[row * width + col + 1] * second_arr[row * width + col + 1];
            out[row * width + col + 1] = first_arr[row * width + col] * second_arr[row * width + col + 1] + first_arr[row * width + col + 1] * second_arr[row * width + col];
        }
    }*/
}


void conv_fft_ifft2D(dash_re_flt_type *in, dash_re_flt_type *out, int height, int width){
    size_t half_width = width/2;
    bool forwardTrans = false;
    
    dash_re_flt_type *row_row_fft_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    dash_cmplx_flt_type **fft_in_0; // height x (width/2)
    dash_cmplx_flt_type **fft_out_0; // height x (width/2)
    fft_in_0 = ((dash_cmplx_flt_type**) calloc(height, sizeof(dash_cmplx_flt_type*)));
    fft_out_0 = ((dash_cmplx_flt_type**) calloc(height, sizeof(dash_cmplx_flt_type*)));

    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    pthread_mutex_lock(barrier.mutex);

    // Row-by-row IFFT
    for(int row = 0; row < height; row++){
	fft_in_0[row] = (dash_cmplx_flt_type *) &(in[row*width]);
	fft_out_0[row] = (dash_cmplx_flt_type *) &(row_row_fft_output[row*width]);
        DASH_FFT_flt_nb(&fft_in_0[row], &fft_out_0[row], &half_width, &forwardTrans, &barrier);
    }

    while (completion_ctr != height) {
        pthread_cond_wait(barrier.cond, barrier.mutex);
    }
    pthread_mutex_unlock(barrier.mutex);

    // Moving complex part of the pixels from right to below for col-by-col IFFT
    dash_re_flt_type *row_col_swap = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col+=2){
	    row_col_swap[(col/2)*height*2 + (row*2)] = row_row_fft_output[row*width + col];
	    row_col_swap[(col/2)*height*2 + (row*2) + 1] = row_row_fft_output[row*width + col + 1];
        }
    }

    dash_re_flt_type *col_col_fft_output = (dash_re_flt_type *) malloc (sizeof (dash_re_flt_type) * (height * width));

    cond = PTHREAD_COND_INITIALIZER;
    mutex = PTHREAD_MUTEX_INITIALIZER;
    completion_ctr = 0;
    barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    pthread_mutex_lock(barrier.mutex);

    // Col-by-col IFFT
    for(int row = 0; row < height; row++){
        fft_in_0[row] = (dash_cmplx_flt_type *) &(row_col_swap[row*width]);
        fft_out_0[row] = (dash_cmplx_flt_type *) &(col_col_fft_output[row*width]);
        DASH_FFT_flt_nb(&fft_in_0[row], &fft_out_0[row], &half_width, &forwardTrans, &barrier);
    }

    while (completion_ctr != height) {
        pthread_cond_wait(barrier.cond, barrier.mutex);
    }
    pthread_mutex_unlock(barrier.mutex);

    // Moving complex part of the pixels from below to right
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col+=2){
	    out[(col/2)*height*2 + (row*2)] = col_col_fft_output[row*width + col];
	    out[(col/2)*height*2 + (row*2) + 1] = col_col_fft_output[row*width + col + 1];
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

