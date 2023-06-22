#include "conv_fft.hpp"
#include "dash.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

// Print helper for matrix
void print(double *arr, int size0, int size1, std::string name){
    printf("%s\n", name.c_str());
    for(int i = 0; i < size0; i++){
        for(int j = 0; j < size1; j++){
            printf("%f ", arr[i * size1 + j]);
        }
        printf("\n");
    }
        printf("\n");
}

// Padding both input and filter to FFT compatible size
void conv_fft_pad_hw(double *in, double *out, int height, int width, int h, int w){
    for(int row = 0; row < h; row++){
        for(int col = 0; col < w; col++){
            if(row < height && col < width){
                out[row * w + col] = in[row * width + col];
            }else{
                out[row * w + col] = 0;
            }
        }
    }
}

// Making input and filter by adding 0 at n%2==1 indexes 
void conv_fft_complex_hw(double *in, double *out, int height, int width){
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            out[(row * width +  col) * 2] = in[row * width +  col];
            out[(row * width +  col) * 2 + 1] = 0;
        }
    }
}

// 2D forward FFT
void conv_fft_fft2D_hw(double *in, double *out, int height, int width){
    double row_temp[width];
    double row_output[width];
    double *row_row_fft_output = (double *) malloc (sizeof (double) * (height * width));

    // Row-by-row FFT
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            row_temp[col] = in[row * width + col];
        }
        DASH_FFT(row_temp, row_output, width / 2, true);
        for(int col = 0; col < width; col++){
            row_row_fft_output[row * width + col] = row_output[col];
        }
    }

    double *row_col_swap = (double *) malloc (sizeof (double) * (height * width));

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

    double col_temp[width];
    double col_output[width];
    double *col_col_fft_output = (double *) malloc (sizeof (double) * (height * width));

    // Col-by-col FFT
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            col_temp[col] = row_col_swap[col * height + row];
        }
        DASH_FFT(col_temp, col_output, width / 2, true);
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
    free(col_col_fft_output);
    free(row_row_fft_output);
    free(row_col_swap);
}

// Conjugation operation, complex parts are multiplied by - 1
void conv_fft_conj_hw(double *in, double *out, int height, int width, int mode){    
    if(mode == 0){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                if(((i * width + j) % 2 == 1) && (in[i * width + j] != 0))
                    out[i * width + j] = -in[i * width + j];
                else out[i * width + j] = in[i * width + j];

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
void conv_fft_mult(double *first_arr, double *second_arr, double *out, int height, int width){
//    DASH_ZIP(first_arr,second_arr,out,height*width/2,ZIP_CMP_MULT);
    int index;
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col+=2){
            index = row * width + col;
            out[index] = first_arr[index] * second_arr[index] - first_arr[index + 1] * second_arr[index + 1];
            out[index + 1] = first_arr[index] * second_arr[index + 1] + first_arr[index + 1] * second_arr[index];
        }
    }
}


void conv_fft_ifft2D_hw(double *in, double *out, int height, int width){
    double row_temp[width];
    double row_output[width];
    double *row_row_fft_output = (double *) malloc (sizeof (double) * (height * width));

    // Row-by-row IFFT
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            row_temp[col] = in[row * width + col];
        }
        
        DASH_FFT(row_temp, row_output, width / 2, false);
	for(int col = 0; col < width; col++){
            row_row_fft_output[row * width + col] = row_output[col];
        }
    }

    
    // Moving complex part of the pixels from right to below for col-by-col IFFT
    double *row_col_swap = (double *) malloc (sizeof (double) * (height * width));
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


    double col_temp[height];
    double col_output[height];
    double *col_col_fft_output = (double *) malloc (sizeof (double) * (height * width));

    // Col-by-col IFFT
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            col_temp[col] = row_col_swap[col * height + row];
        }
        DASH_FFT(col_temp, col_output, width / 2, false);
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
    free(col_col_fft_output);
    free(row_row_fft_output);
    free(row_col_swap);
}

// Moving complex filters back to real 
void conv_fft_real_hw(double *in, double *out, int height, int width){
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width / 2; j++){
            out[i * (width / 2) + j] = in[(i * width + j * 2)];
        }
    }

}

// Cropping result back to original size
void conv_fft_crop_hw(double *in, double *out, int height, int width, int init_height, int init_width){
    for(int i = 0; i < init_height; i++){
        for(int j = 0; j < init_width; j++){
            out[i * init_width + j] = in[i * width + j];
        }
    }
}

void filter_conversion(double *filter, int height, int width, int kernel_size, double *out){
    int pad_h = pow(2, ceil(log2(height)));
    int pad_w = pow(2, ceil(log2(width)));
    double *padded_filter = (double *) malloc (sizeof (double) * (pad_h * pad_w));
    conv_fft_pad_hw(filter, padded_filter, kernel_size, kernel_size, pad_h, pad_w);

    int complex_w = pad_w * 2;
    double *complex_filter = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    conv_fft_complex_hw(padded_filter, complex_filter, pad_h, pad_w);

    double *fft_filter = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    conv_fft_fft2D_hw(complex_filter, fft_filter, pad_h, complex_w);

    conv_fft_conj_hw(fft_filter, out, pad_h, complex_w, 0);

    free(padded_filter);
    free(complex_filter);
    free(fft_filter);
}

void conv(double *input, double *g_filter_1, double *g_filter_2, double *gx, double *gy, double *gx_out, double *gy_out, int height, int width, int g_size_1, int g_size_2, int g_size_3){
    
    int pad_h = pow(2, ceil(log2(height)));
    int pad_w = pow(2, ceil(log2(width)));
    double *padded_input = (double *) malloc (sizeof (double) * (pad_h * pad_w));
    conv_fft_pad_hw(input, padded_input, height, width, pad_h, pad_w);

    int complex_w = pad_w * 2;
    double *complex_input = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    conv_fft_complex_hw(padded_input, complex_input, pad_h, pad_w);

    free(padded_input);

    double *fft_input = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    conv_fft_fft2D_hw(complex_input, fft_input, pad_h, complex_w);

    free(complex_input);

    double *conj_filter_g_1 = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    double *conj_filter_g_2 = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    double *conj_filter_gx = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    double *conj_filter_gy = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    filter_conversion(g_filter_1, height, width, g_size_1, conj_filter_g_1);
    filter_conversion(g_filter_2, height, width, g_size_2, conj_filter_g_2);
    filter_conversion(gx, height, width, g_size_3, conj_filter_gx);
    filter_conversion(gy, height, width, g_size_3, conj_filter_gy);

    double *mult_output_g_1 = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    double *mult_output_g_2 = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    double *mult_output_g_x = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    double *mult_output_g_y = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    conv_fft_mult(fft_input, conj_filter_g_1, mult_output_g_1, pad_h, complex_w);
    conv_fft_mult(mult_output_g_1, conj_filter_g_2, mult_output_g_2, pad_h, complex_w);
    conv_fft_mult(mult_output_g_2, conj_filter_gx, mult_output_g_x, pad_h, complex_w);
    conv_fft_mult(mult_output_g_2, conj_filter_gy, mult_output_g_y, pad_h, complex_w);

    free(fft_input);
    free(conj_filter_g_1);
    free(conj_filter_g_2);
    free(conj_filter_gx);
    free(conj_filter_gy);
    free(mult_output_g_1);
    free(mult_output_g_2);
    
    double *ifft_output_gx = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    double *ifft_output_gy = (double *) malloc (sizeof (double) * (pad_h * complex_w));
    conv_fft_ifft2D_hw(mult_output_g_x, ifft_output_gx, pad_h, complex_w);
    conv_fft_ifft2D_hw(mult_output_g_y, ifft_output_gy, pad_h, complex_w);

    free(mult_output_g_x);
    free(mult_output_g_y);

    double *real_output_gx = (double *) malloc (sizeof (double) * (pad_h * (complex_w / 2)));
    double *real_output_gy = (double *) malloc (sizeof (double) * (pad_h * (complex_w / 2)));
    conv_fft_real_hw(ifft_output_gx, real_output_gx, pad_h, complex_w);
    conv_fft_real_hw(ifft_output_gy, real_output_gy, pad_h, complex_w);

    free(ifft_output_gx);
    free(ifft_output_gy);
    
    conv_fft_crop_hw(real_output_gx, gx_out, pad_h, pad_w, height, width);
    conv_fft_crop_hw(real_output_gy, gy_out, pad_h, pad_w, height, width);

    free(real_output_gx);
    free(real_output_gy);

}
