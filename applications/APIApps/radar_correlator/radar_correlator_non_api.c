#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <gsl/gsl_fft_complex_float.h>
#include <gsl/gsl_fft_complex.h>
#include <string.h>

#include "dash.h"

void gsl_fft_wrapper(dash_cmplx_flt_type* input, dash_cmplx_flt_type* output, size_t size, bool isForwardTransform) {
//  printf("[fft] Running a %lu-Pt %s on the CPU\n", (*size), *isForwardTransform ? "FFT" : "IFFT");
  
  // Note: we copy to a new buffer here because the API doesn't state that we modify the input
  dash_cmplx_flt_type *data = (dash_cmplx_flt_type*) malloc((size) * sizeof(dash_cmplx_flt_type));
  memcpy(data, input, size * sizeof(dash_cmplx_flt_type));
  
  int check;
  // Our base floating point type is 4-byte (float)
  if (sizeof(dash_re_flt_type) == 4) {
    if (isForwardTransform) {
      check = gsl_fft_complex_float_radix2_forward((float*)(data), 1, size);
    } else {
      check = gsl_fft_complex_float_radix2_inverse((float*)(data), 1, size);
    }
  } 
  // Otherwise it's 8-byte (double)
  else {
    if (isForwardTransform) {
      check = gsl_fft_complex_radix2_forward((double*)(data), 1, size);
    } else {
      check = gsl_fft_complex_radix2_inverse((double*)(data), 1, size);
    }
  }

  if (check != 0) {
    fprintf(stderr, "[libdash] Failed to complete DASH_FFT_flt_cpu using libgsl with message %d!\n", check);
    free(data);
    return;
  }

  memcpy(output, data, size * sizeof(dash_cmplx_flt_type));
  free(data);
  return;
}

int main(void)
{
    size_t len, x_count, y_count, i, j, k;
    size_t n_samples; 
    size_t dft_size; 

    double lag;
    double T;
    double B;
    double sampling_rate;
    double max_corr;
    double index;

    double *c, *d, *X1, *X2, *corr_freq;
    double *time, *received, *dftMatrix, *indftMatrix, *corr, *gen_wave;

    clock_t begin, end;

    FILE *fp;

    int row, column, row2, column2;

    n_samples = 256;
    dft_size = 2 * n_samples;
    len = 2 * n_samples;

    T = 0.000512;
    B = 500000;
    sampling_rate = 1000;
    max_corr = 0;
    index = 0;

    time = malloc(n_samples*sizeof(double));
    received = malloc(2 * n_samples*sizeof(double));
    dftMatrix = malloc(2 * dft_size * dft_size * sizeof(double));
    indftMatrix = malloc(2 * dft_size * dft_size * sizeof(double));
    corr = malloc(2 * len * sizeof(double));
    gen_wave = malloc(2 * n_samples * sizeof(double));

    fp = fopen("./input/time_input.txt","r");
    if (fp == NULL) { printf("Unable to open time_input.txt!\n"); return 1; }
    for(i = 0; i < n_samples; i++) {
        fscanf(fp, "%lf", &time[i]);
    }
    fclose(fp);
    fp = NULL;

    for (i = 0; i < len; i += 2) {
        gen_wave[i] = sin(M_PI * B / T * pow(time[i / 2], 2));
        gen_wave[i + 1] = cos(M_PI * B / T * pow(time[i / 2], 2));
    }

    fp = fopen("./input/received_input.txt","r");
    if (fp == NULL) { printf("Unable to open received_input.txt!\n"); return 1; }
    for(i = 0; i < len; i++) {
        fscanf(fp,"%lf", &received[i]);
    }
    fclose(fp);
    fp = NULL;

    c = malloc(2 * len * sizeof(double));
    d = malloc(2 * len * sizeof(double));

    x_count = 0;
    y_count = 0;
    for (i = 0; i < 2 * len; i += 2) {
        if (i/2 > n_samples - 1) {
            c[i] = gen_wave[x_count];
            c[i + 1] = gen_wave[x_count + 1];
            x_count += 2;
        } else {
            c[i] = 0;
            c[i + 1] = 0;
        }
        if (i > n_samples) {
            d[i] = 0;
            d[i + 1] = 0;
        } else {
            d[i] = received[y_count];
            d[i + 1] = received[y_count + 1];
            y_count += 2;
        }
    }

    X1 = malloc(2 * len * sizeof(double));
    X2 = malloc(2 * len * sizeof(double));
    corr_freq = malloc(2 * len * sizeof(double));
   
    dash_cmplx_flt_type *fft_inp = (dash_cmplx_flt_type*) malloc(len * sizeof(dash_cmplx_flt_type));
    dash_cmplx_flt_type *fft_out = (dash_cmplx_flt_type*) malloc(len * sizeof(dash_cmplx_flt_type));

    // FFT 1
    for (size_t i = 0; i < len; i++) {
      fft_inp[i].re = (dash_re_flt_type) c[2*i];
      fft_inp[i].im = (dash_re_flt_type) c[2*i+1];
    }

    gsl_fft_wrapper(fft_inp, fft_out, len, true);
    // DASH_FFT_flt(fft_inp, fft_out, len, true);

    for (size_t i = 0; i < len; i++) {
      X1[2*i] = (double) fft_out[i].re;
      X1[2*i+1] = (double) fft_out[i].im;
    }

    // FFT 2
    for (size_t i = 0; i < len; i++) {
      fft_inp[i].re = (dash_re_flt_type) d[2*i];
      fft_inp[i].im = (dash_re_flt_type) d[2*i+1];
    }

    gsl_fft_wrapper(fft_inp, fft_out, len, true);
    // DASH_FFT_flt(fft_inp, fft_out, len, true);

    for (size_t i = 0; i < len; i++) {
      X2[2*i] = (double) fft_out[i].re;
      X2[2*i+1] = (double) fft_out[i].im;
    }
    
    // Multiplication
    for (i = 0; i < 2 * len; i += 2) {
        corr_freq[i] = (X1[i] * X2[i]) + (X1[i + 1] * X2[i + 1]);
        corr_freq[i + 1] = (X1[i + 1] * X2[i]) - (X1[i] * X2[i + 1]);
    }

    // IFFT
    for (size_t i = 0; i < len; i++) {
      fft_inp[i].re = (dash_re_flt_type) corr_freq[2*i];
      fft_inp[i].im = (dash_re_flt_type) corr_freq[2*i+1];
    }

    gsl_fft_wrapper(fft_inp, fft_out, len, false);
    // DASH_FFT_flt(fft_inp, fft_out, len, false);

    for (size_t i = 0; i < len; i++) {
      corr[2*i] = (double) fft_out[i].re;
      corr[2*i+1] = (double) fft_out[i].im;
    }

    free(fft_inp);
    free(fft_out);

    for (i = 0; i < 2 * len; i += 2)
    {
        // Only finding maximum of real part of correlation
        // (corr[i] > max_corr) && (max_corr = corr[i], index = i / 2);
        if (corr[i] > max_corr) {
            max_corr = corr[i];
            index = i / 2;
        }
    }

    lag = (n_samples - index) / sampling_rate;
    printf("Radar correlator complete; lag Value is: %lf\n", lag);
       
    free(time);    
    free(received);
    free(dftMatrix);
    free(indftMatrix);
    free(corr);
    free(gen_wave);
    free(c);
    free(d);
    free(X1);
    free(X2);
    free(corr_freq);

    return 0;
}
