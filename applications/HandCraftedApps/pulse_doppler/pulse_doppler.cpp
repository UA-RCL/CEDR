#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <dlfcn.h>
#ifdef INCLUDE_MAIN
#include <fstream>
#include <nlohmann/json.hpp>
#endif
#include "pulse_doppler.hpp"

#define PD_DEBUG 0
#define M 128
#define N 64
#define INPUT_DIR "./"

size_t m ;                               // number of pulses
size_t n_samples ;
double PRI;
double *mf;
double *p;
double *pulse;
double *corr;

fftwf_complex *in_xcorr1, *out_xcorr1, *in_xcorr2, *out_xcorr2, *in_xcorr3, *out_xcorr3;
fftwf_plan *p1, *p2, *p3;
double *q;
double *r;
double *f;
double *max, *a, *b;
fftwf_complex *in_fft, *out_fft;
fftwf_plan *p4;

double *X1, *X2;
double *corr_freq;

// Pointer to use to hold the shared object file handle
void *dlhandle;

// arg1: input
// arg2: output
// arg3: fft size
// arg4: is forward transform/FFT? if not, it's IFFT
void (*fft_accel_func)(double**, double**, size_t*, bool*, unsigned int);
void (*ifft_accel_func)(double**, double**, size_t*, bool*, unsigned int);

__attribute__((__visibility__("default"))) thread_local unsigned int __CEDR_TASK_ID__ = 0;
__attribute__((__visibility__("default"))) thread_local unsigned int __CEDR_CLUSTER_IDX__ = 0;

#ifndef INCLUDE_MAIN
void __attribute__((constructor)) setup(void) {
  printf("Starting constructor!\n");
  //fflush(stdout);
  m = M;
  n_samples = N;
  PRI = 6.3e-5;
  mf = (double*) malloc((2*N)*M*2*sizeof(double));
  pulse = (double*) malloc(2*N *sizeof(double));
  p = (double*) malloc(2*N*M *sizeof(double));
  X1 = (double*) malloc(2*(2*(n_samples))*M *sizeof(double));
  X2 = (double*) malloc(2*(2*(n_samples))*M *sizeof(double));
  corr_freq = (double*) malloc(2*(2*(n_samples))*M *sizeof(double));
  corr = (double*) malloc(2*(2*N)*M *sizeof(double));
  in_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * 2*(n_samples)* (m));
  out_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples))* (m));
  in_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples))* (m));
  out_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples))* (m));
  in_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples))* (m));
  out_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples))* (m));
  p1 = (fftwf_plan*) malloc(m * sizeof(fftwf_plan));
  p2 = (fftwf_plan*) malloc(m * sizeof(fftwf_plan));
  p3 = (fftwf_plan*) malloc(m * sizeof(fftwf_plan));
  for (int i =0; i < m; i++){
    p1[i] = fftwf_plan_dft_1d((2*(n_samples)), &(in_xcorr1[i * (2*(n_samples))]), &(out_xcorr1[i * (2*(n_samples))]), FFTW_FORWARD, FFTW_ESTIMATE);
  }
  for (int i =0; i <m;i++){
    p2[i] = fftwf_plan_dft_1d((2*(n_samples)), &(in_xcorr2[i * (2*(n_samples))]), &(out_xcorr2[i * (2*(n_samples))]), FFTW_FORWARD, FFTW_ESTIMATE);
  }
  for (int i =0; i <m;i++){
    p3[i] = fftwf_plan_dft_1d((2*(n_samples)), &(in_xcorr3[i * (2*(n_samples))]), &(out_xcorr3[i * (2*(n_samples))]), FFTW_BACKWARD, FFTW_ESTIMATE);
  }

  q = (double*) malloc(2*m*sizeof(double) * 2*(n_samples));
  r = (double*) malloc(m*sizeof(double) * 2*(n_samples));
  //*l = malloc(2*m*sizeof(double));
  f = (double*) malloc(m*(2*n_samples)*sizeof(double));
  max = (double*) malloc((2*n_samples)*sizeof(double));
  for (int i=0 ; i < 2*n_samples; i= i+ 1){
    max[i] = 0;
  }
  a = (double*) malloc((2*n_samples)*sizeof(double));
  b = (double*) malloc((2*n_samples)*sizeof(double));
  //max = 0;
  in_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * m * (2*(n_samples)));
  out_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * m * (2*(n_samples)));
  //p4 = fftwf_plan_dft_1d(m, in_fft, out_fft, FFTW_FORWARD, FFTW_ESTIMATE);
  p4 = (fftwf_plan*) malloc((n_samples)*2 * sizeof(fftwf_plan));
  for (int i =0; i < (n_samples)*2; i++) {
    p4[i] = fftwf_plan_dft_1d(((m)), &(in_fft[i * (m)]), &(out_fft[i * (m)]), FFTW_FORWARD, FFTW_ESTIMATE);
  }

  printf("About to read input_pd_pulse!\n");
  FILE *fp;
  std::string pd_pulse = std::string("./input/input_pd_pulse.txt");
  fp = fopen(pd_pulse.c_str(), "r");
  if (fp == nullptr) {
    printf("Error opening input_pd_pulse");
  }
  for(int i=0; i<2*N; i++){
    fscanf(fp, "%lf", &(pulse[i]));
  }
  fclose(fp);

  printf("About to read input_pd_ps!\n");
  std::string pd_ps = std::string("./input/input_pd_ps.txt");
  fp = fopen(pd_ps.c_str(), "r");
  for(int j = 0; j < 2 * N*M; j++){
    fscanf(fp, "%lf", &(p[j]));
  }
  fclose(fp);


  ///////////////accelerators////////////////////////////////
  #ifdef ARM
  printf("About to read accelelator binary\n");
  dlhandle = dlopen("./libdash-rt.so", RTLD_LAZY);
  if (dlhandle == nullptr) {
    printf("Unable to open libdash-rt shared object!\n");
  }
  fft_accel_func = (void(*)(double**, double**, size_t*, bool*, unsigned int)) dlsym(dlhandle, "DASH_FFT_fft");
  if (fft_accel_func == nullptr) {
    printf("Unable to get function handle for FFT accelerator function!\n");
  }
  ifft_accel_func = (void(*)(double**, double**, size_t*, bool*, unsigned int)) dlsym(dlhandle, "DASH_FFT_fft");
  if (ifft_accel_func == nullptr) {
    printf("Unable to get function handle for IFFT accelerator function!\n");
  }
  #endif

  #ifdef CUDA
  printf("About to read accelelator CUDA binary\n");
  dlhandle = dlopen("./libdash-rt.so", RTLD_LAZY);
  if (dlhandle == nullptr) {
    printf("Unable to open libdash-rt shared object!\n");
  }
  fft_accel_func = (void(*)(double**, double**, size_t*, bool*, unsigned int)) dlsym(dlhandle, "DASH_FFT_gpu");
  if (fft_accel_func == nullptr) {
    printf("Unable to get function handle for FFT accelerator function!\n");
  }
  ifft_accel_func = (void(*)(double**, double**, size_t*, bool*, unsigned int)) dlsym(dlhandle, "DASH_FFT_gpu");
  if (ifft_accel_func == nullptr) {
    printf("Unable to get function handle for IFFT accelerator function!\n");
  }
  #endif
  printf("Finished pulse_doppler_init\n");
}

void __attribute__((destructor)) teardown(void) {
  printf("Starting Destructor!\n");
  free(mf);
  free(p);
  free(pulse);
  free(max);
  free(a);
  free(b);
  free(q);
  free(f);
  free(r);
  free(X1);
  free(X2);
  free(corr_freq);
  free(corr);
  for (int i = 0; i < m; i++){
    fftwf_destroy_plan((p1[i]));
    fftwf_destroy_plan((p2[i]));
    fftwf_destroy_plan((p3[i]));
  }
  for (int i =0; i < (n_samples)*2; i++){
    fftwf_destroy_plan((p4[i]));
  }
  free(p1);
  free(p2);
  free(p3);
  free(p4);
  fftwf_free(in_xcorr1);
  fftwf_free(out_xcorr1);
  fftwf_free(in_xcorr2);
  fftwf_free(out_xcorr2);
  fftwf_free(in_xcorr3);
  fftwf_free(out_xcorr3);
  fftwf_free(in_fft);
  fftwf_free(out_fft);
  if (dlhandle != nullptr) {
    dlclose(dlhandle);
  }
  printf("Finished pulse_doppler destructor\n");
}
#endif

extern "C" void fftwf_fft(double *input_array, fftwf_complex *in, fftwf_complex *out, double *output_array, size_t n_elements, fftwf_plan p) {
    for(size_t i = 0; i < 2*n_elements; i+=2)
    {
        in[i/2][0] = input_array[i];
        in[i/2][1] = input_array[i+1];
    }
    fftwf_execute(p);
    for(size_t i = 0; i < 2*n_elements; i+=2)
    {
        output_array[i] = (double) out[i/2][0];
        output_array[i+1] = (double) out[i/2][1];
    }
}

extern "C" void pulse_doppler_nop(void) {
    printf("Finished pulse_doppler_nop, task id %d\n", __CEDR_TASK_ID__);
}

extern "C" void pulse_doppler_FFT_0_cpu(void) {
    size_t len;
    len = 2 * (n_samples);
    printf("Starting CPU execution of FFT_0, task id: %d\n", __CEDR_TASK_ID__);

    double *p_loc;
    double *X1_loc;
    p_loc = &((p[ 2*n_samples * ((__CEDR_TASK_ID__)/2)]));
    X1_loc = &((X1[2*(2*(n_samples)) * ((__CEDR_TASK_ID__)/2)]));

    double *c = (double*) malloc( 2*len *sizeof(double));
    for(size_t i = 0; i<2*(n_samples-1); i+=2){
        c[i] = 0;
        c[i+1] = 0;
    }
    memcpy(c+2*(n_samples - 1), p_loc, 2*(n_samples)*sizeof(double));
    c[2*len-2] = 0;
    c[2*len - 1] = 0;
    fftwf_fft(c, &(in_xcorr1[(__CEDR_TASK_ID__/2)*2*(n_samples)]), &(out_xcorr1[(__CEDR_TASK_ID__/2)*2*(n_samples)]),  X1_loc, len, p1[__CEDR_TASK_ID__/2]);

    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*(n_samples)*2); i=i+2) {
            printf("IT %d X1 index %d real: %lf  imag %lf \n",(__CEDR_TASK_ID__)/2 , i/2, X1[i+(2*(2*(n_samples)) * ((__CEDR_TASK_ID__)/2))],X1[i+1+ (2*(2*(n_samples)) * ((__CEDR_TASK_ID__)/2))]);
        }
    }
    free(c);
    printf("Finished pulse_doppler_FFT_0, task id: %d\n", __CEDR_TASK_ID__);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, __CEDR_TASK_ID__, task->task_name, self);
}

extern "C" void pulse_doppler_FFT_0_accel(void) {
    size_t len;
    bool isFwd;
    len = 2 * (n_samples);
    isFwd = true;
    printf("Starting accelerator execution of FFT_0, task id: %u, cluster idx: %u\n", __CEDR_TASK_ID__, __CEDR_CLUSTER_IDX__);

    double *p_loc;
    double *X1_loc;
    p_loc = &((p[ 2*n_samples * ((__CEDR_TASK_ID__)/2)]));
    X1_loc = &((X1[2*(2*(n_samples)) * ((__CEDR_TASK_ID__)/2)]));

    double *c = (double*) malloc( 2*len *sizeof(double));
    for(size_t i = 0; i<2*(n_samples-1); i+=2) {
        c[i] = 0;
        c[i+1] = 0;
    }
    memcpy(c+2*(n_samples - 1), p_loc, 2*(n_samples)*sizeof(double));
    c[2*len-2] = 0;
    c[2*len - 1] = 0;
    // fftwf_fft(c, &(in_xcorr1[(__CEDR_TASK_ID__/2)*2*(n_samples)]), &(out_xcorr1[(__CEDR_TASK_ID__/2)*2*(n_samples)]),  X1_loc, len, p1[__CEDR_TASK_ID__/2]);
    (*fft_accel_func)(&c, &X1_loc, &len, &isFwd, __CEDR_CLUSTER_IDX__);
    if (PD_DEBUG == 1) {
        for (int i = 0; i <  (2*(n_samples)*2); i=i+2) {
            printf("IT %d X1 index %d real: %lf  imag %lf \n",(__CEDR_TASK_ID__)/2 , i/2, X1[i+(2*(2*(n_samples)) * ((__CEDR_TASK_ID__)/2))],X1[i+1+ (2*(2*(n_samples)) * ((__CEDR_TASK_ID__)/2))]);
        }
    }
    free(c);
    printf("Finished accelerator execution of pulse_doppler_FFT_0, task id: %d\n", __CEDR_TASK_ID__);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, __CEDR_TASK_ID__, task->task_name, self);
}

extern "C" void pulse_doppler_FFT_1_cpu(void) {
    printf("Starting CPU execution of FFT_1, task id: %d\n", __CEDR_TASK_ID__);
    size_t len;
    len = 2 * (n_samples);
    double *d = (double*) malloc( 2*len *sizeof(double));
    double *z = (double*) malloc( 2*(n_samples) *sizeof(double));
    double *X2_loc;
    X2_loc = &(X2[2*(2*(n_samples)) * (((__CEDR_TASK_ID__)/2) - 1)]);
    double *y;
    y = pulse;

    for(size_t i = 0; i<2*(n_samples); i+=2){
        z[i] = 0;
        z[i+1] = 0;
    }
    memcpy(d, y, 2*(n_samples)*sizeof(double));
    memcpy(d+2*(n_samples), z, 2*(n_samples)*sizeof(double));

    fftwf_fft(d, &(in_xcorr2[((__CEDR_TASK_ID__/2)-1)*2*(n_samples)]),
              &(out_xcorr2[((__CEDR_TASK_ID__/2)-1)*2*(n_samples)]), X2_loc, len,p2[(__CEDR_TASK_ID__/2)-1]);

    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*(n_samples)*2); i=i+2) {
            //printf("X2 index %d real: %lf  imag %lf \n", i/2, X2[i],X2[i+1]);
            printf("IT %d X2 index %d real: %lf  imag %lf \n",(__CEDR_TASK_ID__/2)-1 , i/2, X2[i+(2*(2*(n_samples)) * ((__CEDR_TASK_ID__/2)-1))], X2[i+1+ (2*(2*(n_samples)) * ((__CEDR_TASK_ID__/2) - 1))]);
        }
    }

    free(d);
    free(z);
    printf("Finished pulse_doppler_FFT_1, task id: %d\n", __CEDR_TASK_ID__);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, __CEDR_TASK_ID__, task->task_name, self);
}

extern "C" void pulse_doppler_FFT_1_accel(void) {
    printf("Starting accelerator execution of FFT_1, task id: %u, cluster idx: %u\n", __CEDR_TASK_ID__, __CEDR_CLUSTER_IDX__);
    size_t len;
    bool isFwd;
    len = 2 * (n_samples);
    isFwd = true;
    double *d = (double*) malloc( 2*len *sizeof(double));
    double *z = (double*) malloc( 2*(n_samples) *sizeof(double));
    double *X2_loc;
    X2_loc = &(X2[2*(2*(n_samples)) * (((__CEDR_TASK_ID__)/2) - 1)]);
    double *y;
    y = pulse;

    for(size_t i = 0; i<2*(n_samples); i+=2){
        z[i] = 0;
        z[i+1] = 0;
    }
    memcpy(d, y, 2*(n_samples)*sizeof(double));
    memcpy(d+2*(n_samples), z, 2*(n_samples)*sizeof(double));
    // fftwf_fft(d, &(in_xcorr2[((__CEDR_TASK_ID__/2)-1)*2*(n_samples)]), \
              &(out_xcorr2[((__CEDR_TASK_ID__/2)-1)*2*(n_samples)]), X2_loc, len,p2[(__CEDR_TASK_ID__/2)-1]);
    (*fft_accel_func)(&d, &X2_loc, &len, &isFwd, __CEDR_CLUSTER_IDX__);
    
    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*(n_samples)*2); i=i+2) {
            //printf("X2 index %d real: %lf  imag %lf \n", i/2, X2[i],X2[i+1]);
            printf("IT %d X2 index %d real: %lf  imag %lf \n",(__CEDR_TASK_ID__/2)-1 , i/2, X2[i+(2*(2*(n_samples)) * ((__CEDR_TASK_ID__/2)-1))], X2[i+1+ (2*(2*(n_samples)) * ((__CEDR_TASK_ID__/2) - 1))]);
        }
    }

    free(d);
    free(z);
    printf("Finished pulse_doppler_FFT_1, task id: %d\n", __CEDR_TASK_ID__);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, __CEDR_TASK_ID__, task->task_name, self);
}

extern "C" void pulse_doppler_MUL(void) {
    size_t len;
    len = 2 * (n_samples);
    double *X1_loc;
    double *X2_loc;
    double *corr_freq_loc;
    int index = 2*m + 1;

    X1_loc = &((X1[2*(2*(n_samples)) * (__CEDR_TASK_ID__-index)]));
    X2_loc = &(X2[2*(2*(n_samples)) * (__CEDR_TASK_ID__-index)]);
    corr_freq_loc =  &(corr_freq[2*(2*(n_samples)) * (__CEDR_TASK_ID__-index)]);

    for(size_t i =0;i<2*len;i+=2){
        corr_freq_loc[i] =   (X1_loc[i] * X2_loc[i]) + (X1_loc[i+1] * X2_loc[i+1]);
        corr_freq_loc[i+1] = (X1_loc[i+1] * X2_loc[i]) - (X1_loc[i] * X2_loc[i+1]);
    }

    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*len); i=i+2) {
            //printf("corr_freq index %d real: %lf  imag %lf \n", i/2, corr_freq[i],corr_freq[i+1]);
            printf("IT %d corr_freq index %d real: %lf  imag %lf \n",(__CEDR_TASK_ID__-index) , i/2, corr_freq[i+(2*(2*(n_samples)) * (__CEDR_TASK_ID__-index))],corr_freq[i+1+ (2*(2*(n_samples)) * (__CEDR_TASK_ID__ - index))]);
        }
    }
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, __CEDR_TASK_ID__, task->task_name, self);
    printf("Finished pulse_doppler_MUL, task id: %d\n", __CEDR_TASK_ID__);
}

extern "C" void pulse_doppler_IFFT_cpu(void) {
    size_t len;
    len = 2 * (n_samples);

    double *corr_freq_loc;
    double *corr_loc;
    int index = 3*m + 1;
    corr_loc = &((corr[2*(2*(n_samples)) * (__CEDR_TASK_ID__-index)]));
    corr_freq_loc = &((corr_freq[2*(2*(n_samples)) * (__CEDR_TASK_ID__-index)]));

    fftwf_fft(corr_freq_loc, &(in_xcorr3[(__CEDR_TASK_ID__-index)*2*(n_samples)]), &(out_xcorr3[(__CEDR_TASK_ID__-index)*2*(n_samples)]), corr_loc, len, p3[__CEDR_TASK_ID__-index]);
    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*len); i=i+2) {
            //printf("corr index %d real: %lf  imag %lf \n", i/2, corr[i],corr[i+1]);
            printf("IT %d corr index %d real: %lf  imag %lf \n",(__CEDR_TASK_ID__-index) , i/2, corr[i+(2*(2*(n_samples)) * (__CEDR_TASK_ID__-index))],corr[i+1+ (2*(2*(n_samples)) * (__CEDR_TASK_ID__ - index))]);
        }
    }
    printf("Finished pulse_doppler_IFFT, task id: %d\n", __CEDR_TASK_ID__);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, __CEDR_TASK_ID__, task->task_name, self);
}

extern "C" void pulse_doppler_IFFT_accel(void) {
    size_t len;
    bool isFwd;
    len = 2 * (n_samples);
    isFwd = false;

    double *corr_freq_loc;
    double *corr_loc;
    int index = 3*m + 1;
    corr_loc = &((corr[2*(2*(n_samples)) * (__CEDR_TASK_ID__-index)]));
    corr_freq_loc = &((corr_freq[2*(2*(n_samples)) * (__CEDR_TASK_ID__-index)]));

    //fftwf_fft(corr_freq_loc, &(in_xcorr3[(__CEDR_TASK_ID__-index)*2*(n_samples)]), &(out_xcorr3[(__CEDR_TASK_ID__-index)*2*(n_samples)]), corr_loc, len, p3[__CEDR_TASK_ID__-index]);
    (*ifft_accel_func)(&corr_freq_loc, &corr_loc, &len, &isFwd, __CEDR_CLUSTER_IDX__);
    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*len); i=i+2) {
            //printf("corr index %d real: %lf  imag %lf \n", i/2, corr[i],corr[i+1]);
            printf("IT %d corr index %d real: %lf  imag %lf \n",(__CEDR_TASK_ID__-index) , i/2, corr[i+(2*(2*(n_samples)) * (__CEDR_TASK_ID__-index))],corr[i+1+ (2*(2*(n_samples)) * (__CEDR_TASK_ID__ - index))]);
        }
    }
    printf("Finished pulse_doppler_IFFT, task id: %d\n", __CEDR_TASK_ID__);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, __CEDR_TASK_ID__, task->task_name, self);
}

extern "C" void pulse_doppler_REALIGN_MAT(void) {
    size_t len;
    len = 2 * (n_samples);
    double *mf_loc;
    double *corr_loc;
    mf_loc = mf;
    corr_loc = corr;
    for(int k = 0; k < m; k++){
        for(int j = 0; j < 2*(2 * n_samples); j+=2){
            mf_loc[j/2+(2*k)*(2*n_samples)] = corr_loc[k*(2*(2 * n_samples)) + j];
            mf_loc[j/2+(2*k+1)*(2*n_samples)]= corr_loc[k*(2*(2 * n_samples)) + j+1];
        }
    }
    if (PD_DEBUG == 1){
        for(int k = 0; k < m; k++){
            for(int j = 0; j < 2*(2 * n_samples); j+=2){
                //printf("mf %f %f\n",mf[k*2*(2 * n_samples) + j], mf[k*2*(2 * n_samples) + j+1]);
                //printf("mf %f %f\n",mf[j/2+(2*k)*(2*n_samples)], mf[j/2+(2*k+1)*(2*n_samples)]);
                printf("mf %f %f\n",(mf)[j/2+(2*k)*(2*n_samples)], (mf)[j/2+(2*k+1)*(2*n_samples)]);

            }
        }
    }
    printf("Finished pulse_doppler_REALIGN_MAT, task id: %d\n", __CEDR_TASK_ID__);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, __CEDR_TASK_ID__, task->task_name, self);
}

extern "C" void pulse_doppler_FFT_2_cpu(void) {
    double *l = (double*) malloc(2*(m)*sizeof(double));
    double *q_loc;
    int index = 4*m + 2;
    int x = __CEDR_TASK_ID__ - index;
    q_loc = &((q)[ x*2*m]);
    for(int o = 0; o < 2*m; o++){
        l[o] = mf[x+o*(2*n_samples)];
    }

    fftwf_fft(l, &(in_fft[(__CEDR_TASK_ID__ - index)*m]), &(out_fft[(__CEDR_TASK_ID__ - index)*m]), q_loc, m, p4[__CEDR_TASK_ID__ - index]);

    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*(m)); i=i+2) {
            printf("IT %d FFT2 index %d real: %lf  imag %lf \n",x , i/2, q[i+ (x*2*m) ],q[i+1+ (x*2*m)]);
        }
    }

    free(l);
    printf("Finished pulse_doppler_FFT_2, task id: %d\n", __CEDR_TASK_ID__);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, __CEDR_TASK_ID__, task->task_name, self);

}

extern "C" void pulse_doppler_FFT_2_accel(void) {
    size_t len = 2 * m;
    bool isFwd = true;
    double *l = (double*) malloc(2*(m)*sizeof(double));
    double *q_loc;
    int index = 4*m + 2;
    int x = __CEDR_TASK_ID__ - index;
    q_loc = &((q)[ x*2*m]);
    for(int o = 0; o < 2*m; o++){
        l[o] = mf[x+o*(2*n_samples)];
    }

    (*fft_accel_func)(&l, &q_loc, &len, &isFwd, __CEDR_CLUSTER_IDX__);

    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*(m)); i=i+2) {
            printf("IT %d FFT2 index %d real: %lf  imag %lf \n",x , i/2, q[i+ (x*2*m) ],q[i+1+ (x*2*m)]);
        }
    }

    free(l);
    printf("Finished pulse_doppler_FFT_2, task id: %d\n", __CEDR_TASK_ID__);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, __CEDR_TASK_ID__, task->task_name, self);

}

extern "C" void pulse_doppler_AMPLITUDE(void) {
    int index = (2*n_samples)  + (4*m) + 2;
    int x = __CEDR_TASK_ID__ - index;

    double *q_loc;
    double *r_loc;
    q_loc = &((q)[ x*2*m]);
    r_loc = &((r)[ x*m]);
    for(int y = 0; y < 2*m; y+=2) {
        r_loc[y/2] = sqrt(q_loc[y]*q_loc[y]+q_loc[y+1]*q_loc[y+1]);
    }

    if (PD_DEBUG == 1){
        for (int i = 0; i < ((m)); i=i+1) {
            printf("IT %d AMP index %d real: %lf \n",x , i, r[i+(x*m)]);
        }
    }
    printf("Finished pulse_doppler_AMPLITUDE, task id: %d\n", __CEDR_TASK_ID__);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, __CEDR_TASK_ID__, task->task_name, self);
}

extern "C" void swap(double *v1, double *v2) {
    double tmp = *v1;
    *v1 = *v2;
    *v2 = tmp;
}

extern "C" void fftshift(double *data, double count)
{
    int k = 0;
    int c = (double) floor((double)count/2);
    // For odd and for even numbers of element use different algorithm
    if ((int)count % 2 == 0)
    {
        for (k = 0; k < c; k++)
            swap(&data[k], &data[k+c]);
    }
    else
    {
        double tmp = data[0];
        for (k = 0; k < c; k++)
        {
            data[k] = data[c + k + 1];
            data[c + k + 1] = data[k + 1];
        }
        data[c] = tmp;
    }
}

extern "C" void pulse_doppler_FFTSHIFT(void) {
    int index = (2*2*n_samples)  + (4*m) + 2;
    int x = __CEDR_TASK_ID__ - index;

    double *r_loc;
    r_loc = &((r)[x*m]);

    fftshift(r_loc, m);

    if (PD_DEBUG == 1){
        for (int i = 0; i <  ((m)); i=i+1) {
            printf("IT %d SHIFT index %d real: %lf \n",x , i, r[i+ (x*m) ]);
        }
    }
    printf("Finished pulse_doppler_FFTSHIFT, task id: %d\n", __CEDR_TASK_ID__);
}

extern "C" void pulse_doppler_MAX0(void) {
    int index = (3*2*n_samples)  + (4*m) + 2;
    int x = __CEDR_TASK_ID__ - index;

    double *r_loc;
    r_loc = &((r)[ x*m]);
    double *max_loc, *a_loc, *b_loc;
    max_loc = &((max)[x]);
    a_loc = &((a)[x]);
    b_loc = &((b)[x]);

    for(int z = 0; z < m; z++){
        f[x+z*(2*n_samples)] = r_loc[z];
        if( r_loc[z] > max_loc[0] ){
            max_loc[0] = r_loc[z];
            a_loc[0] = z + 1;
            b_loc[0] = x + 1;
        }
    }

    if (PD_DEBUG == 1){
        printf("IT %d MAX0 real: %lf %lf %lf\n", x, max[x], a[x], b[x]);
    }
    printf("Finished pulse_doppler_MAX0, task id: %d\n", __CEDR_TASK_ID__);
}

extern "C" void pulse_doppler_FINAL_MAX(void) {
    int index = (4*2*n_samples)  + (4*m) + 2;
    int x = __CEDR_TASK_ID__ - index;

    double max_temp, a_temp, b_temp;
    max_temp = 0; a_temp = 0; b_temp =0;

    for(int z = 0; z < (2*n_samples); z++){
        if(  max[z] > max_temp){
            max_temp = max[z];
            a_temp = a[z];
            b_temp = b[z];
        }
    }

    if (PD_DEBUG == 0){
        double rg, dp;
        rg = (b_temp-n_samples)/(n_samples-1)*PRI;
        dp = (a_temp-(m+1)/2)/(m-1)/PRI;
        for (int j = 0; j<(m);j++){
            for(int k = 0; k < 2 * n_samples; k++){
                //printf("recive IT %d index %d value %f\n" ,j,k, p[k + j*2*(n_samples)]);
            }
            for (int i = 0; i <  (2*(n_samples)*2); i=i+2) {
                //printf("IT %d X1 index %d real: %lf  imag %lf \n",j , i/2, X1[i+(2*(2*(n_samples)) * j)],X1[i+1+ (2*(2*(n_samples)) * j)]);
            }
            for (int i = 0; i <  (2*(n_samples)*2); i=i+2) {
                //printf("IT %d X2 index %d real: %lf  imag %lf \n",j , i/2, X2[i+(2*(2*(n_samples)) * j)],X2[i+1+ (2*(2*(n_samples)) * j)]);
            }

        }
        printf("doppler shift = %lf, time delay = %lf\n", dp, rg);
    }
    printf("Finished pulse_doppler_FINAL_MAX, task id: %d\n", __CEDR_TASK_ID__);
}

#ifdef INCLUDE_MAIN
using json = nlohmann::json;

void generateJSON(void) {
    json output, DAG;
    unsigned int idx_offset = 0;

    //--- Application information
    output["AppName"] = "pulse_doppler";
    output["SharedObject"] = "pulse_doppler.so";
    output["Variables"] = json::object();
    //output["FieldsBytes"] = sizeof(struct pulse_doppler_fields);

    //--- Initial NOP Stage node
    DAG["S0"] = json::object();
    DAG["S0"]["platforms"] = json::array({
        {
            {"name", "cpu"},
            {"nodecost", 1.0f},
            {"runfunc", "pulse_doppler_nop"}
        }
    });
    DAG["S0"]["predecessors"] = json::array();
    DAG["S0"]["successors"] = json::array();
    DAG["S0"]["task_id"] = idx_offset + 0;

    //--- FFT0 and FFT1 nodes
    idx_offset += 1; // Account for S0
    for (int i = 0; i < 2*M; i++) {
        std::string node_key, run_func;
        if (i % 2 == 0) {
            node_key = "FFT_0_" + std::to_string(i/2);
            run_func = "FFT_0";
        } else {
            node_key = "FFT_1_" + std::to_string(i/2);
            run_func = "FFT_1";
        }

        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "pulse_doppler_" + run_func + "_cpu"}
            },
           {
               {"name", "fft"},
               {"nodecost", 28.0f},
               {"runfunc", "pulse_doppler_" + run_func + "_accel"}
           }
        });

        DAG["S0"]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "S0"},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }

    //--- MUL Nodes
    idx_offset += 2*M; // S0 and FFT0/FFT1s
    for (int i = 0; i < M; i++) {
        std::string node_key = "MUL_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "pulse_doppler_MUL"}
            }
        });

        DAG["FFT_0_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG["FFT_1_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "FFT_0_" + std::to_string(i)},
                {"edgecost", 0}
            },
            {
                {"name", "FFT_1_" + std::to_string(i)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }

    //IFFT
    idx_offset += M; // S0, FFT0/1, MUL
    for (int i = 0; i < M; i++) {
        std::string node_key = "IFFT_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "pulse_doppler_IFFT_cpu"}
            },
           {
               {"name", "fft"},
               {"nodecost", 28.0f},
               {"runfunc", "pulse_doppler_IFFT_accel"}
           }
        });

        DAG["MUL_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "MUL_" + std::to_string(i)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }

    //--- Realign Matrix
    idx_offset += M; //S0, FFT0/1, MUL, IFFT
    DAG["REALIGN_MAT"] = json::object();
    DAG["REALIGN_MAT"]["platforms"] = json::array({
        {
            {"name", "cpu"},
            {"nodecost", 30.0f},
            {"runfunc", "pulse_doppler_REALIGN_MAT"}
        }
    });
    DAG["REALIGN_MAT"]["predecessors"] = json::array();
    for (int i = 0; i < M; i++) {
        DAG["IFFT_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", "REALIGN_MAT"},
            {"edgecost", 0}
        }));
        DAG["REALIGN_MAT"]["predecessors"].push_back(json::object({
            {"name", "IFFT_" + std::to_string(i)},
            {"edgecost", 0}
        }));
    }
    DAG["REALIGN_MAT"]["successors"] = json::array();
    DAG["REALIGN_MAT"]["task_id"] = idx_offset + 0;

    //--- FFT 2
    idx_offset += 1; //S0, FFT0/1, MUL, IFFT, REALIGN_MAT
    for (int i = 0; i < 2*N; i++) {
        std::string node_key = "FFT_2_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "pulse_doppler_FFT_2_cpu"}
            },
           {
               {"name", "fft"},
               {"nodecost", 28.0f},
               {"runfunc", "pulse_doppler_FFT_2_accel"}
           }
        });

        DAG["REALIGN_MAT"]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "REALIGN_MAT"},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }

    //--- Amplitude
    idx_offset += 2*N; // S0, FFT0/1, MUL, IFFT, REALIGN, FFT2
    for (int i = 0; i < 2*N; i++) {
        std::string node_key = "AMPLITUDE_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "pulse_doppler_AMPLITUDE"}
            }
        });

        DAG["FFT_2_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "FFT_2_" + std::to_string(i)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }

    //--- FFT-Shift
    idx_offset += 2*N; //S0, FFT0/1, MUL, IFFT, REALIGN, FFT2, AMP
    for (int i = 0; i < 2*N; i++) {
        std::string node_key = "FFTSHIFT_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "pulse_doppler_FFTSHIFT"}
            }
        });

        DAG["AMPLITUDE_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "AMPLITUDE_" + std::to_string(i)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }

    //--- MAX
    idx_offset += 2*N; //S0, FFT0/1, MUL, IFFT, REALIGN, FFT2, AMP, FFT-SHIFT
    for (int i = 0; i < 2*N; i++) {
        std::string node_key = "MAX_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "pulse_doppler_MAX0"}
            }
        });

        DAG["FFTSHIFT_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "FFTSHIFT_" + std::to_string(i)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }

    //--- FINAL MAX
    idx_offset += 2*N; //S0, FFT0/1, MUL, IFFT, REALIGN, FFT2, AMP, FFT-SHIFT, MAX
    DAG["FINAL_MAX"] = json::object();
    DAG["FINAL_MAX"]["platforms"] = json::array({
        {
            {"name", "cpu"},
            {"nodecost", 30.0f},
            {"runfunc", "pulse_doppler_FINAL_MAX"}
        }
    });

    DAG["FINAL_MAX"]["predecessors"] = json::array();
    for (int i = 0; i < 2*N; i++) {
        DAG["MAX_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", "FINAL_MAX"},
            {"edgecost", 0}
        }));
        DAG["FINAL_MAX"]["predecessors"].push_back(json::object({
            {"name", "MAX_" + std::to_string(i)},
            {"edgecost", 0}
        }));
    }

    DAG["FINAL_MAX"]["successors"] = json::array();
    DAG["FINAL_MAX"]["task_id"] = idx_offset + 0;

    output["DAG"] = DAG;

    std::ofstream output_file("pulse_doppler.json");
    if (!output_file.is_open()) {
        fprintf(stderr, "Failed to open output file for writing JSON\n");
        exit(1);
    }
    output_file << std::setw(2) << output;
}

int main(void) {
    generateJSON();
}
#endif
