#include <iostream>
#include <cstdio>
#include "correlator.hpp"
#include <math.h>
#include <unistd.h>
#include <fftw3.h>
#include <complex.h>
#include <stdlib.h>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include "debug.h"

int iter;
size_t n_samples;
size_t time_n_samples;
double T;
double B;
double sampling_rate;
double *time_lfm;
double *received;
double *lfm_waveform;
fftwf_complex *in_xcorr1, *out_xcorr1, *in_xcorr2, *out_xcorr2, *in_xcorr3, *out_xcorr3;
fftwf_plan p1, p2, p3;
double *X1, *X2;
double *corr_freq;
double *corr;


double *_lfm_waveform;
fftwf_complex *_in_xcorr1, *_out_xcorr1, *_in_xcorr2, *_out_xcorr2, *_in_xcorr3, *_out_xcorr3;
fftwf_plan _p1, _p2, _p3;
double *_X1, *_X2;
double *_corr_freq;
double *_corr;

// Pointer to use to hold the shared object file handle
void *dlhandle;

// arg1: input
// arg2: output
// arg3: fft size
// arg4: is forward transform/FFT? if not, it's IFFT
void (*fft_accel_func)(double**, double**, size_t*, bool*, unsigned int);

__attribute__((__visibility__("default"))) thread_local unsigned int __CEDR_CLUSTER_IDX__ = 0;

void __attribute__((constructor)) setup(void) {
  printf("[Correlator] intializing variables\n");

  iter = 0;  
  n_samples=256;
  time_n_samples =1;
  T = (double)(256.0/500000);
  B = (double)500000;
  sampling_rate = 1000;
  time_lfm = (double *) malloc((2 * n_samples)*sizeof(double));   // TODO: Shouldn't it use time_n_samples?
  received = (double *)malloc(2 * n_samples*sizeof(double));
  lfm_waveform = (double *)malloc(2 * n_samples*sizeof(double));
  in_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  out_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  in_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  out_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  in_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  out_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  p1 = fftwf_plan_dft_1d(n_samples, in_xcorr1, out_xcorr1, FFTW_FORWARD, FFTW_ESTIMATE);
  p2 = fftwf_plan_dft_1d(n_samples, in_xcorr2, out_xcorr2, FFTW_FORWARD, FFTW_ESTIMATE);
  p3 = fftwf_plan_dft_1d(n_samples, in_xcorr3, out_xcorr3, FFTW_BACKWARD, FFTW_ESTIMATE);
  X1 =  (double *)malloc(2*(n_samples) *sizeof(double));
  X2 =  (double *)malloc(2*(n_samples) *sizeof(double));
  corr_freq =  (double *)malloc(2*(n_samples) *sizeof(double));
  corr =  (double *)malloc(2*(n_samples) *sizeof(double));

  _lfm_waveform = (double *)malloc(2 * n_samples*sizeof(double));
  _in_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  _out_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  _in_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  _out_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  _in_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  _out_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_samples);
  _p1 = fftwf_plan_dft_1d(n_samples, _in_xcorr1, _out_xcorr1, FFTW_FORWARD, FFTW_ESTIMATE);
  _p2 = fftwf_plan_dft_1d(n_samples, _in_xcorr2, _out_xcorr2, FFTW_FORWARD, FFTW_ESTIMATE);
  _p3 = fftwf_plan_dft_1d(n_samples, _in_xcorr3, _out_xcorr3, FFTW_BACKWARD, FFTW_ESTIMATE);
  _X1 =  (double *)malloc(2*n_samples *sizeof(double));
  _X2 =  (double *)malloc(2*n_samples *sizeof(double));
  _corr_freq =  (double *)malloc(2*n_samples *sizeof(double));
  _corr =  (double *)malloc(2*n_samples *sizeof(double));


  FILE *fp;
  fp = fopen("./input/time_input.txt","r");
  for(size_t i=0; i<time_n_samples; i++)
  {
          fscanf(fp,"%lf", &(time_lfm[i]));
  }
  fclose(fp);

  fp = fopen("./input/received_input.txt","r");

  for(size_t i=0; i<2*(n_samples); i++)
  {
          fscanf(fp,"%lf", &(received[i]));
  }
  fclose(fp);
  
  ///////////////accelerator////////////////////////////////
	dlhandle = dlopen("./libdash-rt.so", RTLD_LAZY);
	if (dlhandle == nullptr) {
		printf("Unable to open libdash-rt shared object!\n");
	}
	fft_accel_func = (void(*)(double**, double**, size_t*, bool*, unsigned int)) dlsym(dlhandle, "DASH_FFT_fft");
	if (fft_accel_func == nullptr) {
		printf("Unable to get function handle for FFT accelerator function!\n");
	}
  
  printf("[Correlator] intialization done\n");
  
}

#if defined(DEBUG)
void __attribute__((destructor)) clean_app(void) {
  printf("[Correlator] destroying variables\n");
  free(time_lfm);
  printf("Destroying received\n");
  free(received);
  printf("Destroying lfm_waveform\n");
  free(lfm_waveform);
  printf("Destroying X1\n");
  free(X1);
  printf("Destroying X2\n");
  free(X2);
  printf("Destroying corr_freq\n");
  free(corr_freq);
  printf("Destroying corr\n");
  free(corr);
  printf("Destroying p1\n");
  fftwf_destroy_plan(p1);
  printf("Destroying p2\n");
  fftwf_destroy_plan(p2);
  printf("Destroying p3\n");
  fftwf_destroy_plan(p3);
  printf("Destroying in_xcorr1\n");
  fftwf_free(in_xcorr1);
  printf("Destroying out_xcorr1\n");
  fftwf_free(out_xcorr1);
  printf("Destroying in_xcorr2\n");
  fftwf_free(in_xcorr2);
  printf("Destroying out_xcorr2\n");
  fftwf_free(out_xcorr2);
  printf("Destroying in_xcorr3\n");
  fftwf_free(in_xcorr3);
  printf("Destroying out_xcorr3\n");
  fftwf_free(out_xcorr3);

  printf("Destroying _lfm_waveform\n");
  free(_lfm_waveform);
  printf("Destroying _X1\n");
  free(_X1);
  printf("Destroying _X2\n");
  free(_X2);
  printf("Destroying _corr_freq\n");
  free(_corr_freq);
  printf("Destroying _corr\n");
  free(_corr);
  printf("Destroying _p1\n");
  fftwf_destroy_plan(_p1);
  printf("Destroying _p2\n");
  fftwf_destroy_plan(_p2);
  printf("Destroying _p3\n");
  fftwf_destroy_plan(_p3);
  printf("Destroying _in_xcorr1\n");
  fftwf_free(_in_xcorr1);
  printf("Destroying _out_xcorr1\n");
  fftwf_free(_out_xcorr1);
  printf("Destroying _in_xcorr2\n");
  fftwf_free(_in_xcorr2);
  printf("Destroying _out_xcorr2\n");
  fftwf_free(_out_xcorr2);
  printf("Destroying _in_xcorr3\n");
  fftwf_free(_in_xcorr3);
  printf("Destroying _out_xcorr3\n");
  fftwf_free(_out_xcorr3);
  printf("Destroying dlhandle\n");
  if (dlhandle != nullptr) {
    dlclose(dlhandle);
  }
  printf("[Correlator] destruction done\n");
}
#else
void __attribute__((destructor)) clean_app(void) {
  free(time_lfm);
  free(received);
  free(lfm_waveform);
  free(X1);
  free(X2);
  free(corr_freq);
  free(corr);
  fftwf_destroy_plan(p1);
  fftwf_destroy_plan(p2);
  fftwf_destroy_plan(p3);
  fftwf_free(in_xcorr1);
  fftwf_free(out_xcorr1);
  fftwf_free(in_xcorr2);
  fftwf_free(out_xcorr2);
  fftwf_free(in_xcorr3);
  fftwf_free(out_xcorr3);

  free(_lfm_waveform);
  free(_X1);
  free(_X2);
  free(_corr_freq);
  free(_corr);
  fftwf_destroy_plan(_p1);
  fftwf_destroy_plan(_p2);
  fftwf_destroy_plan(_p3);
  fftwf_free(_in_xcorr1);
  fftwf_free(_out_xcorr1);
  fftwf_free(_in_xcorr2);
  fftwf_free(_out_xcorr2);
  fftwf_free(_in_xcorr3);
  fftwf_free(_out_xcorr3);
  if (dlhandle != nullptr) {
    dlclose(dlhandle);
  }
  printf("[Correlator] destruction done\n");
}
#endif

void fftwf_fft(double *input_array, fftwf_complex *in, fftwf_complex *out, double *output_array, size_t n_elements, fftwf_plan p )
{
    for(size_t i = 0; i < n_elements; i++)
    {
        in[i][0] = input_array[2*i];
        in[i][1] = input_array[(2*i)+1];
    }
    fftwf_execute(p);
    for(size_t i = 0; i < n_elements; i++)
    {
        output_array[(2*i)] = out[i][0];
        output_array[(2*i)+1] = out[i][1];
    }
}

extern "C" void RD_head_node(void) {
}

extern "C" void RD_LFM(void) {
        static int iteration = 0;

        if (iteration%2) {
            for (size_t i = 0; i < (time_n_samples); i++){
                  lfm_waveform[(2*i)] = creal(cexp(I *  M_PI * (B/T) * pow((time_lfm[i]),2)));
                  lfm_waveform[(2*i)+1] = cimag(cexp(I *  M_PI * (B/T) * pow((time_lfm[i]),2)));
            }

            for(size_t i =(time_n_samples);i<2*(n_samples);i++){
                  lfm_waveform[i] = 0.0;
                }
        }
        else {
            for (size_t i = 0; i < time_n_samples; i++){
                  _lfm_waveform[2*i] = creal(cexp(I *  M_PI * (B/T) * pow((time_lfm[i]),2)));
                  _lfm_waveform[(2*i)+1] = cimag(cexp(I *  M_PI * (B/T) * pow((time_lfm[i]),2)));
                }

            for(size_t i =(time_n_samples);i<2*(n_samples);i++){
                  _lfm_waveform[i] = 0.0;
                }
            }
        
        iteration ++;
}


extern "C" void RD_FFT0_cpu(void) {
        static int iteration = 0;
        size_t len;

        #if defined(DEBUG)
        // DEBUG variables
        static char FUNC_NAME[20] = "FFT0_CPU";
        static char x1[10] = "X1";
        static char _x1[10] = "_X1";
        #endif

	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running FFT0 on CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
        len = 2 * (n_samples);
        double *c = (double *)malloc( len *sizeof(double));
        for(size_t i = 0; i< n_samples; i++) {
                c[(2*i)] = 0;
                c[(2*i)+1] = 0;
        }
        memcpy(c, received, 2*(n_samples)*sizeof(double));

                if (iteration%2) {
                  fftwf_fft(c, _in_xcorr1, _out_xcorr1,  X1, n_samples, _p1);
                  #if defined(DEBUG)
                  PRINT_ARRAY(X1, len, x1, FUNC_NAME, iteration);
                  #endif
                }
                else {
                  fftwf_fft(c, _in_xcorr1, _out_xcorr1,  _X1, n_samples, _p1);
                  #if defined(DEBUG)
                  PRINT_ARRAY(_X1, len, _x1, FUNC_NAME, iteration);
                  #endif
                }
        free(c);
        iteration ++;
}

extern "C" void RD_FFT0_acce(void) {
        static int iteration = 0;
        size_t len;
        bool isFwd;
        #if defined(DEBUG)
        // DEBUG variables
        static char FUNC_NAME[20] = "FFT0_acce";
        static char x1[10] = "X1";
        static char _x1[10] = "_X1";
        #endif

        printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running FFT0 on ACCELERATOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
        len = 2 * n_samples;
        isFwd = true;
        double *c = (double *)malloc( len *sizeof(double));
        for(size_t i = 0; i < n_samples; i++){
                c[(2*i)] = 0;
                c[(2*i)+1] = 0;
        }

        memcpy(c, received, 2*(n_samples)*sizeof(double));

        if (iteration%2) {
          (*fft_accel_func)(&c, &X1, &n_samples, &isFwd, __CEDR_CLUSTER_IDX__);
          #if defined(DEBUG)
          PRINT_ARRAY(X1, len, x1, FUNC_NAME, iteration);
          #endif
        }
        else {
          (*fft_accel_func)(&c, &_X1, &n_samples, &isFwd, __CEDR_CLUSTER_IDX__);
          #if defined(DEBUG)
          PRINT_ARRAY(_X1, len, _x1, FUNC_NAME, iteration);
          #endif
        }
        free(c);
        iteration++;
}


extern "C" void RD_FFT1_cpu(void) {
        static int iteration = 0;
        size_t len;

        #if defined(DEBUG)
        // DEBUG variables
        static char FUNC_NAME[20] = "FFT1_CPU";
        static char x2[10] = "X2";
        static char _x2[10] = "_X2";
        #endif

        printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running FFT1 on CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");



        len = 2 * (n_samples);
        double *d = (double *)malloc( len *sizeof(double));

        if (iteration%2) {
          memcpy(d, lfm_waveform, 2*(n_samples)*sizeof(double));
          fftwf_fft(d, in_xcorr2, out_xcorr2, X2, n_samples,p2);
          #if defined(DEBUG)
          PRINT_ARRAY(X2, len, x2, FUNC_NAME, iteration);
          #endif
        }
        else {
          memcpy(d, _lfm_waveform, 2*(n_samples)*sizeof(double));
          //memcpy(d+2*(n_samples), z, 2*(n_samples)*sizeof(double));
          fftwf_fft(d, _in_xcorr2, _out_xcorr2, _X2, n_samples,_p2);
          #if defined(DEBUG)
          PRINT_ARRAY(_X2, len, _x2, FUNC_NAME, iteration);
          #endif
        }

        free(d);
        iteration ++;
}


extern "C" void RD_FFT1_acce(void) {
        static int iteration = 0;
        size_t len;
        bool isFwd;
        printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running FFT1 on ACCELERATOR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");

        len = 2 * (n_samples);
        isFwd = true;
        double *d = (double *)malloc( len *sizeof(double));

        #if defined(DEBUG)
        // DEBUG variables
        static char FUNC_NAME[20] = "FFT1_acce";
        static char x2[10] = "X2";
        static char _x2[10] = "_X2";
        #endif

        if (iteration%2) {
          memcpy(d, lfm_waveform, 2*(n_samples)*sizeof(double));
          (*fft_accel_func)(&d, &X2, &n_samples, &isFwd, __CEDR_CLUSTER_IDX__);
          #if defined(DEBUG)
          PRINT_ARRAY(X2, len, x2, FUNC_NAME, iteration);
          #endif
        }
        else {
          memcpy(d, _lfm_waveform, 2*(n_samples)*sizeof(double));
          (*fft_accel_func)(&d, &_X2, &n_samples, &isFwd, __CEDR_CLUSTER_IDX__);
          #if defined(DEBUG)
          PRINT_ARRAY(_X2, len, _x2, FUNC_NAME, iteration);
          #endif
	      }

        free(d);
        iteration ++;

}


extern "C" void RD_MUL(void) {
        static int iteration = 0;

        printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running RD_MUL on CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
        if (iteration%2) {
          for(size_t i = 0 ;i < n_samples; i++){
                corr_freq[(2*i)] = (X1[2*i] * X2[2*i]) + (X1[(2*i)+1] * X2[(2*i)+1]);
                corr_freq[(2*i)+1] = (X1[(2*i)+1] * X2[2*i]) - (X1[2*i] * X2[(2*i)+1]);
              }
        }
        else {
          for(size_t i = 0 ;i < n_samples; i++){
                _corr_freq[2*i] = (_X1[2*i] * _X2[2*i]) + (_X1[(2*i)+1] * _X2[(2*i)+1]);
                _corr_freq[(2*i)+1] = (_X1[(2*i)+1] * _X2[2*i]) - (_X1[2*i] * _X2[(2*i)+1]);
          }
        }
        iteration ++;
}


extern "C" void RD_IFFT_cpu(void) {
        printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running RD_IFFT on CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
        static int iteration = 0;

        if (iteration%2) {
          fftwf_fft(corr_freq, in_xcorr3, out_xcorr3, corr, n_samples, p3);
        }
        else {
          fftwf_fft(_corr_freq, _in_xcorr3, _out_xcorr3, _corr, n_samples, _p3);
        }
        iteration ++;
}


extern "C" void RD_MAX(void) {
        printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running RD_MAX on CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");

        #if defined(DEBUG)
        static char FUNC_NAME[20] = "RD_MAX";
        static char CORR[10] = "corr";
        static char _CORR[10] = "_corr";
        #endif

        static int iteration = 0;
        int index =0;
	double lag;
	double max_corr = 0;
    
        if (iteration%2) {
          #if defined(DEBUG)
          PRINT_ARRAY(corr, 2*n_samples, CORR, FUNC_NAME, iteration);
          #endif
            for(size_t i = 0; i < n_samples; i++){
                if (corr[2*i] > max_corr){
                        max_corr = corr[2*i];
                        index = i;
                }
            }
        }

        else {
          #if defined(DEBUG)
          PRINT_ARRAY(_corr, 2*n_samples, _CORR, FUNC_NAME, iteration);
          #endif
            for(size_t i =0; i < n_samples; i++){
                if (_corr[2*i] > max_corr){
                        max_corr = _corr[2*i];
                        index = i;
                }
            }
        }

        lag = ((double)index - (double)n_samples)/sampling_rate;
        printf ("LAG value is %lf \n", lag);
        printf ("MAX index  %d and max value %lf \n", index, max_corr);
        
        iteration ++;
	printf("##########################################\n");
	printf("End of Radar Correlator Frame %d\n", iteration);
	printf("##########################################\n");
}
int main(void) {}
