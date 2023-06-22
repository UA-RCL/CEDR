#include <iostream>
#include <cstdio>
#include "correlator.hpp"
#include <math.h>
#include <unistd.h>
#include <fftw3.h>
#include <complex.h>
#include <stdlib.h>
#include <cstring>

int iter;
size_t n_samples;
size_t time_n_samples;
double T;
double B;
float sampling_rate;
double *time_lfm;
float *received;
float *lfm_waveform;
fftwf_complex *in_xcorr1, *out_xcorr1, *in_xcorr2, *out_xcorr2, *in_xcorr3, *out_xcorr3;
fftwf_plan p1, p2, p3;
float *X1, *X2;
float *corr_freq;
float *corr;



void __attribute__((constructor)) setup(void) {
  printf("[Correlator] intializing variables\n");

  iter = 0;  
  n_samples=256;
  time_n_samples =1;
  T = (float)(256.0/500000);
  B = (float)500000;
  sampling_rate = 1000;
  time_lfm = (double *) malloc((n_samples)*sizeof(double));
  received = (float *)malloc(2*(n_samples)*sizeof(float));
  lfm_waveform = (float *)malloc(2*(n_samples)*sizeof(float));
  in_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
  out_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
  in_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
  out_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
  in_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
  out_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
  p1 = fftwf_plan_dft_1d((2*(n_samples)), in_xcorr1, out_xcorr1, FFTW_FORWARD, FFTW_ESTIMATE);
  p2 = fftwf_plan_dft_1d((2*(n_samples)), in_xcorr2, out_xcorr2, FFTW_FORWARD, FFTW_ESTIMATE);
  p3 = fftwf_plan_dft_1d((2*(n_samples)), in_xcorr3, out_xcorr3, FFTW_BACKWARD, FFTW_ESTIMATE);
  X1 =  (float *)malloc(2*(2*(n_samples)) *sizeof(float));
  X2 =  (float *)malloc(2*(2*(n_samples)) *sizeof(float));
  corr_freq =  (float *)malloc(2*(2*(n_samples)) *sizeof(float));
  corr =  (float *)malloc(2*(2*(n_samples)) *sizeof(float));


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
          fscanf(fp,"%f", &(received[i]));
  }
  fclose(fp);
  
  

  printf("[Correlator] intialization done\n");
  
}


void __attribute__((destructor)) clean_app(void) {
  printf("[Correlator] destroying variables\n");
  free(time_lfm);
  //printf("task:start cleaning up lag detection\n");
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
 
  printf("[Correlator] destruction done\n");
  
}

void fftwf_fft(float *input_array, fftwf_complex *in, fftwf_complex *out, float *output_array, size_t n_elements, fftwf_plan p )
{

    for(size_t i = 0; i < 2*n_elements; i+=2)
    {
        in[i/2][0] = input_array[i];
        in[i/2][1] = input_array[i+1];
    }
    fftwf_execute(p);
    for(size_t i = 0; i < 2*n_elements; i+=2)
    {
        output_array[i] = out[i/2][0];
        output_array[i+1] = out[i/2][1];
    }
}





extern "C" void RD_head_node(void) {
}

extern "C" void RD_LFM(void) {
        for (size_t i = 0; i < 2*(time_n_samples); i+=2){
                lfm_waveform[i] = creal(cexp(I *  M_PI * (B/T) * pow((time_lfm[i/2]),2)));
                lfm_waveform[i+1] = cimag(cexp(I *  M_PI * (B/T) * pow((time_lfm[i/2]),2)));

        }
        for(size_t i =(time_n_samples);i<2*(n_samples);i++){
                lfm_waveform[i] = 0.0;
        }

}


extern "C" void RD_FFT0(void) {
        size_t len;

	//fftwf_complex *in_xcorr1, *out_xcorr1;
	//fftwf_plan p1;
  	//in_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
  	//out_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
	//p1 = fftwf_plan_dft_1d((2*(n_samples)), in_xcorr1, out_xcorr1, FFTW_FORWARD, FFTW_ESTIMATE);

        len = 2 * (n_samples);
        float *c = (float *)malloc( 2*len *sizeof(float));
        for(size_t i = 0; i<2*(n_samples-1); i+=2){
                c[i] = 0;
                c[i+1] = 0;
        }
        memcpy(c+2*(n_samples), received, 2*(n_samples)*sizeof(float));
        c[2*len-2] = 0;
        c[2*len - 1] = 0;
                fftwf_fft(c, in_xcorr1, out_xcorr1,  X1, len, p1);
        free(c);
  	//fftwf_destroy_plan(p1);
  	//fftwf_free(in_xcorr1);
  	//fftwf_free(out_xcorr1);


}



extern "C" void RD_FFT1(void) {
        size_t len;

	//fftwf_complex *in_xcorr2, *out_xcorr2;
	//fftwf_plan p2;
  	//in_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
  	//out_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
	//p2 = fftwf_plan_dft_1d((2*(n_samples)), in_xcorr2, out_xcorr2, FFTW_FORWARD, FFTW_ESTIMATE);


        len = 2 * (n_samples);
        float *d = (float *)malloc( 2*len *sizeof(float));
        float *z = (float *)malloc( 2*(n_samples) *sizeof(float));
        for(size_t i = 0; i<2*(n_samples); i+=2){
                z[i] = 0;
                z[i+1] = 0;
        }
        memcpy(d, lfm_waveform, 2*(n_samples)*sizeof(float));
        memcpy(d+2*(n_samples), z, 2*(n_samples)*sizeof(float));
        fftwf_fft(d, in_xcorr2, out_xcorr2, X2, len,p2);

        free(d);
        free(z);
  	//fftwf_destroy_plan(p2);
  	//fftwf_free(in_xcorr2);
  	//fftwf_free(out_xcorr2);



}


extern "C" void RD_MUL(void) {
        size_t len;
        len = 2 * (n_samples);
        for(size_t i =0;i<2*len;i+=2){
                corr_freq[i] = (X1[i] * X2[i]) + (X1[i+1] * X2[i+1]);
                corr_freq[i+1] = (X1[i+1] * X2[i]) - (X1[i] * X2[i+1]);
        }

}



extern "C" void RD_IFFT(void) {
        size_t len;
	//fftwf_complex *in_xcorr3, *out_xcorr3;
	//fftwf_plan p3;
  	//in_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
  	//out_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(n_samples)));
  	//p3 = fftwf_plan_dft_1d((2*(n_samples)), in_xcorr3, out_xcorr3, FFTW_BACKWARD, FFTW_ESTIMATE);

        len = 2 * (n_samples);
        fftwf_fft(corr_freq, in_xcorr3, out_xcorr3, corr, len, p3);

  	//fftwf_destroy_plan(p3);
  	//fftwf_free(in_xcorr3);
  	//fftwf_free(out_xcorr3);
}


extern "C" void RD_MAX(void) {

        int index =0;
	float lag;
	float max_corr = 0;


        for(size_t i =0;i<2*(2*(n_samples));i+=2){
                if (corr[i] > max_corr){
                        max_corr = corr[i];
                        index = i/2;
                }


        }

        lag = (index - n_samples)/sampling_rate;
        printf ("LAG value is %f \n", lag);
        printf ("MAX index  %d and max value %f \n", index, max_corr);


}
int main(void) {}
