#define _GNU_SOURCE
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <fftw3.h>
#include <complex.h>
#include <semaphore.h>
#include <time.h>
#include "common.h"
#include "radar.h"
#include <pthread.h>
#include <assert.h>
#define _GNU_SOURCE
#include <sched.h>

#define SEC2NANOSEC 1000000000
#define DEBUG 0

#include <sched.h>
#include <errno.h>

#ifdef PAPI
#include "papi.h"
void papi_init();
int elab_papi_end (const char *format, ...);
#endif

#define M_PI 3.14159265358979323846

unsigned int hex_data;
float rd_fft1_data[1024];
float rd_fft2_data[1024];
float rd_ifft_data[1024];

#include <acc_fft.h>

extern sem_t mutex;
sem_t        mutex;

/* Function Declarations */
//void fftwf_fft(float *, fftwf_complex *, fftwf_complex *, float *, size_t n_elements, fftwf_plan p );
void xcorr_LD (float *,float *,size_t, fftwf_complex *, fftwf_complex *, fftwf_complex *, fftwf_complex *, fftwf_complex *, fftwf_complex *, fftwf_plan, fftwf_plan, fftwf_plan, float *);

#ifndef THREAD_PER_TASK
void fftwf_fft(float *input_array, fftwf_complex *in, fftwf_complex *out, float *output_array, size_t n_elements, fftwf_plan p ) {
#else
void* fftwf_fft(void *input) {
    
    float *input_array = ((struct args_fftwf_fft *)input)->input_array;
    fftwf_complex *in = ((struct args_fftwf_fft *)input)->in;
    fftwf_complex *out = ((struct args_fftwf_fft *)input)->out;
    float *output_array = ((struct args_fftwf_fft *)input)->output_array;
    size_t n_elements = ((struct args_fftwf_fft *)input)->n_elements;
    fftwf_plan p = ((struct args_fftwf_fft *)input)->p;
#endif

    #ifdef DISPLAY_CPU_ASSIGNMENT
        printf("[INFO] Radar-FFT assigned to CPU: %d\n", sched_getcpu());
    #endif

    for(size_t i = 0; i < 2*n_elements; i+=2)
    {
        in[i/2][0] = input_array[i];
        in[i/2][1] = input_array[i+1];
    }
    fftwf_execute(p);
    for(size_t i = 0; i < 2*n_elements; i+=2)
    {
        output_array[i]   = out[i/2][0];
        output_array[i+1] = out[i/2][1];
        //printf("[DATA] hex_data = 0x%x; output_array[%d]   = *(float *)&hex_data;\n", *(unsigned int*)&out[i/2][0], i);
        //printf("[DATA] hex_data = 0x%x; output_array[%d+1] = *(float *)&hex_data;\n", *(unsigned int*)&out[i/2][1], i);
    }
        //printf("[DATA] ------------------\n");
}

#ifndef THREAD_PER_TASK
void lfm_waveform_fn(size_t time_n_samples, size_t n_samples, double *time, double *lfm_waveform, float *lfm_waveform_float, double T, double B) {
#else
void* lfm_waveform_fn(void *input) {

    size_t time_n_samples = ((struct args_lfm *)input)->time_n_samples;
    size_t n_samples = ((struct args_lfm *)input)->n_samples;
    double *time = ((struct args_lfm *)input)->time;
    double *lfm_waveform = ((struct args_lfm *)input)->lfm_waveform;
    float *lfm_waveform_float = ((struct args_lfm *)input)->lfm_waveform_float;
    double T = ((struct args_lfm *)input)->T;
    double B = ((struct args_lfm *)input)->B;
#endif

    #ifdef DISPLAY_CPU_ASSIGNMENT
        printf("[INFO] RD-LFM assigned to CPU: %d\n", sched_getcpu());
    #endif

	for (size_t i = 0; i < 2*time_n_samples; i+=2)
	{
		lfm_waveform[i]   = creal(cexp(I *  M_PI * B/T * pow(time[i/2],2)));
		lfm_waveform[i+1] = cimag(cexp(I *  M_PI * B/T * pow(time[i/2],2)));
	}

	for(size_t i =time_n_samples;i<2*n_samples;i++)
	{
		lfm_waveform[i] = 0.0;
	}

	for (size_t i = 0; i < 2*time_n_samples; i+=1)
	{
        lfm_waveform_float[i] = (float)lfm_waveform[i];
    }
}

#ifndef THREAD_PER_TASK
void conjugate(float *X1, float *X2, float *corr_freq, size_t len) {
#else
void* conjugate(void *input) {

    float *X1 = ((struct args_conjugate *)input)->in1;
    float *X2 = ((struct args_conjugate *)input)->in2;
    float *corr_freq = ((struct args_conjugate *)input)->out;
    size_t len = ((struct args_conjugate *)input)->len;
#endif
        #ifdef DISPLAY_CPU_ASSIGNMENT
            printf("[INFO] Radar-Conjugate assigned to CPU: %d\n", sched_getcpu());
        #endif

	for(size_t i =0;i<2*len;i+=2)
	{
		corr_freq[i] = (X1[i] * X2[i]) + (X1[i+1] * X2[i+1]);
		corr_freq[i+1] = (X1[i+1] * X2[i]) - (X1[i] * X2[i+1]);
	}
}

#ifndef THREAD_PER_TASK
void lag_detection(size_t n_samples, float *corr, float *max_corr, float *index, float *lag, float sampling_rate) {
#else
void* lag_detection(void *input) {

    size_t n_samples = ((struct args_lag_detection *)input)->n_samples;
    float *corr = ((struct args_lag_detection *)input)->corr;
    float *max_corr = ((struct args_lag_detection *)input)->max_corr;
    float *index = ((struct args_lag_detection *)input)->index;
    float *lag = ((struct args_lag_detection *)input)->lag;
    float sampling_rate = ((struct args_lag_detection *)input)->sampling_rate;
#endif

        #ifdef DISPLAY_CPU_ASSIGNMENT
            printf("[INFO] RD-Detection assigned to CPU: %d\n", sched_getcpu());
        #endif

	for(size_t i=0;i<2*(2*n_samples);i+=2)
	{
		// Only finding maximum of real part of correlation
		if (corr[i] > *max_corr)
		{
			*max_corr = corr[i];
			*index = i/2;
		}

	}

	*lag = (*index - n_samples)/sampling_rate;
}

void xcorr_LD( float *x, float *y, size_t n_samp, fftwf_complex *in1, fftwf_complex *out1, fftwf_complex *in2, fftwf_complex *out2, fftwf_complex *in3, fftwf_complex *out3, fftwf_plan p1, fftwf_plan p2, fftwf_plan p3, float *corr)
{

    //#######################################################################
	// X-corr-init
	//#######################################################################

    struct timespec start1, end1;
    float exec_time;
    #ifdef PRINT_BLOCK_EXECUTION_TIMES_XCORR 
    clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
    #endif

	size_t len;

    len = 2*n_samp;
    float *c = malloc( 2*len *sizeof(float));
	float *d = malloc( 2*len *sizeof(float));

    size_t x_count = 0;
	size_t y_count = 0;

    float *z = malloc( 2*(n_samp) *sizeof(float));
    for(size_t i = 0; i<2*(n_samp); i+=2)
    {
        z[i] = 0;
        z[i+1] = 0;
    }
    for(size_t i = 0; i<2*(n_samp-1); i+=2)
    {
        c[i] = 0;
        c[i+1] = 0;
    }
    memcpy(c+2*(n_samp), x, 2*n_samp*sizeof(float));
    c[2*len-2] = 0;
    c[2*len - 1] = 0;
    memcpy(d, y, 2*n_samp*sizeof(float));
    memcpy(d+2*n_samp, z, 2*(n_samp)*sizeof(float));
	float *X1 = malloc(2*len *sizeof(float));
	float *X2 = malloc(2*len *sizeof(float));
	float *corr_freq = malloc(2*len *sizeof(float));

    #ifdef PRINT_BLOCK_EXECUTION_TIMES_XCORR 
	clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
	printf("[ INFO] X-corr-init execution time (ns): %f\n", exec_time);
    #endif

    //#######################################################################
	// FFT-1
	//#######################################################################
    #ifdef PRINT_BLOCK_EXECUTION_TIMES_XCORR 
    clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
    #endif

    // Compiler directives to run it on hardware / A53
    #ifdef FFT1_HW
        if (sem_wait(&mutex) != 0) {
            printf("[ERROR] Semaphore wait failed ...\n");
            exit(-1);
        }

        fft_hs_PD(1, c, X1, len);
        
        if (sem_post(&mutex) != 0) {
            printf("[ERROR] Semaphore post failed ...\n");
            exit(-1);
        }
    #else
	    //fftwf_fft(c, in1, out1, X1, len, p1); 

        #ifndef THREAD_PER_TASK
	    #ifndef ACC_RD_FFT1
                fftwf_fft(c, in1, out1, X1, len, p1);
            #else
                memcpy(X1, rd_fft1_data, 1024*sizeof(float));
            #endif
        #else
        pthread_t thread_fft1;
        pthread_attr_t attr_thread_fft1;
        pthread_attr_init(&attr_thread_fft1);
        pthread_attr_setname(&attr_thread_fft1, "FFT");
        struct args_fftwf_fft *thread_param_fft1 = (struct args_fftwf_fft *)malloc(sizeof(struct args_fftwf_fft));
        thread_param_fft1->input_array  = c;
        thread_param_fft1->in           = in1;
        thread_param_fft1->out          = out1;
        thread_param_fft1->output_array = X1;
        thread_param_fft1->n_elements   = len;
        thread_param_fft1->p            = p1;
        assert(pthread_create(&thread_fft1, &attr_thread_fft1, fftwf_fft, (void *)thread_param_fft1) == 0);
        assert(pthread_join(thread_fft1, NULL) == 0);
        free(thread_param_fft1);
        #endif
    #endif

    #ifdef PRINT_BLOCK_EXECUTION_TIMES_XCORR 
	clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
	printf("[ INFO] FFT-1 execution time (ns): %f\n", exec_time);
    #endif

    //#######################################################################
	// FFT-2
	//#######################################################################
    #ifdef PRINT_BLOCK_EXECUTION_TIMES_XCORR 
    clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
    #endif

    // Compiler directives to run it on hardware / A53
    #ifdef FFT1_HW
        if (sem_wait(&mutex) != 0) {
            printf("[ERROR] Semaphore wait failed ...\n");
            exit(-1);
        }

        fft_hs_PD(1, d, X2, len);
        
        if (sem_post(&mutex) != 0) {
            printf("[ERROR] Semaphore post failed ...\n");
            exit(-1);
        }
    #else
        //fftwf_fft(d, in2, out2, X2, len, p2); 
        #ifndef THREAD_PER_TASK
	    #ifndef ACC_RD_FFT1
                fftwf_fft(d, in2, out2, X2, len, p2); 
            #else
                memcpy(X2, rd_fft2_data, 1024*sizeof(float));
            #endif
        #else
        pthread_t thread_fft2;
        pthread_attr_t attr_thread_fft2;
        pthread_attr_init(&attr_thread_fft2);
        pthread_attr_setname(&attr_thread_fft2, "FFT");
        struct args_fftwf_fft *thread_param_fft2 = (struct args_fftwf_fft *)malloc(sizeof(struct args_fftwf_fft));
        thread_param_fft2->input_array  = d;
        thread_param_fft2->in           = in2;
        thread_param_fft2->out          = out2;
        thread_param_fft2->output_array = X2;
        thread_param_fft2->n_elements   = len;
        thread_param_fft2->p            = p2;
        assert(pthread_create(&thread_fft2, &attr_thread_fft2, fftwf_fft, (void *)thread_param_fft2) == 0);
        assert(pthread_join(thread_fft2, NULL) == 0);
        free(thread_param_fft2);
        #endif

    #endif

    #ifdef PRINT_BLOCK_EXECUTION_TIMES_XCORR 
    clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
	printf("[ INFO] FFT-2 execution time (ns): %f\n", exec_time);
    #endif

    free(c);
    free(d);
    free(z);

	//if (DEBUG == 1)	{
	//	for (int i = 0; i <  (2*len); i=i+2) {
	//		printf("X1 index %d real: %lf  imag %lf \n", i/2, X1[i],X1[i+1]);
	//	}
	//	for (int i = 0; i <  (2*len); i=i+2) {
	//		printf("X2 index %d real: %lf  imag %lf \n", i/2, X2[i],X2[i+1]);
	//	}
	//}

    //#######################################################################
	// Complex Conjugate
	//#######################################################################
    #ifdef PRINT_BLOCK_EXECUTION_TIMES_XCORR 
    clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
    #endif
    
    #ifndef THREAD_PER_TASK
    conjugate(X1, X2, corr_freq, len);
    #else
    pthread_t thread_conjugate;
    pthread_attr_t attr_thread_conjugate;
    pthread_attr_init(&attr_thread_conjugate);
    pthread_attr_setname(&attr_thread_conjugate, "conjugate");
    struct args_conjugate *thread_param_conjugate = (struct args_conjugate *)malloc(sizeof(struct args_conjugate));
    thread_param_conjugate->in1 = X1;
    thread_param_conjugate->in2 = X2;
    thread_param_conjugate->out = corr_freq;
    thread_param_conjugate->len = len;
    assert(pthread_create(&thread_conjugate, &attr_thread_conjugate, conjugate, (void *)thread_param_conjugate) == 0);
    assert(pthread_join(thread_conjugate, NULL) == 0);
    free(thread_param_conjugate);
    #endif

    #ifdef PRINT_BLOCK_EXECUTION_TIMES_XCORR 
    clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
	printf("[ INFO] ComplexConj execution time (ns): %f\n", exec_time);
    #endif

    free(X1);
    free(X2);
    
    //#######################################################################
	// IFFT
	//#######################################################################
    #ifdef PRINT_BLOCK_EXECUTION_TIMES_XCORR 
    clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
    #endif

    // Compiler directives to run it on hardware / A53
    #ifdef FFT1_HW
        if (sem_wait(&mutex) != 0) {
            printf("[ERROR] Semaphore wait failed ...\n");
            exit(-1);
        }

        ifft_hs_PD(1, corr_freq, corr, len);
        
        if (sem_post(&mutex) != 0) {
            printf("[ERROR] Semaphore post failed ...\n");
            exit(-1);
        }
    #else
	    //fftwf_fft(corr_freq, in3, out3, corr, len, p3);
        #ifndef THREAD_PER_TASK
	    #ifndef ACC_RD_IFFT
	        fftwf_fft(corr_freq, in3, out3, corr, len, p3);
            #else
                memcpy(corr, rd_ifft_data, 1024*sizeof(float));
            #endif
        #else
        pthread_t thread_ifft;
        pthread_attr_t attr_thread_ifft;
        pthread_attr_init(&attr_thread_ifft);
        pthread_attr_setname(&attr_thread_ifft, "FFT");
        struct args_fftwf_fft *thread_param_ifft = (struct args_fftwf_fft *)malloc(sizeof(struct args_fftwf_fft));
        thread_param_ifft->input_array  = corr_freq;
        thread_param_ifft->in           = in3;
        thread_param_ifft->out          = out3;
        thread_param_ifft->output_array = corr;
        thread_param_ifft->n_elements   = len;
        thread_param_ifft->p            = p3;
        assert(pthread_create(&thread_ifft, &attr_thread_ifft, fftwf_fft, (void *)thread_param_ifft) == 0);
        assert(pthread_join(thread_ifft, NULL) == 0);
        free(thread_param_ifft);
        #endif
              
    #endif

    #ifdef PRINT_BLOCK_EXECUTION_TIMES_XCORR 
    clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
	printf("[ INFO] IFFT execution time (ns): %f\n", exec_time);
    #endif

	//if (DEBUG == 1)	{
	//	for (int i = 0; i <  (2*len); i=i+2) {
	//		corr[i] = corr[i]/len;
	//		corr[i+1] = corr[i+1]/len;
	//		printf("corr index %d real: %lf  imag %lf \n", i/2, corr[i],corr[i+1]);
	//	}
	//}
    
    free(corr_freq);

}


//int main(int argc, char *argv[]) {
void* range_detection() {
  
    //#ifdef FFT1_HW
    //    // Virtual Address to DMA Control Slave
    //    init_dma1();

    //    init_fft1();

    //    // Virtual Address to udmabuf Buffer
    //    init_udmabuf1();

    //#endif

    //#################################################################
    //## Initialize Semaphores, DMA and FFT IPs
    //#################################################################
        
    //if (sem_init(&mutex, 0, 1) != 0) {
    //    printf("[ERROR] Semaphore creation failed ...\n");
    //    exit(-1);
    //}
    
	size_t n_samples;
	size_t time_n_samples;
	double T;
	double B;
	double sampling_rate;
    int value = 2;

    if (value == 1) {
	    n_samples      = atoi("512");
	    time_n_samples = atoi("3");
	    T              = atof("128/500000");
	    B              = atof("500000");
	    sampling_rate  = atof("10000");
    } else {
	    n_samples      = atoi("63");
	    time_n_samples = atoi("1");
	    T              = atof("32/500000");
	    B              = atof("500000");
	    sampling_rate  = atof("1000");
    }

	//size_t n_samples      = int(256)          ; //atoi(argv[1]);
	//size_t time_n_samples = int(1)            ; //atoi(argv[2]);
	//double T              = float(256/500000) ; //atof(argv[3]);
	//double B              = float(500000)     ; //atof(argv[4]);
	//double sampling_rate  = float(1000)       ; //atof(argv[5]);

	//size_t n_samples      = atoi("256");
	//size_t time_n_samples = atoi("1");
	//float T               = atof("256/500000");
	//float B               = atof("500000");
	//float sampling_rate   = atof("1000");

	double *time = malloc(n_samples*sizeof(double));;
	float *received = malloc(2*n_samples*sizeof(float));
	
    struct timespec start1, end1;
    double exec_time;
    double total_exec_time = 0;
	
	// DASH_DATA
	if( !getenv("DASH_DATA") )
	{
		printf("in range_detection.c:\n\tFATAL: DASH_DATA is not set. Exiting...");
		exit(1);
	}

	char* file6 = "Dash-RadioCorpus/QPR8_RFConvSys/time_input.txt";
	char* path6 = (char* )malloc(FILEPATH_SIZE*sizeof(char) );
	strcat(path6, getenv("DASH_DATA"));
	strcat(path6, file6);
	FILE* fp6 = fopen(path6, "r");
	free(path6);

    if(fp6 == NULL) {
        printf("in range_detection.c:\n\tFATAL: %s was not found!\n", file6);
        exit(1);
    }

	for(size_t i=0; i<time_n_samples; i++) 
	{
		fscanf(fp6,	"%lf", &time[i]);
	}	
	fclose(fp6);

	char* path7 = (char* )malloc( FILEPATH_SIZE*sizeof(char) );
	char* file7 = "Dash-RadioCorpus/QPR8_RFConvSys/received_input.txt";
	strcat(path7, getenv("DASH_DATA"));
	strcat(path7, file7);
	FILE* fp7 = fopen(path7, "r");
	free(path7);

    if(fp7 == NULL) {
        printf("in range_detection.c:\n\tFATAL: %s was not found!\n", file7);
        exit(1);
    }

	for(size_t i=0; i<2*n_samples; i++) 
	{
		fscanf(fp7,"%f", &received[i]);
	}	
	fclose(fp7);

	float lag;
	float *corr = malloc( (2*(2*n_samples)) * sizeof(float));

	double *lfm_waveform = malloc( 2*n_samples * sizeof(double));	
	float *lfm_waveform_float = malloc( 2*n_samples * sizeof(float));	   
 
    #ifdef THREAD_PER_TASK 
    pthread_t thread_lfm;
    pthread_attr_t attr_thread_lfm;
    pthread_attr_init(&attr_thread_lfm);
    pthread_attr_setname(&attr_thread_lfm, "LFM");

    pthread_t thread_lag_detection;
    pthread_attr_t attr_thread_lag_detection;
    pthread_attr_init(&attr_thread_lag_detection);
    pthread_attr_setname(&attr_thread_lag_detection, "lag_detection");
    #endif

    //#######################################################################
	// LFM Waveform Generator
	//#######################################################################

    #ifdef ACC_RD_FFT1 
        get_rd_fft1_data();
    #endif

    #ifdef ACC_RD_FFT2 
        get_rd_fft2_data();
    #endif

    #ifdef ACC_RD_IFFT 
        get_rd_ifft_data();
    #endif

    //create plans for FFT in matched filter
    
    fftwf_complex *in_xcorr1, *out_xcorr1, *in_xcorr2, *out_xcorr2, *in_xcorr3, *out_xcorr3;
    in_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
    out_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
    in_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
    out_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
    in_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
    out_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
    fftwf_plan p1, p2, p3;
    p1 = fftwf_plan_dft_1d((2*n_samples), in_xcorr1, out_xcorr1, FFTW_FORWARD, FFTW_ESTIMATE);
    p2 = fftwf_plan_dft_1d((2*n_samples), in_xcorr2, out_xcorr2, FFTW_FORWARD, FFTW_ESTIMATE);
    p3 = fftwf_plan_dft_1d((2*n_samples), in_xcorr3, out_xcorr3, FFTW_BACKWARD, FFTW_ESTIMATE);

    #ifdef PRINT_BLOCK_EXECUTION_TIMES 
    clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
    #endif

    #ifndef THREAD_PER_TASK
    lfm_waveform_fn(time_n_samples, n_samples, time, lfm_waveform, lfm_waveform_float, T, B);
    #else
    struct args_lfm *thread_param_lfm = (struct args_lfm *)malloc(sizeof(struct args_lfm));
    thread_param_lfm->time_n_samples = time_n_samples;
    thread_param_lfm->n_samples = n_samples;
    thread_param_lfm->time = time;
    thread_param_lfm->lfm_waveform = lfm_waveform;
    thread_param_lfm->lfm_waveform_float = lfm_waveform_float;
    thread_param_lfm->T = T;
    thread_param_lfm->B = B;
    assert(pthread_create(&thread_lfm, &attr_thread_lfm, lfm_waveform_fn, (void *)thread_param_lfm) == 0);
    assert(pthread_join(thread_lfm, NULL) == 0);
    free(thread_param_lfm);
    #endif

    #ifdef PRINT_BLOCK_EXECUTION_TIMES 
	clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
	printf("[ INFO] LFM-Waveform execution time (ns): %f\n", exec_time);
    #else
    total_exec_time = total_exec_time + exec_time;
    #endif

    //#######################################################################
	// X-Corr
	//#######################################################################
    clock_gettime(CLOCK_MONOTONIC_RAW, &start1);

    xcorr_LD(received, lfm_waveform_float, n_samples, in_xcorr1, out_xcorr1, in_xcorr2, out_xcorr2, in_xcorr3, out_xcorr3, p1, p2, p3, corr);

	clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
    #ifdef PRINT_BLOCK_EXECUTION_TIMES 
	printf("[ INFO] X-corr execution time (ns): %f\n", exec_time);
    #else
    total_exec_time += exec_time;
    #endif

	//Code to find maximum
	float max_corr = 0;
	float index = 0;
    
    //#######################################################################
	// Detection
	//#######################################################################
    #ifdef PRINT_BLOCK_EXECUTION_TIMES 
    clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
    #endif

    #ifndef THREAD_PER_TASK
    lag_detection(n_samples, corr, &max_corr, &index, &lag, sampling_rate);
    #else
    struct args_lag_detection *thread_param_lag_detection = (struct args_lag_detection *)malloc(sizeof(struct args_lag_detection));
    thread_param_lag_detection->n_samples = n_samples;
    thread_param_lag_detection->corr = corr;
    thread_param_lag_detection->max_corr = &max_corr;
    thread_param_lag_detection->index = &index;
    thread_param_lag_detection->lag = &lag;
    thread_param_lag_detection->sampling_rate = sampling_rate;
    assert(pthread_create(&thread_lag_detection, &attr_thread_lag_detection, lag_detection, (void *)thread_param_lag_detection) == 0);
    assert(pthread_join(thread_lag_detection, NULL) == 0);
    free(thread_param_lag_detection);
    #endif

    #ifndef PAPI
    #ifdef PRINT_BLOCK_EXECUTION_TIMES 
	clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));

	printf("[ INFO] Detection execution time (ns): %f\n", exec_time);
    #else
    total_exec_time += exec_time;
	printf("[ INFO] Lag-Detection execution time (ns): %f\n", total_exec_time);
    #endif

	printf ("[ INFO] ARM LAG value is %f \n", lag);
	printf ("[ INFO] ARM MAX index  %f and max value %f \n", index, max_corr);
    #endif

    //fftwf_destroy_plan(p1);
    //fftwf_destroy_plan(p2);
    //fftwf_destroy_plan(p3);
    fftwf_free(in_xcorr1);
    fftwf_free(out_xcorr1);
    fftwf_free(in_xcorr2);
    fftwf_free(out_xcorr2);
    fftwf_free(in_xcorr3);
    fftwf_free(out_xcorr3);
    fftwf_cleanup();
	
    free(corr);
    free(lfm_waveform);
    free(lfm_waveform_float);
    free(time);
    free(received);
    //#################################################################
    //## Destroy Semaphore
    //#################################################################
        
    //if (sem_destroy(&mutex) != 0) {
    //    printf("[ERROR] Semaphore destroy failed ...\n");
    //    exit(-1);
    //}
    
}
