#define _GNU_SOURCE
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <fftw3.h>
#include <time.h>
#include <semaphore.h>
#include "common.h"
#include "radar.h"
#include <pthread.h>
#include <assert.h>
#include <sched.h>

#define SEC2NANOSEC 1000000000

#ifdef PAPI
#include <papi.h>
void papi_init();
int elab_papi_end (const char *format, ...);
#endif

//###############################################################
//## Variable Declarations for FFT HW Implementation
//###############################################################

extern int          dma1_control_fd;
extern unsigned int *dma1_control_base_addr;
extern int          fd_udmabuf1;
extern float        *udmabuf1_base_addr;
extern unsigned int udmabuf1_phys_addr;

extern int          dma2_control_fd;
extern unsigned int *dma2_control_base_addr;
extern int          fd_udmabuf2;
extern float        *udmabuf2_base_addr;
extern unsigned int udmabuf2_phys_addr;

int                 dma1_control_fd;
unsigned int        *dma1_control_base_addr;
int                 fd_udmabuf1;
float               *udmabuf1_base_addr;
unsigned int        udmabuf1_phys_addr;

int                 dma2_control_fd;
unsigned int        *dma2_control_base_addr;
int                 fd_udmabuf2;
float               *udmabuf2_base_addr;
unsigned int        udmabuf2_phys_addr;

extern sem_t mutex;
sem_t        mutex;

/* Function Declarations */
#ifndef THREAD_PER_TASK
void fftwf_fft(float *input_array, fftwf_complex *in, fftwf_complex *out, float *output_array, size_t n_elements, fftwf_plan p );
#else
void* fftwf_fft(void *input);
#endif

#ifndef THREAD_PER_TASK
void conjugate(float *X1, float *X2, float *corr_freq, size_t len);
#else
void* conjugate(void *input);
#endif

#ifndef THREAD_PER_TASK
void amplitude(size_t m, float *q, float *r) {
#else
void* amplitude(void *input) {
    size_t m = ((struct args_amplitude *)input)->m;
    float *q = ((struct args_amplitude *)input)->q;
    float *r = ((struct args_amplitude *)input)->r;
#endif

    #ifdef DISPLAY_CPU_ASSIGNMENT
        printf("[INFO] PD-Amplitude assigned to CPU: %d\n", sched_getcpu());
    #endif

    for(int y = 0; y < 2*m; y+=2)
    {
        r[y/2] = sqrt(q[y]*q[y]+q[y+1]*q[y+1]);   // calculate the absolute value of the output 
    }
}

void swap(float *, float *);

#ifndef THREAD_PER_TASK
void fftshift(float *data, float count) {
#else
void* fftshift(void *input) {

    float *data = ((struct args_fftshift *)input)->data;
    size_t count = ((struct args_fftshift *)input)->count;
#endif

    #ifdef DISPLAY_CPU_ASSIGNMENT
        printf("[INFO] PD-Shift assigned to CPU: %d\n", sched_getcpu());
    #endif

    int k = 0;
    int c = (float) floor((float)count/2);
    // For odd and for even numbers of element use different algorithm
    if ((int)count % 2 == 0)
    {
        for (k = 0; k < c; k++)
            swap(&data[k], &data[k+c]);
    }
    else
    {
        float tmp = data[0];
        for (k = 0; k < c; k++)
        {
            data[k] = data[c + k + 1];
            data[c + k + 1] = data[k + 1];
        }
        data[c] = tmp;
    }
}

void xcorr (float *,float *,size_t, fftwf_complex *, fftwf_complex *, fftwf_complex *, fftwf_complex *, fftwf_complex *, fftwf_complex *, fftwf_plan, fftwf_plan, fftwf_plan, float *);
void xcorr( float *x, float *y, size_t n_samp, fftwf_complex *in1, fftwf_complex *out1, fftwf_complex *in2, fftwf_complex *out2, fftwf_complex *in3, fftwf_complex *out3, fftwf_plan p1, fftwf_plan p2, fftwf_plan p3, float *corr)
{

    //#######################################################################
	// X-corr-init
	//#######################################################################

    struct timespec start1, end1;
    #ifdef PRINT_BLOCK_EXECUTION_TIMES_XCORR 
    clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
    #endif

	size_t len;
    double exec_time;
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
    memcpy(c+2*(n_samp-1), x, 2*n_samp*sizeof(float));
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

	//clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
	//fftwf_fft(c, in1, out1, X1, len, p1); 
	//clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	//exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
	//printf("[ INFO] ARM FFT length: %d, execution time (ns): %f\n",len , exec_time);
 
 
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
      
        #ifndef THREAD_PER_TASK
	    fftwf_fft(c, in1, out1, X1, len, p1);
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
    
        #ifndef THREAD_PER_TASK
        fftwf_fft(d, in2, out2, X2, len, p2); 
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
    
        #ifndef THREAD_PER_TASK
	    fftwf_fft(corr_freq, in3, out3, corr, len, p3);
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

    free(corr_freq);
}

void swap(float *v1, float *v2)
{
    float tmp = *v1;
    *v1 = *v2;
    *v2 = tmp;
}

void* pulse_doppler() {
    
	size_t m         = 64;     //atoi(argv[1]);                               // number of pulses
	size_t n_samples = 32;     //atoi(argv[2]);                       // length of single pulse
    float PRI        = 1.27e-4; //atoi(argv[3]);
    int i, j, k, n, x, y, z, o;
    
    float *mf    = malloc((2*n_samples)*m*2*sizeof(float)); // build a 2D array for the output of the matched filter
    float *p     = malloc(2*n_samples *sizeof(float));   // array for pulse with noise
    float *pulse = malloc(2*n_samples *sizeof(float));  // array for the original pulse
    float *corr  = malloc(2*(2*n_samples) *sizeof(float));   // array for the output of matched filter
    struct timespec start1, end1;
	double exec_time;

    //#################################################################
    //## Initialize Semaphores, DMA and FFT IPs
    //#################################################################
        
    //#ifdef FFT1_HW
    //    // Virtual Address to DMA Control Slave
    //    init_dma1();

    //    init_fft1();
    //    
    //    // Virtual Address to udmabuf Buffer
    //    init_udmabuf1();

    //#endif

    fftwf_complex *in_xcorr1, *out_xcorr1, *in_xcorr2, *out_xcorr2, *in_xcorr3, *out_xcorr3;
    fftwf_plan p1, p2, p3;
    in_xcorr1  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
    out_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
    in_xcorr2  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
    out_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
    in_xcorr3  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
    out_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*n_samples));
	
    float total_exec_time = 0;
    
    //#######################################################################
	// Create plans for FFT - FFTW library
	//#######################################################################
    
    #ifndef FFT1_HW

        clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
        
        p1 = fftwf_plan_dft_1d((2*n_samples), in_xcorr1, out_xcorr1, FFTW_FORWARD, FFTW_ESTIMATE);
        p2 = fftwf_plan_dft_1d((2*n_samples), in_xcorr2, out_xcorr2, FFTW_FORWARD, FFTW_ESTIMATE);
        p3 = fftwf_plan_dft_1d((2*n_samples), in_xcorr3, out_xcorr3, FFTW_BACKWARD, FFTW_ESTIMATE);

	    clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	    exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
        
        #ifdef PRINT_BLOCK_EXECUTION_TIMES 
	    printf("[ INFO] FFT-Prep-1 execution time (ns): %f\n", exec_time);
        #endif

        total_exec_time += exec_time;

    #endif

	// DASH_DATA
	if( !getenv("DASH_DATA") )
	{
		printf("in pulse_doppler.c:\n\tFATAL: DASH_DATA is not set. Exiting...");
		exit(1);
	}

	char* file0 = "Dash-RadioCorpus/wifi/input_pd_pulse.txt";
	char* path0 = (char* )malloc( FILEPATH_SIZE*sizeof(char) );
	strcat(path0, getenv("DASH_DATA"));
	strcat(path0, file0);
	FILE* fp0 = fopen(path0, "r");
	free(path0);

    if(fp0 == NULL) {
        printf("in pulse_doppler.c:\n\tFATAL: %s not found!\n", file0);
        exit(1);
    }

	char* file1 = "Dash-RadioCorpus/wifi/input_pd_ps.txt";
	char* path1 = (char* )malloc( FILEPATH_SIZE*sizeof(char) );
	strcat(path1, getenv("DASH_DATA"));
	strcat(path1, file1);
	FILE* fp1 = fopen(path1, "r");
	free(path1);

    if(fp1 == NULL) {
        printf("in pulse_doppler.c:\n\tFATAL: %s not found!\n", file1);
        exit(1);
    }

	// read file0
    for(i=0; i<2*n_samples; i++) 
	{
		fscanf(fp0, "%f", &pulse[i]);
	}
    fclose(fp0);
    
	//#######################################################################
	// Match Filter
	//#######################################################################

    for(k = 0; k < m; k++)
    {        
		// read file1
        for(j = 0; j < 2 * n_samples; j++)
        {
            fscanf(fp1, "%f", &p[j]);
        }
        
	    //#######################################################################
	    // Cross-Correlator
	    //#######################################################################
        clock_gettime(CLOCK_MONOTONIC_RAW, &start1);

        xcorr(p, pulse, n_samples, in_xcorr1, out_xcorr1, in_xcorr2, out_xcorr2, in_xcorr3, out_xcorr3, p1, p2, p3, corr);
        
        // Put output into 2D array
        for(n = 0; n < 2*(2 * n_samples); n+=2)
        {
            mf[n/2+(2*k)*(2*n_samples)] = corr[n];
            mf[n/2+(2*k+1)*(2*n_samples)] = corr[n+1];
        }

	    clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	    exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
        #ifdef PRINT_BLOCK_EXECUTION_TIMES 
	    printf("[ INFO] X-corr execution time (ns): %f\n", exec_time);
        #else
        total_exec_time += exec_time;
        #endif

    }
    fclose(fp1);
    free(p);
    free(pulse);
    free(corr);
    
    #ifndef PRINT_BLOCK_EXECUTION_TIMES 
    clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
    #endif
    
    float *q = malloc(2*m*sizeof(float)); 
    float *r = malloc(m*sizeof(float)); 
    float *l = malloc(2*m*sizeof(float));
    float *f = malloc(m*(2*n_samples)*sizeof(float));
    float max = 0, a, b;
    fftwf_complex *in_fft, *out_fft;
    in_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * m);
    out_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * m);
    fftwf_plan p4;

    #ifndef FFT1_HW

        clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
        
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

        p4 = fftwf_plan_dft_1d(m, in_fft, out_fft, FFTW_FORWARD, FFTW_ESTIMATE);
	
        clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	    exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
        
        #ifdef PRINT_BLOCK_EXECUTION_TIMES 
	    printf("[ INFO] FFT-Prep-2 execution time (ns): %f\n", exec_time);
        #endif

        total_exec_time += exec_time;

    #endif
    
    for(x = 0; x < 2*n_samples; x++)
    {
        for(o = 0; o < 2*m; o++)
        {
            l[o] = mf[x+o*(2*n_samples)];
        }

	    //#######################################################################
	    // FFT-3
	    //#######################################################################
        #ifdef PRINT_BLOCK_EXECUTION_TIMES 
        clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
        #endif

        // Compiler directives to run it on hardware / A53
        #ifdef FFT1_HW
            if (sem_wait(&mutex) != 0) {
                printf("[ERROR] Semaphore wait failed ...\n");
                exit(-1);
            }

            fft_hs_PD(1, l, q, m);

            if (sem_post(&mutex) != 0) {
                printf("[ERROR] Semaphore post failed ...\n");
                exit(-1);
            }
        #else
        
            #ifndef THREAD_PER_TASK
	        fftwf_fft(l, in_fft, out_fft, q, m, p4);
            #else
            pthread_t thread_fft3;
            pthread_attr_t attr_thread_fft3;
            pthread_attr_init(&attr_thread_fft3);
            pthread_attr_setname(&attr_thread_fft3, "FFT");
            struct args_fftwf_fft *thread_param_fft3 = (struct args_fftwf_fft *)malloc(sizeof(struct args_fftwf_fft));
            thread_param_fft3->input_array  = l;
            thread_param_fft3->in           = in_fft;
            thread_param_fft3->out          = out_fft;
            thread_param_fft3->output_array = q;
            thread_param_fft3->n_elements   = m;
            thread_param_fft3->p            = p4;
            assert(pthread_create(&thread_fft3, &attr_thread_fft3, fftwf_fft, (void *)thread_param_fft3) == 0);
            assert(pthread_join(thread_fft3, NULL) == 0);
            free(thread_param_fft3);
            #endif
            
        #endif
    
        #ifdef PRINT_BLOCK_EXECUTION_TIMES 
	    clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	    exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
	    printf("[ INFO] FFT-3 execution time (ns): %f\n", exec_time);
        #endif

        //#######################################################################
	    // Amplitude Computation
	    //#######################################################################
        #ifdef PRINT_BLOCK_EXECUTION_TIMES 
        clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
        #endif

        #ifndef THREAD_PER_TASK
        amplitude(m, q, r);
        #else
        pthread_t thread_amplitude;
        pthread_attr_t attr_thread_amplitude;
        pthread_attr_init(&attr_thread_amplitude);
        pthread_attr_setname(&attr_thread_amplitude, "amplitude");
        struct args_amplitude *thread_param_amplitude = (struct args_amplitude *)malloc(sizeof(struct args_amplitude));
        thread_param_amplitude->m = m;
        thread_param_amplitude->q = q;
        thread_param_amplitude->r = r;
        assert(pthread_create(&thread_amplitude, &attr_thread_amplitude, amplitude, (void *)thread_param_amplitude) == 0);
        assert(pthread_join(thread_amplitude, NULL) == 0);
        free(thread_param_amplitude);
        #endif

        #ifdef PRINT_BLOCK_EXECUTION_TIMES 
	    clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	    exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
	    printf("[ INFO] Amplitude-Comp execution time (ns): %f\n", exec_time);
        #endif

        //#######################################################################
	    // FFT Shift
	    //#######################################################################
        #ifdef PRINT_BLOCK_EXECUTION_TIMES 
        clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
        #endif

        #ifndef THREAD_PER_TASK
        fftshift(r, m);
        #else
        pthread_t thread_shift;
        pthread_attr_t attr_thread_shift;
        pthread_attr_init(&attr_thread_shift);
        pthread_attr_setname(&attr_thread_shift, "shift");
        struct args_fftshift *thread_param_shift = (struct args_fftshift *)malloc(sizeof(struct args_fftshift));
        thread_param_shift->data = r;
        thread_param_shift->count = m;
        assert(pthread_create(&thread_shift, &attr_thread_shift, fftshift, (void *)thread_param_shift) == 0);
        assert(pthread_join(thread_shift, NULL) == 0);
        free(thread_param_shift);
        #endif

        #ifdef PRINT_BLOCK_EXECUTION_TIMES 
	    clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	    exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
	    printf("[ INFO] FFT-Shift execution time (ns): %f\n", exec_time);
        #endif

        for(z = 0; z < m; z++)
        {
            f[x+z*(2*n_samples)] = r[z];      // put the elements of output into corresponding location of the 2D array
            if( r[z] > max )
            {
                max = r[z];
                a = z + 1;
                b = x + 1;
            }
        }
    }

    //fp = fopen("./output_pd_f.txt","w");  // write the output
    //for(i = 0; i < m; i++)
    //{
    //    for(j = 0; j < 2*n_samples; j++)
    //    {
    //        fprintf(fp, "%lf ", f[j+i*(2*n_samples)]);
    //    }
    //    fprintf(fp, "\n");
    //}
    //fclose(fp);

    #ifndef FFT1_HW
        //fftwf_destroy_plan(p4);
        fftwf_free(in_fft);
        fftwf_free(out_fft);
        fftwf_cleanup();
    #endif

    free(mf);
    free(q);
    free(r);
    free(l);

    #ifndef PAPI
    #ifndef PRINT_BLOCK_EXECUTION_TIMES 
	clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	exec_time = (((double)end1.tv_sec*SEC2NANOSEC + (double)end1.tv_nsec)) - (((double)start1.tv_sec*SEC2NANOSEC + (double)start1.tv_nsec));
	printf("\n[ INFO] Pulse-Doppler execution time (ms): %f\n", (total_exec_time + exec_time) / 1000000);
    #endif
    #endif

    //#################################################################
    //## Close FFT hardware calls
    //#################################################################
        
    //#ifdef FFT1_HW
    //    close_dma1();
    //    close_fft1();
    //    munmap(udmabuf1_base_addr, 8192);
    //    close(fd_udmabuf1);
    //#endif

    #ifndef PAPI
    float rg, dp;
    rg = (b-n_samples)/(n_samples-1)*PRI;
    dp = (a-(m+1)/2)/(m-1)/PRI;
    printf("[ INFO] Doppler shift = %lf, time delay = %lf\n\n", dp, rg);
    #endif
}


