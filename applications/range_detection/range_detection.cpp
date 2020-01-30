#include <string>
#include <cmath>
#include "range_detection.hpp"

#define RANGE_DETECT_DEBUG 0
#define SEC2NANOSEC 1000000000

void fftwf_fft(float *input_array, fftwf_complex *in, fftwf_complex *out, float *output_array, size_t n_elements, fftwf_plan p) {
    for(size_t i = 0; i < 2*n_elements; i+=2)
    {
        in[i/2][0] = input_array[i];
        in[i/2][1] = input_array[i+1];
    }
    fftwf_execute(p);
    for(size_t i = 0; i < 2*n_elements; i+=2)
    {
        output_array[i] = (float) out[i/2][0];
        output_array[i+1] = (float) out[i/2][1];
    }
}

void range_detect_fields_cleanup(struct range_detect_fields *range_detect_param){
	free(range_detect_param->time);
	free(range_detect_param->received);
	free(range_detect_param->lfm_waveform);
	free(range_detect_param->X1);
	free(range_detect_param->X2);
	free(range_detect_param->corr_freq);
	free(range_detect_param->corr);
	fftwf_destroy_plan(range_detect_param->p1);
	fftwf_destroy_plan(range_detect_param->p2);
	fftwf_destroy_plan(range_detect_param->p3);
	fftwf_free(range_detect_param->in_xcorr1);
	fftwf_free(range_detect_param->out_xcorr1);
	fftwf_free(range_detect_param->in_xcorr2);
	fftwf_free(range_detect_param->out_xcorr2);
	fftwf_free(range_detect_param->in_xcorr3);
	fftwf_free(range_detect_param->out_xcorr3);
}

void range_detect_nop(task_nodes *task){
    printf("Finished Nop task\n");
}

void range_detect_LFM( task_nodes *task){
	struct range_detect_fields *range_detect_param;
	range_detect_param = ( struct range_detect_fields *)task->fields;

	for (size_t i = 0; i < 2*(range_detect_param->time_n_samples); i+=2){
		range_detect_param->lfm_waveform[i] = creal(cexp(I *  M_PI * (range_detect_param->B/range_detect_param->T) * pow((range_detect_param->time[i/2]),2)));
		range_detect_param->lfm_waveform[i+1] = cimag(cexp(I *  M_PI * (range_detect_param->B/range_detect_param->T) * pow((range_detect_param->time[i/2]),2)));

	}
	for(size_t i =(range_detect_param->time_n_samples);i<2*(range_detect_param->n_samples);i++){
		range_detect_param->lfm_waveform[i] = 0.0;
	}

	if (RANGE_DETECT_DEBUG == 1){
		for (size_t i = 0; i < 2*range_detect_param->n_samples; i+=2){
			 printf("index %lu  real %f   imag %f \n",i/2, range_detect_param->lfm_waveform[i], range_detect_param->lfm_waveform[i+1] );
		}
	}
	//printf("Ending range detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
	//clock_gettime(CLOCK_MONOTONIC, &task->end);
	//task->actual_execution_time = (long long)(task->end.tv_sec*SEC2NANOSEC + task->end.tv_nsec) - (long long)(task->start.tv_sec*SEC2NANOSEC + task->start.tv_nsec);
	printf("Finished LFM task\n");
}

void range_detect_FFT_0(task_nodes *task) {
	struct range_detect_fields *range_detect_param = (struct range_detect_fields *) task->fields;
	
	size_t len;
	len = 2 * (range_detect_param->n_samples);
	float *c = (float*) malloc( 2*len *sizeof(float));
	for(size_t i = 0; i<2*(range_detect_param->n_samples-1); i+=2){
		c[i] = 0;
		c[i+1] = 0;
	}
	memcpy(c+2*(range_detect_param->n_samples), range_detect_param->received, 2*(range_detect_param->n_samples)*sizeof(float));
	c[2*len-2] = 0;
	c[2*len - 1] = 0;
	if (strcmp(task->actual_resource_assign, "cpu") == 0){
	    printf("About to run FFT 0 on CPU\n");
		fftwf_fft(c, range_detect_param->in_xcorr1, range_detect_param->out_xcorr1,  range_detect_param->X1, len, range_detect_param->p1);
    }
	else if (strcmp(task->actual_resource_assign, "fft") == 0){
        printf("About to run FFT 0 on FPGA\n");
		//fftwf_fft(c, range_detect_param->in_xcorr1, range_detect_param->out_xcorr1,  range_detect_param->X1, len, range_detect_param->p1);
		DASH_fft(task->alloc_resource_config_input, c, range_detect_param->X1, len);
		
	}
	else {
        fprintf(stderr, "Assigned resource %s is not supported for running task %s task id %d app id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id);
    }
	
	if (RANGE_DETECT_DEBUG == 1){
		for (int i = 0; i < (2*(range_detect_param->n_samples)*2); i=i+2) {
			printf("X1 index %d real: %lf  imag %lf \n", i/2, range_detect_param->X1[i],range_detect_param->X1[i+1]);
		}
	}
	free(c);
	//printf("Ending range detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
	//clock_gettime(CLOCK_MONOTONIC, &task->end);
	//task->actual_execution_time = (long long)(task->end.tv_sec*SEC2NANOSEC + task->end.tv_nsec) - (long long)(task->start.tv_sec*SEC2NANOSEC + task->start.tv_nsec);
	printf("Finished FFT 0\n");
}

void range_detect_FFT_1(task_nodes *task) {
	//clock_gettime(CLOCK_MONOTONIC, &task->start);
	//pthread_t self;
	//self = pthread_self();
	//printf("Starting range detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
	struct range_detect_fields *range_detect_param;
	range_detect_param = ( struct range_detect_fields *)task->fields;
	
	size_t len;
	len = 2 * (range_detect_param->n_samples);
	float *d = (float*) malloc( 2*len *sizeof(float));
	size_t x_count = 0;
	size_t y_count = 0;
	float *z = (float*) malloc( 2*(range_detect_param->n_samples) *sizeof(float));
	for(size_t i = 0; i<2*(range_detect_param->n_samples); i+=2){
		z[i] = 0;
		z[i+1] = 0;
	}
	memcpy(d, range_detect_param->lfm_waveform, 2*(range_detect_param->n_samples)*sizeof(float));
	memcpy(d+2*(range_detect_param->n_samples), z, 2*(range_detect_param->n_samples)*sizeof(float));
	if (strcmp(task->actual_resource_assign, "cpu") == 0){
        printf("About to run FFT 1 on CPU\n");
		fftwf_fft(d, range_detect_param->in_xcorr2, range_detect_param->out_xcorr2, range_detect_param->X2, len,range_detect_param->p2);	
	}
	else if (strcmp(task->actual_resource_assign, "fft") == 0){
        printf("About to run FFT 1 on FPGA\n");
		//fftwf_fft(d, range_detect_param->in_xcorr2, range_detect_param->out_xcorr2, range_detect_param->X2, len,range_detect_param->p2);	
		DASH_fft(task->alloc_resource_config_input, d, range_detect_param->X2, len);
	}
	else {printf("Assigned resource %s is not supported for running task %s  task id %d app id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id);}
	if (RANGE_DETECT_DEBUG == 1){
		for (int i = 0; i <  (2*(range_detect_param->n_samples)*2); i=i+2) {
			printf("X2 index %d real: %lf  imag %lf \n", i/2, range_detect_param->X2[i],range_detect_param->X2[i+1]);
		}
	}
	
	free(d);
	free(z);
	//printf("Ending range detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
	//clock_gettime(CLOCK_MONOTONIC, &task->end);
	//task->actual_execution_time = (long long)(task->end.tv_sec*SEC2NANOSEC + task->end.tv_nsec) - (long long)(task->start.tv_sec*SEC2NANOSEC + task->start.tv_nsec);
	printf("Finished FFT 1\n");
}

void range_detect_MUL(task_nodes *task) {
	//clock_gettime(CLOCK_MONOTONIC, &task->start);
	//pthread_t self;
	//self = pthread_self();
	//printf("Starting range detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
	struct range_detect_fields *range_detect_param;
	range_detect_param = ( struct range_detect_fields *)task->fields;
	size_t len;
	len = 2 * (range_detect_param->n_samples);
	for(size_t i =0;i<2*len;i+=2){
		range_detect_param->corr_freq[i] = (range_detect_param->X1[i] * range_detect_param->X2[i]) + (range_detect_param->X1[i+1] * range_detect_param->X2[i+1]);
		range_detect_param->corr_freq[i+1] = (range_detect_param->X1[i+1] * range_detect_param->X2[i]) - (range_detect_param->X1[i] * range_detect_param->X2[i+1]);
	}	

	if (RANGE_DETECT_DEBUG == 1){
		for (int i = 0; i <  (2*len); i=i+2) {
			printf("corr_freq index %d real: %lf  imag %lf \n", i/2, range_detect_param->corr_freq[i],range_detect_param->corr_freq[i+1]);
		}
	}	
	//printf("Ending range detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
	//clock_gettime(CLOCK_MONOTONIC, &task->end);
	//task->actual_execution_time = (long long)(task->end.tv_sec*SEC2NANOSEC + task->end.tv_nsec) - (long long)(task->start.tv_sec*SEC2NANOSEC + task->start.tv_nsec);
    printf("Finished MUL\n");
}

void range_detect_IFFT(task_nodes *task) {
	//clock_gettime(CLOCK_MONOTONIC, &task->start);
	//pthread_t self;
	//self = pthread_self();
	//printf("Starting range detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
	struct range_detect_fields *range_detect_param;
	range_detect_param = ( struct range_detect_fields *)task->fields;
	size_t len;
	len = 2 * (range_detect_param->n_samples);


	if (strcmp(task->actual_resource_assign, "cpu") == 0){
		printf("About to run IFFT on CPU\n");
	    fftwf_fft(range_detect_param->corr_freq, range_detect_param->in_xcorr3, range_detect_param->out_xcorr3, range_detect_param->corr, len, range_detect_param->p3);
	}
	else if (strcmp(task->actual_resource_assign, "fft") == 0){
		printf("About to run IFFT on FPGA\n");
		//fftwf_fft(range_detect_param->corr_freq, range_detect_param->in_xcorr3, range_detect_param->out_xcorr3, range_detect_param->corr, len, range_detect_param->p3);
		DASH_ifft(task->alloc_resource_config_input, range_detect_param->corr_freq, range_detect_param->corr, len);
	}
	else {printf("Assigned resource %s is not supported for running task %s  task id %d app id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id);}
	if (RANGE_DETECT_DEBUG == 1){
		for (int i = 0; i <  (2*len); i=i+2) {
			printf("corr index %d real: %lf  imag %lf \n", i/2, range_detect_param->corr[i],range_detect_param->corr[i+1]);
			
		}
	}
	//printf("Ending range detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
	//clock_gettime(CLOCK_MONOTONIC, &task->end);
	//task->actual_execution_time = (long long)(task->end.tv_sec*SEC2NANOSEC + task->end.tv_nsec) - (long long)(task->start.tv_sec*SEC2NANOSEC + task->start.tv_nsec);
    printf("Finished IFFT\n");
}

void range_detect_MAX(task_nodes *task) {
	//clock_gettime(CLOCK_MONOTONIC, &task->start);
	//pthread_t self;
	//self = pthread_self();
	//printf("Starting range detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
	struct range_detect_fields *range_detect_param;
	range_detect_param = ( struct range_detect_fields *)task->fields;
	
	range_detect_param->index =0;
	

	for(size_t i =0;i<2*(2*(range_detect_param->n_samples));i+=2){
		if (range_detect_param->corr[i] > range_detect_param->max_corr){
			range_detect_param->max_corr = range_detect_param->corr[i];
			range_detect_param->index = i/2;
		}
	}

	range_detect_param->lag = (range_detect_param->index - range_detect_param->n_samples)/range_detect_param->sampling_rate;
	for (int i = 0; i <  (2*(range_detect_param->n_samples)*2); i=i+2) {
		//printf("X1 index %d real: %lf  imag %lf \n", i/2, range_detect_param->X1[i],range_detect_param->X1[i+1]);
	}
	for (int i = 0; i <  (2*(range_detect_param->n_samples)*2); i=i+2) {
		//printf("X2 index %d real: %lf  imag %lf \n", i/2, range_detect_param->X2[i],range_detect_param->X2[i+1]);
	}
	//clock_gettime(CLOCK_MONOTONIC, &task->end);
	//task->actual_execution_time = (long long)(task->end.tv_sec*SEC2NANOSEC + task->end.tv_nsec) - (long long)(task->start.tv_sec*SEC2NANOSEC + task->start.tv_nsec);
	printf("LAG value is %f \n", range_detect_param->lag);
	printf("MAX index  %f and max value %f \n", range_detect_param->index, range_detect_param->max_corr);
	//printf("Ending range detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
    printf("Finished MAX\n");
}

void range_detection_init(dag_app *range_detect) {
	//512 3 128/500000 500000 10000	
	// //256 1 256/500000 500000 1000

    struct range_detect_fields* range_detect_param = (struct range_detect_fields*) range_detect->app_fields;

	range_detect_param->n_samples=256;
	range_detect_param->time_n_samples =1;
	range_detect_param->T = (float) (256.0/500000);
	range_detect_param->B = (float) 500000;
	range_detect_param->sampling_rate = 1000;
	range_detect_param->time =  (double*) malloc((range_detect_param->n_samples)*sizeof(double));
	range_detect_param->received = (float*) malloc(2*(range_detect_param->n_samples)*sizeof(float));
	range_detect_param->lfm_waveform = (float*) malloc(2*(range_detect_param->n_samples)*sizeof(float));
	range_detect_param->in_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(range_detect_param->n_samples)));
	range_detect_param->out_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(range_detect_param->n_samples)));
	range_detect_param->in_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(range_detect_param->n_samples)));
	range_detect_param->out_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(range_detect_param->n_samples)));
	range_detect_param->in_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(range_detect_param->n_samples)));
	range_detect_param->out_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(range_detect_param->n_samples)));
	range_detect_param->p1 = fftwf_plan_dft_1d((2*(range_detect_param->n_samples)), range_detect_param->in_xcorr1, range_detect_param->out_xcorr1, FFTW_FORWARD, FFTW_ESTIMATE);
	range_detect_param->p2 = fftwf_plan_dft_1d((2*(range_detect_param->n_samples)), range_detect_param->in_xcorr2, range_detect_param->out_xcorr2, FFTW_FORWARD, FFTW_ESTIMATE);
	range_detect_param->p3 = fftwf_plan_dft_1d((2*(range_detect_param->n_samples)), range_detect_param->in_xcorr3, range_detect_param->out_xcorr3, FFTW_BACKWARD, FFTW_ESTIMATE);
	range_detect_param->X1 = (float*) malloc(2*(2*(range_detect_param->n_samples)) *sizeof(float));
	range_detect_param->X2 = (float*) malloc(2*(2*(range_detect_param->n_samples)) *sizeof(float));
	range_detect_param->corr_freq = (float*) malloc(2*(2*(range_detect_param->n_samples)) *sizeof(float));
	range_detect_param->corr = (float*) malloc(2*(2*(range_detect_param->n_samples)) *sizeof(float));
	
	FILE *fp;
	std::string timeInput = std::string(INPUT_DIR) + std::string("time_input.txt");
	fp = fopen(timeInput.c_str(),"r");
	for (size_t i = 0; i < range_detect_param->time_n_samples; i++)
	{
        fscanf(fp,"%lf", &(range_detect_param->time[i]));
	}
	fclose(fp);

	std::string receivedInput = std::string(INPUT_DIR) + std::string("received_input.txt");
	fp = fopen(receivedInput.c_str(),"r");
	for(size_t i=0; i<2*(range_detect_param->n_samples); i++)
	{
        fscanf(fp,"%f", &(range_detect_param->received[i]));
	}
	fclose(fp);
	printf("Finished range_detection_init\n");
}

int main(void) {}