#include <cmath>
#include <iomanip>
#include <iostream>
#ifdef INCLUDE_MAIN
#include <fstream>
#include <nlohmann/json.hpp>
#endif
#include "pulse_doppler.hpp"

#define PD_DEBUG 0
#define M 128
#define N 64

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

void pulse_doppler_fields_cleanup(struct pulse_doppler_fields *pulse_doppler_param) {
    free(pulse_doppler_param->mf);
    free(pulse_doppler_param->p);
    free(pulse_doppler_param->pulse);
    free(pulse_doppler_param->max);
    free(pulse_doppler_param->a);
    free(pulse_doppler_param->b);
    free(pulse_doppler_param->q);
    free(pulse_doppler_param->f);
    free(pulse_doppler_param->r);
    free(pulse_doppler_param->X1);
    free(pulse_doppler_param->X2);
    free(pulse_doppler_param->corr_freq);
    free(pulse_doppler_param->corr);
    for (int i =0; i <pulse_doppler_param->m;i++){
        fftwf_destroy_plan((pulse_doppler_param->p1[i]));
        fftwf_destroy_plan((pulse_doppler_param->p2[i]));
        fftwf_destroy_plan((pulse_doppler_param->p3[i]));
    }
    for (int i =0; i <(pulse_doppler_param->n_samples)*2;i++){
        fftwf_destroy_plan((pulse_doppler_param->p4[i]));

    }
    free(pulse_doppler_param->p1);
    free(pulse_doppler_param->p2);
    free(pulse_doppler_param->p3);
    free(pulse_doppler_param->p4);
    //fftwf_destroy_plan(pulse_doppler_param->p2);
    //fftwf_destroy_plan(pulse_doppler_param->p3);
    //fftwf_destroy_plan(pulse_doppler_param->p4);
    fftwf_free(pulse_doppler_param->in_xcorr1);
    fftwf_free(pulse_doppler_param->out_xcorr1);
    fftwf_free(pulse_doppler_param->in_xcorr2);
    fftwf_free(pulse_doppler_param->out_xcorr2);
    fftwf_free(pulse_doppler_param->in_xcorr3);
    fftwf_free(pulse_doppler_param->out_xcorr3);
    fftwf_free(pulse_doppler_param->in_fft);
    fftwf_free(pulse_doppler_param->out_fft);
    //fftwf_cleanup();
}

void pulse_doppler_nop(task_nodes *task) {
    printf("Finished pulse_doppler_nop, task id %d\n", task->task_id);
}

void pulse_doppler_FFT_0(task_nodes *task) {
    struct pulse_doppler_fields *pulse_doppler_param;
    pulse_doppler_param = ( struct pulse_doppler_fields *)task->fields;

    size_t len;
    len = 2 * (pulse_doppler_param->n_samples);

    float *p;
    float *X1;
    p = &((pulse_doppler_param->p[ 2*pulse_doppler_param->n_samples * ((task->task_id)/2)]));
    X1 = &((pulse_doppler_param->X1[2*(2*(pulse_doppler_param->n_samples)) * ((task->task_id)/2)]));

    float *c = (float*) malloc( 2*len *sizeof(float));
    for(size_t i = 0; i<2*(pulse_doppler_param->n_samples-1); i+=2){
        c[i] = 0;
        c[i+1] = 0;
    }
    memcpy(c+2*(pulse_doppler_param->n_samples - 1), p, 2*(pulse_doppler_param->n_samples)*sizeof(float));
    c[2*len-2] = 0;
    c[2*len - 1] = 0;
    if (strcmp(task->actual_resource_assign, "cpu") == 0){
        fftwf_fft(c, &(pulse_doppler_param->in_xcorr1[(task->task_id/2)*2*(pulse_doppler_param->n_samples)]), &(pulse_doppler_param->out_xcorr1[(task->task_id/2)*2*(pulse_doppler_param->n_samples)]),  X1, len, pulse_doppler_param->p1[task->task_id/2]);
    }
    else if (strcmp(task->actual_resource_assign, "fft") == 0){
        //printf("On FPGA FFT %s running task %s  task id %d app id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id);
        //printf("On FPGA FFT %s running task %s  task id %d app id %d FFT id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id, task->alloc_resource_config_input);
        DASH_fft(task->alloc_resource_config_input, c, X1, len);
        //fftwf_fft(c, &(pulse_doppler_param->in_xcorr1[(task->task_id/2)*2*(pulse_doppler_param->n_samples)]), &(pulse_doppler_param->out_xcorr1[(task->task_id/2)*2*(pulse_doppler_param->n_samples)]),  X1, len, pulse_doppler_param->p1[task->task_id/2]);
    }
    else {printf("Assigned resource %s is not supported for running task %s  task id %d app id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id);}

    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*(pulse_doppler_param->n_samples)*2); i=i+2) {
            printf("IT %d X1 index %d real: %lf  imag %lf \n",(task->task_id)/2 , i/2, pulse_doppler_param->X1[i+(2*(2*(pulse_doppler_param->n_samples)) * ((task->task_id)/2))],pulse_doppler_param->X1[i+1+ (2*(2*(pulse_doppler_param->n_samples)) * ((task->task_id)/2))]);
        }
    }
    free(c);
    printf("Finished pulse_doppler_FFT_0, task id: %d\n", task->task_id);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
}

void pulse_doppler_FFT_1(task_nodes *task) {
    struct pulse_doppler_fields *pulse_doppler_param;
    pulse_doppler_param = ( struct pulse_doppler_fields *)task->fields;

    size_t len;
    len = 2 * (pulse_doppler_param->n_samples);
    float *d = (float*) malloc( 2*len *sizeof(float));
    float *z = (float*) malloc( 2*(pulse_doppler_param->n_samples) *sizeof(float));
    float *X2;
    X2 = &(pulse_doppler_param->X2[2*(2*(pulse_doppler_param->n_samples)) * (((task->task_id)/2) - 1)]);
    float *y;
    y = pulse_doppler_param->pulse;

    size_t x_count = 0;
    size_t y_count = 0;
    for(size_t i = 0; i<2*(pulse_doppler_param->n_samples); i+=2){
        z[i] = 0;
        z[i+1] = 0;
    }
    memcpy(d, y, 2*(pulse_doppler_param->n_samples)*sizeof(float));
    memcpy(d+2*(pulse_doppler_param->n_samples), z, 2*(pulse_doppler_param->n_samples)*sizeof(float));


    if (strcmp(task->actual_resource_assign, "cpu") == 0){
        fftwf_fft(d, &(pulse_doppler_param->in_xcorr2[((task->task_id/2)-1)*2*(pulse_doppler_param->n_samples)]), &(pulse_doppler_param->out_xcorr2[((task->task_id/2)-1)*2*(pulse_doppler_param->n_samples)]), X2, len,pulse_doppler_param->p2[(task->task_id/2)-1]);
    }
    else if (strcmp(task->actual_resource_assign, "fft") == 0){
        //printf("On FPGA FFT %s running task %s  task id %d app id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id);
        //printf("On FPGA FFT %s running task %s  task id %d app id %d FFT id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id, task->alloc_resource_config_input);
        DASH_fft(task->alloc_resource_config_input, d, X2, len);
        //fftwf_fft(d, &(pulse_doppler_param->in_xcorr2[((task->task_id/2)-1)*2*(pulse_doppler_param->n_samples)]), &(pulse_doppler_param->out_xcorr2[((task->task_id/2)-1)*2*(pulse_doppler_param->n_samples)]), X2, len,pulse_doppler_param->p2[(task->task_id/2)-1]);
    }
    else {printf("Assigned resource %s is not supported for running task %s  task id %d app id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id);}
    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*(pulse_doppler_param->n_samples)*2); i=i+2) {
            //printf("X2 index %d real: %lf  imag %lf \n", i/2, X2[i],X2[i+1]);
            printf("IT %d X2 index %d real: %lf  imag %lf \n",(task->task_id/2)-1 , i/2, pulse_doppler_param->X2[i+(2*(2*(pulse_doppler_param->n_samples)) * ((task->task_id/2)-1))],pulse_doppler_param->X2[i+1+ (2*(2*(pulse_doppler_param->n_samples)) * ((task->task_id/2) - 1))]);
        }
    }

    free(d);
    free(z);
    printf("Finished pulse_doppler_FFT_1, task id: %d\n", task->task_id);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
}

void pulse_doppler_MUL(task_nodes *task) {
    struct pulse_doppler_fields *pulse_doppler_param;
    pulse_doppler_param = ( struct pulse_doppler_fields *)task->fields;
    size_t len;
    len = 2 * (pulse_doppler_param->n_samples);
    float *X1;
    float *X2;
    float *corr_freq;
    int index = 2*pulse_doppler_param->m + 1;

    X1 = &((pulse_doppler_param->X1[2*(2*(pulse_doppler_param->n_samples)) * (task->task_id-index)]));
    X2 = &(pulse_doppler_param->X2[2*(2*(pulse_doppler_param->n_samples)) * (task->task_id-index)]);
    corr_freq =  &(pulse_doppler_param->corr_freq[2*(2*(pulse_doppler_param->n_samples)) * (task->task_id-index)]);

    for(size_t i =0;i<2*len;i+=2){
        corr_freq[i] =   (X1[i] * X2[i]) +   (X1[i+1] * X2[i+1]);
        corr_freq[i+1] = (X1[i+1] * X2[i]) - (X1[i] * X2[i+1]);
    }

    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*len); i=i+2) {
            //printf("corr_freq index %d real: %lf  imag %lf \n", i/2, pulse_doppler_param->corr_freq[i],pulse_doppler_param->corr_freq[i+1]);
            printf("IT %d corr_freq index %d real: %lf  imag %lf \n",(task->task_id-index) , i/2, pulse_doppler_param->corr_freq[i+(2*(2*(pulse_doppler_param->n_samples)) * (task->task_id-index))],pulse_doppler_param->corr_freq[i+1+ (2*(2*(pulse_doppler_param->n_samples)) * (task->task_id - index))]);
        }
    }
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
    printf("Finished pulse_doppler_MUL, task id: %d\n", task->task_id);
}

void pulse_doppler_IFFT(task_nodes *task) {
    struct pulse_doppler_fields *pulse_doppler_param;
    pulse_doppler_param = ( struct pulse_doppler_fields *)task->fields;
    size_t len;
    len = 2 * (pulse_doppler_param->n_samples);

    float *corr_freq;
    float *corr;
    int index = 3*pulse_doppler_param->m + 1;
    corr= &((pulse_doppler_param->corr[2*(2*(pulse_doppler_param->n_samples)) * (task->task_id-index)]));
    corr_freq= &((pulse_doppler_param->corr_freq[2*(2*(pulse_doppler_param->n_samples)) * (task->task_id-index)]));

    if (strcmp(task->actual_resource_assign, "cpu") == 0){
        fftwf_fft(corr_freq, &(pulse_doppler_param->in_xcorr3[(task->task_id-index)*2*(pulse_doppler_param->n_samples)]), &(pulse_doppler_param->out_xcorr3[(task->task_id-index)*2*(pulse_doppler_param->n_samples)]), corr, len, pulse_doppler_param->p3[task->task_id-index]);
    }
    else if (strcmp(task->actual_resource_assign, "fft") == 0){
        //printf("On FPGA FFT %s running task %s  task id %d app id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id);
        //printf("On FPGA IFFT %s running task %s  task id %d app id %d FFT id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id,task->alloc_resource_config_input );
        DASH_ifft(task->alloc_resource_config_input, corr_freq, corr, len);
        //fftwf_fft(corr_freq, &(pulse_doppler_param->in_xcorr3[(task->task_id-index)*2*(pulse_doppler_param->n_samples)]), &(pulse_doppler_param->out_xcorr3[(task->task_id-index)*2*(pulse_doppler_param->n_samples)]), corr, len, pulse_doppler_param->p3[task->task_id-index]);
    }
    else {printf("Assigned resource %s is not supported for running task %s  task id %d app id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id);}
    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*len); i=i+2) {
            //printf("corr index %d real: %lf  imag %lf \n", i/2, pulse_doppler_param->corr[i],pulse_doppler_param->corr[i+1]);
            printf("IT %d corr index %d real: %lf  imag %lf \n",(task->task_id-index) , i/2, pulse_doppler_param->corr[i+(2*(2*(pulse_doppler_param->n_samples)) * (task->task_id-index))],pulse_doppler_param->corr[i+1+ (2*(2*(pulse_doppler_param->n_samples)) * (task->task_id - index))]);

        }
    }
    printf("Finished pulse_doppler_IFFT, task id: %d\n", task->task_id);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
}

void pulse_doppler_REALIGN_MAT(task_nodes *task) {
    struct pulse_doppler_fields *pulse_doppler_param;
    pulse_doppler_param = ( struct pulse_doppler_fields *)task->fields;
    size_t len;
    len = 2 * (pulse_doppler_param->n_samples);
    float *mf;
    float *corr;
    mf = pulse_doppler_param->mf;
    corr = pulse_doppler_param->corr;
    for(int k = 0; k < pulse_doppler_param->m; k++){
        for(int j = 0; j < 2*(2 * pulse_doppler_param->n_samples); j+=2){
            mf[j/2+(2*k)*(2*pulse_doppler_param->n_samples)] = corr[k*(2*(2 * pulse_doppler_param->n_samples)) + j];
            mf[j/2+(2*k+1)*(2*pulse_doppler_param->n_samples)]= corr[k*(2*(2 * pulse_doppler_param->n_samples)) + j+1];

        }

    }
    if (PD_DEBUG == 1){
        for(int k = 0; k < pulse_doppler_param->m; k++){
            for(int j = 0; j < 2*(2 * pulse_doppler_param->n_samples); j+=2){
                //printf("mf %f %f\n",mf[k*2*(2 * pulse_doppler_param->n_samples) + j], mf[k*2*(2 * pulse_doppler_param->n_samples) + j+1]);
                //printf("mf %f %f\n",mf[j/2+(2*k)*(2*pulse_doppler_param->n_samples)], mf[j/2+(2*k+1)*(2*pulse_doppler_param->n_samples)]);
                printf("mf %f %f\n",(pulse_doppler_param->mf)[j/2+(2*k)*(2*pulse_doppler_param->n_samples)], (pulse_doppler_param->mf)[j/2+(2*k+1)*(2*pulse_doppler_param->n_samples)]);

            }
        }
    }
    printf("Finished pulse_doppler_REALIGN_MAT, task id: %d\n", task->task_id);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
}

void pulse_doppler_FFT_2(task_nodes *task) {

    struct pulse_doppler_fields *pulse_doppler_param;
    pulse_doppler_param = ( struct pulse_doppler_fields *)task->fields;

    float *l = (float*) malloc(2*(pulse_doppler_param->m)*sizeof(float));
    float *q;
    int index = 4*pulse_doppler_param->m + 2;
    int x = task->task_id - index;
    q = &((pulse_doppler_param->q)[ x*2*pulse_doppler_param->m]);
    for(int o = 0; o < 2*pulse_doppler_param->m; o++){
        l[o] = pulse_doppler_param->mf[x+o*(2*pulse_doppler_param->n_samples)];

    }

    if (strcmp(task->actual_resource_assign, "cpu") == 0){
        fftwf_fft(l, &(pulse_doppler_param->in_fft[(task->task_id - index)*pulse_doppler_param->m]), &(pulse_doppler_param->out_fft[(task->task_id - index)*pulse_doppler_param->m]), q, pulse_doppler_param->m, pulse_doppler_param->p4[task->task_id - index]);
    }
    else if (strcmp(task->actual_resource_assign, "fft") == 0){
        //printf("On FPGA FFT %s running task %s  task id %d app id %d FFT id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id, task->alloc_resource_config_input);
        //printf("On FPGA FFT %s running task %s  task id %d app id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id);
        DASH_fft(task->alloc_resource_config_input, l, q, pulse_doppler_param->m);
        //fftwf_fft(l, &(pulse_doppler_param->in_fft[(task->task_id - index)*pulse_doppler_param->m]), &(pulse_doppler_param->out_fft[(task->task_id - index)*pulse_doppler_param->m]), q, pulse_doppler_param->m, pulse_doppler_param->p4[task->task_id - index]);
    }
    else {printf("Assigned resource %s is not supported for running task %s  task id %d app id %d\n",task->actual_resource_assign, task->task_name,task->task_id,task->app_id);}

    if (PD_DEBUG == 1){
        for (int i = 0; i <  (2*(pulse_doppler_param->m)); i=i+2) {
            printf("IT %d FFT2 index %d real: %lf  imag %lf \n",x , i/2, pulse_doppler_param->q[i+ (x*2*pulse_doppler_param->m) ],pulse_doppler_param->q[i+1+ (x*2*pulse_doppler_param->m)]);
        }
    }

    free(l);
    printf("Finished pulse_doppler_FFT_2, task id: %d\n", task->task_id);
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);

}

void pulse_doppler_AMPLITUDE(task_nodes *task) {

    struct pulse_doppler_fields *pulse_doppler_param;
    pulse_doppler_param = ( struct pulse_doppler_fields *)task->fields;

    int index = (2*pulse_doppler_param->n_samples)  + (4*pulse_doppler_param->m) + 2;
    int x = task->task_id - index;

    float *q;
    float *r;
    q = &((pulse_doppler_param->q)[ x*2*pulse_doppler_param->m]);
    r = &((pulse_doppler_param->r)[ x*pulse_doppler_param->m]);
    for(int y = 0; y < 2*pulse_doppler_param->m; y+=2){
        r[y/2] = sqrt(q[y]*q[y]+q[y+1]*q[y+1]);

    }

    if (PD_DEBUG == 1){
        for (int i = 0; i <  ((pulse_doppler_param->m)); i=i+1) {
            printf("IT %d AMP index %d real: %lf \n",x , i, pulse_doppler_param->r[i+ (x*pulse_doppler_param->m) ]);
        }
    }
    printf("Finished pulse_doppler_AMPLITUDE\n");
    //printf("Ending pulse doppler app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
}

void swap(float *v1, float *v2)
{
    float tmp = *v1;
    *v1 = *v2;
    *v2 = tmp;
}

void fftshift(float *data, float count)
{
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

void pulse_doppler_FFTSHIFT(task_nodes *task) {

    struct pulse_doppler_fields *pulse_doppler_param;
    pulse_doppler_param = ( struct pulse_doppler_fields *)task->fields;

    int index = (2*2*pulse_doppler_param->n_samples)  + (4*pulse_doppler_param->m) + 2;
    int x = task->task_id - index;

    float *r;
    r = &((pulse_doppler_param->r)[ x*pulse_doppler_param->m]);


    fftshift(r, pulse_doppler_param->m);

    if (PD_DEBUG == 1){
        for (int i = 0; i <  ((pulse_doppler_param->m)); i=i+1) {
            printf("IT %d SHIFT index %d real: %lf \n",x , i, pulse_doppler_param->r[i+ (x*pulse_doppler_param->m) ]);
        }
    }
    printf("Finished pulse_doppler_FFTSHIFT\n");
}

void pulse_doppler_MAX0(task_nodes *task) {
    struct pulse_doppler_fields *pulse_doppler_param;
    pulse_doppler_param = ( struct pulse_doppler_fields *)task->fields;

    int index = (3*2*pulse_doppler_param->n_samples)  + (4*pulse_doppler_param->m) + 2;
    int x = task->task_id - index;

    float *r;
    r = &((pulse_doppler_param->r)[ x*pulse_doppler_param->m]);
    float *max, *a, *b;
    max = &((pulse_doppler_param->max)[x]);
    a = &((pulse_doppler_param->a)[x]);
    b = &((pulse_doppler_param->b)[x]);

    for(int z = 0; z < pulse_doppler_param->m; z++){
        pulse_doppler_param->f[x+z*(2*pulse_doppler_param->n_samples)] = r[z];
        if( r[z] > max[0] ){
            max[0] = r[z];
            a[0] = z + 1;
            b[0] = x + 1;

        }
    }

    if (PD_DEBUG == 1){
        printf("IT %d MAX0 real: %lf %lf %lf\n",x ,  pulse_doppler_param->max[x], pulse_doppler_param->a[x], pulse_doppler_param->b[x]);
    }
    printf("Finished pulse_doppler_MAX0\n");
}

void pulse_doppler_FINAL_MAX(task_nodes *task) {
    struct pulse_doppler_fields *pulse_doppler_param;
    pulse_doppler_param = ( struct pulse_doppler_fields *)task->fields;

    int index = (4*2*pulse_doppler_param->n_samples)  + (4*pulse_doppler_param->m) + 2;
    int x = task->task_id - index;

    float max, a, b;
    max = 0; a= 0; b=0;

    for(int z = 0; z < (2*pulse_doppler_param->n_samples); z++){
        if(  pulse_doppler_param->max[z] > max){
            max = pulse_doppler_param->max[z];
            a = pulse_doppler_param->a[z];
            b = pulse_doppler_param->b[z];

        }
    }

    if (PD_DEBUG == 0){
        float rg, dp;
        rg = (b-pulse_doppler_param->n_samples)/(pulse_doppler_param->n_samples-1)*pulse_doppler_param->PRI;
        dp = (a-(pulse_doppler_param->m+1)/2)/(pulse_doppler_param->m-1)/pulse_doppler_param->PRI;
        for (int j = 0; j<(pulse_doppler_param->m);j++){
            for(int k = 0; k < 2 * pulse_doppler_param->n_samples; k++){
                //printf("recive IT %d index %d value %f\n" ,j,k, pulse_doppler_param->p[k + j*2*(pulse_doppler_param->n_samples)]);
            }
            for (int i = 0; i <  (2*(pulse_doppler_param->n_samples)*2); i=i+2) {
                //printf("IT %d X1 index %d real: %lf  imag %lf \n",j , i/2, pulse_doppler_param->X1[i+(2*(2*(pulse_doppler_param->n_samples)) * j)],pulse_doppler_param->X1[i+1+ (2*(2*(pulse_doppler_param->n_samples)) * j)]);
            }
            for (int i = 0; i <  (2*(pulse_doppler_param->n_samples)*2); i=i+2) {
                //printf("IT %d X2 index %d real: %lf  imag %lf \n",j , i/2, pulse_doppler_param->X2[i+(2*(2*(pulse_doppler_param->n_samples)) * j)],pulse_doppler_param->X2[i+1+ (2*(2*(pulse_doppler_param->n_samples)) * j)]);
            }

        }
        printf("doppler shift = %lf, time delay = %lf\n", dp, rg);
    }
    printf("Finished pulse_doppler_FINAL_MAX\n");
}

void pulse_doppler_init(dag_app *pulse_doppler) {
    struct pulse_doppler_fields *pulse_doppler_param = (struct pulse_doppler_fields*) pulse_doppler->app_fields;

    pulse_doppler_param->m = M;
    pulse_doppler_param->n_samples = N;
    pulse_doppler_param->PRI = 6.3e-5;
    pulse_doppler_param->mf = (float*) malloc((2*N)*M*2*sizeof(float));
    pulse_doppler_param->pulse = (float*) malloc(2*N *sizeof(float));

    pulse_doppler_param->p = (float*) malloc(2*N*M *sizeof(float));
    pulse_doppler_param->X1 = (float*) malloc(2*(2*(pulse_doppler_param->n_samples))*M *sizeof(float));
    pulse_doppler_param->X2 = (float*) malloc(2*(2*(pulse_doppler_param->n_samples))*M *sizeof(float));
    pulse_doppler_param->corr_freq = (float*) malloc(2*(2*(pulse_doppler_param->n_samples))*M *sizeof(float));
    pulse_doppler_param->corr = (float*) malloc(2*(2*N)*M *sizeof(float));

    pulse_doppler_param->in_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * 2*(pulse_doppler_param->n_samples)* (pulse_doppler_param->m));
    pulse_doppler_param->out_xcorr1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(pulse_doppler_param->n_samples))* (pulse_doppler_param->m));
    pulse_doppler_param->in_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(pulse_doppler_param->n_samples))* (pulse_doppler_param->m));
    pulse_doppler_param->out_xcorr2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(pulse_doppler_param->n_samples))* (pulse_doppler_param->m));
    pulse_doppler_param->in_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(pulse_doppler_param->n_samples))* (pulse_doppler_param->m));
    pulse_doppler_param->out_xcorr3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (2*(pulse_doppler_param->n_samples))* (pulse_doppler_param->m));
    pulse_doppler_param->p1 = (fftwf_plan*) malloc(pulse_doppler_param->m * sizeof(fftwf_plan));
    pulse_doppler_param->p2 = (fftwf_plan*) malloc(pulse_doppler_param->m * sizeof(fftwf_plan));
    pulse_doppler_param->p3 = (fftwf_plan*) malloc(pulse_doppler_param->m * sizeof(fftwf_plan));
    for (int i =0; i < pulse_doppler_param->m; i++){
        pulse_doppler_param->p1[i] = fftwf_plan_dft_1d((2*(pulse_doppler_param->n_samples)), &(pulse_doppler_param->in_xcorr1[i * (2*(pulse_doppler_param->n_samples))]), &(pulse_doppler_param->out_xcorr1[i * (2*(pulse_doppler_param->n_samples))]), FFTW_FORWARD, FFTW_ESTIMATE);
    }
    for (int i =0; i <pulse_doppler_param->m;i++){
        pulse_doppler_param->p2[i] = fftwf_plan_dft_1d((2*(pulse_doppler_param->n_samples)), &(pulse_doppler_param->in_xcorr2[i * (2*(pulse_doppler_param->n_samples))]), &(pulse_doppler_param->out_xcorr2[i * (2*(pulse_doppler_param->n_samples))]), FFTW_FORWARD, FFTW_ESTIMATE);
    }
    for (int i =0; i <pulse_doppler_param->m;i++){
        pulse_doppler_param->p3[i] = fftwf_plan_dft_1d((2*(pulse_doppler_param->n_samples)), &(pulse_doppler_param->in_xcorr3[i * (2*(pulse_doppler_param->n_samples))]), &(pulse_doppler_param->out_xcorr3[i * (2*(pulse_doppler_param->n_samples))]), FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    pulse_doppler_param->q = (float*) malloc(2*pulse_doppler_param->m*sizeof(float) * 2*(pulse_doppler_param->n_samples));
    pulse_doppler_param->r = (float*) malloc(pulse_doppler_param->m*sizeof(float) * 2*(pulse_doppler_param->n_samples));
    //*l = malloc(2*m*sizeof(float));
    pulse_doppler_param->f = (float*) malloc(pulse_doppler_param->m*(2*pulse_doppler_param->n_samples)*sizeof(float));
    pulse_doppler_param->max = (float*) malloc((2*pulse_doppler_param->n_samples)*sizeof(float));
    for (int i=0 ; i < 2*pulse_doppler_param->n_samples; i= i+ 1){
        pulse_doppler_param->max[i] = 0;
    }
    pulse_doppler_param->a = (float*) malloc((2*pulse_doppler_param->n_samples)*sizeof(float));
    pulse_doppler_param->b = (float*) malloc((2*pulse_doppler_param->n_samples)*sizeof(float));
    //pulse_doppler_param->max = 0;
    pulse_doppler_param->in_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * pulse_doppler_param->m * (2*(pulse_doppler_param->n_samples)));
    pulse_doppler_param->out_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * pulse_doppler_param->m * (2*(pulse_doppler_param->n_samples)));
    //pulse_doppler_param->p4 = fftwf_plan_dft_1d(pulse_doppler_param->m, pulse_doppler_param->in_fft, pulse_doppler_param->out_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    pulse_doppler_param->p4 = (fftwf_plan*) malloc((pulse_doppler_param->n_samples)*2 * sizeof(fftwf_plan));
    for (int i =0; i < (pulse_doppler_param->n_samples)*2; i++) {
        pulse_doppler_param->p4[i] = fftwf_plan_dft_1d(((pulse_doppler_param->m)), &(pulse_doppler_param->in_fft[i * (pulse_doppler_param->m)]), &(pulse_doppler_param->out_fft[i * (pulse_doppler_param->m)]), FFTW_FORWARD, FFTW_ESTIMATE);
    }

    FILE *fp;
    std::string pd_pulse = std::string(INPUT_DIR) + std::string("input_pd_pulse.txt");
    fp = fopen(pd_pulse.c_str(), "r");
    for(int i=0; i<2*N; i++){
        fscanf(fp, "%f", &(pulse_doppler_param->pulse[i]));
    }
    fclose(fp);

    std::string pd_ps = std::string(INPUT_DIR) + std::string("input_pd_ps.txt");
    fp = fopen(pd_ps.c_str(), "r");
    for(int j = 0; j < 2 * N*M; j++){
        fscanf(fp, "%f", &(pulse_doppler_param->p[j]));
    }
    fclose(fp);

    printf("Finished pulse_doppler_init\n");
}

#ifdef INCLUDE_MAIN
using json = nlohmann::json;

void generateJSON(void) {
    json output, DAG;
    unsigned int idx_offset = 0;

    //--- Application information
    output["AppName"] = "pulse_doppler";
    output["SharedObject"] = "pulse_doppler.so";
    output["FieldsBytes"] = sizeof(struct pulse_doppler_fields);

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
                {"runfunc", "pulse_doppler_" + run_func}
            },
            {
                {"name", "fft"},
                {"nodecost", 30.0f},
                {"runfunc", "pulse_doppler_" + run_func}
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
                {"runfunc", "pulse_doppler_IFFT"}
            },
            {
                {"name", "fft"},
                {"nodecost", 30.0f},
                {"runfunc", "pulse_doppler_IFFT"}
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
                {"runfunc", "pulse_doppler_FFT_2"}
            },
            {
                {"name", "fft"},
                {"nodecost", 30.0f},
                {"runfunc", "pulse_doppler_FFT_2"}
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
    output_file << std::setw(4) << output;
//    std::cout << std::setw(4) << output << std::endl;
}

int main(void) {
    generateJSON();
}
#endif