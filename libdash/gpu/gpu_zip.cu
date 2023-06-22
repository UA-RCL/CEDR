#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>

#include "dash_types.h"
#include "platform.h"

/*
typedef enum zip_op {
  ZIP_ADD = 0,
  ZIP_SUB = 1,
  ZIP_MULT = 2,
  ZIP_DIV = 3,
} zip_op_t;
*/
/*
void __attribute__((constructor)) setup(void) {
    printf("[gpu] Allocating initial memory buffers and baseline for ZIP\n");

    printf("[gpu] Initial zip allocation complete!\n");
}

void __attribute__((destructor)) teardown(void) {
    printf("[gpu] Tearing down GPU memory buffers for ZIP\n");
    printf("[gpu] Teardown complete!\n");
}
*/
extern "C" __global__ void vector_add(const dash_cmplx_flt_type* x, const dash_cmplx_flt_type* y, dash_cmplx_flt_type* z, int len) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < len) {
    z[id].re = x[id].re + y[id].re;
    z[id].im = x[id].im + y[id].im;
  }
}

extern "C" __global__ void vector_sub(const dash_cmplx_flt_type* x, const dash_cmplx_flt_type* y, dash_cmplx_flt_type* z, int len) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < len) {
    z[id].re = x[id].re - y[id].re;
    z[id].im = x[id].im - y[id].im;
  }
}

extern "C" __global__ void vector_mult(const dash_cmplx_flt_type* x, const dash_cmplx_flt_type* y, dash_cmplx_flt_type* z, int len) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  dash_re_flt_type r1=x[id].re, r2=y[id].re;
  dash_re_flt_type i1=x[id].im, i2=y[id].im;
  if (id < len) {
    z[id].re = r1*r2 - i1*i2;
    z[id].im = r1*i2 + r2*i1;
  }
}

extern "C" __global__ void vector_div(const dash_cmplx_flt_type* x, const dash_cmplx_flt_type* y, dash_cmplx_flt_type* z, int len) {
  printf("Division for ZIP not working corretly!");
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < len) {
    z[id].re = ( x[id].re * y[id].re + x[id].im * y[id].im)/(y[id].re*y[id].re + y[id].im*y[id].im);
    z[id].im = (-x[id].re * y[id].im + x[id].im * y[id].re)/(y[id].re*y[id].re + y[id].im*y[id].im);
  }
}


extern "C" void DASH_ZIP_flt_gpu(dash_cmplx_flt_type** x, dash_cmplx_flt_type** y, dash_cmplx_flt_type** z, int* h_len, zip_op_t* op, uint8_t resource_idx) {
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cudaSetDevice(resource_idx%dev_count);
//    printf("GPU id is %d\n", (resource_idx%dev_count));
//    printf("---------------------------------------\n");
//    printf("------------- Zip on GPU --------------\n");
//    printf("---------------------------------------\n");
    const int length = *h_len;
    const int size_in_bytes = length * sizeof(dash_cmplx_flt_type);
    const int threadsPerBlock = 512;
    const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t err = cudaSuccess;

    dash_cmplx_flt_type* d_x = NULL;
    dash_cmplx_flt_type* d_y = NULL;
    dash_cmplx_flt_type* d_z = NULL;
    err = cudaMalloc((void**)&d_x, size_in_bytes);
    err = cudaMalloc((void**)&d_y, size_in_bytes);
    err = cudaMalloc((void**)&d_z, size_in_bytes);

    err = cudaMemcpy(d_x, (*x), size_in_bytes, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_y, (*y), size_in_bytes, cudaMemcpyHostToDevice);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel for zip on gpu (error code %s)\n", cudaGetErrorString(err));
    }
    
//    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    switch (*op) {
      case ZIP_ADD:
//	printf("Running vector add kernel\n");
        vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
        break;
      case ZIP_SUB:
//	printf("Running vector sub kernel\n");
        vector_sub<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
        break;
      case ZIP_MULT:
//	printf("Running vector mult kernel\n");
        vector_mult<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
        break;
      case ZIP_DIV:
//	printf("Running vector div kernel\n");
        vector_div<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
        break;
    }
  

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel for zip on gpu (error code %s)\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
  err = cudaMemcpy((*z), d_z, size_in_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);

  return;
}

int main(){
    printf("Running ZIP Cuda Test...\n");
    const int length = 2048;
    const int size_in_bytes = length * sizeof(dash_cmplx_flt_type);
    const int threadsPerBlock = 512;
    const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t err = cudaSuccess;

    dash_cmplx_flt_type* d_x = NULL;
    dash_cmplx_flt_type* d_y = NULL;
    dash_cmplx_flt_type* d_z = NULL;
    err = cudaMalloc((void**)&d_x, size_in_bytes);
    err = cudaMalloc((void**)&d_y, size_in_bytes);
    err = cudaMalloc((void**)&d_z, size_in_bytes);

    dash_cmplx_flt_type *input1, *input2, *output_cpu, *output_gpu;
    input1 = (dash_cmplx_flt_type *)calloc(length, sizeof(dash_cmplx_flt_type));
    input2 = (dash_cmplx_flt_type *)calloc(length, sizeof(dash_cmplx_flt_type));
    output_cpu = (dash_cmplx_flt_type *)calloc(length, sizeof(dash_cmplx_flt_type));
    output_gpu = (dash_cmplx_flt_type *)calloc(length, sizeof(dash_cmplx_flt_type));
    for (int i = 0; i < length; i++){
      input1[i].re=i;
      input2[i].re=i;
      input1[i].im=0;
      input2[i].im=0;
    }
    err = cudaMemcpy(d_x, input1, size_in_bytes, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_y, input2, size_in_bytes, cudaMemcpyHostToDevice);
    printf("Testing vector add...\n");
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
    for (int i = 0; i < length; i++){
      output_cpu[i].re = input1[i].re + input2[i].re;
      output_cpu[i].im = input1[i].im + input2[i].im;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel for zip on gpu (error code %s)\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(output_gpu, d_z, size_in_bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++){
      if (output_cpu[i].re!=output_gpu[i].re){
	      printf("Error! %d: %lf\t%lf\n", i, output_cpu[i].re, output_gpu[i].re);
      }
    }
    printf("Testing vector sub...\n");
    vector_sub<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
    for (int i = 0; i < length; i++){
      output_cpu[i].re = input1[i].re - input2[i].re;
      output_cpu[i].im = input1[i].im - input2[i].im;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel for zip on gpu (error code %s)\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(output_gpu, d_z, size_in_bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++){
      if (output_cpu[i].re!=output_gpu[i].re){
              printf("Error! %d: %lf\t%lf\n", i, output_cpu[i].re, output_gpu[i].re);
      }
    }
    /*
    printf("Testing vector mult...\n");
    vector_mult<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
    for (int i = 0; i < length; i++){
      output_cpu[i] = input1[i] * input2[i];
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel for zip on gpu (error code %s)\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(output_gpu, d_z, size_in_bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++){
      if (output_cpu[i]!=output_gpu[i]){
              printf("Error! %d: %lf\t%lf\n", i, output_cpu[i], output_gpu[i]);
      }
    }
    printf("Testing vector div...\n");
    vector_div<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
    for (int i = 0; i < length; i++){
      if(input2[i]!=0)
        output_cpu[i] = input1[i] / input2[i];
      else
        output_cpu[i] = input1[i];
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel for zip on gpu (error code %s)\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(output_gpu, d_z, size_in_bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++){
      if (output_cpu[i]!=output_gpu[i]){
              printf("Error! %d: %lf\t%lf\n", i, output_cpu[i], output_gpu[i]);
      }
    }
    */
    printf("Testing vector complex mult...\n");
    vector_mult<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
    for (int i = 0; i < length; i++){
      output_cpu[i].re = input1[i].re * input2[i].re - input1[i].im * input2[i].im;
      output_cpu[i].im = input1[i].re * input2[i].im + input1[i].im * input2[i].re;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel for zip on gpu (error code %s)\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(output_gpu, d_z, size_in_bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++){
      if (output_cpu[i].re!=output_gpu[i].re || output_cpu[i].im!=output_gpu[i].im){
              printf("Error! %d: %lf\t%lf\n", i, output_cpu[i].re, output_gpu[i].re);
              printf("Error! %d: %lf\t%lf\n", i, output_cpu[i].im, output_gpu[i].im);
      }
    }
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    return 0;
}

