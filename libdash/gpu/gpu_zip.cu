#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

// TODO: this should rely on dash_types.h
typedef enum zip_op {
  ZIP_ADD = 0,
  ZIP_SUB = 1,
  ZIP_MULT = 2,
  ZIP_DIV = 3,
  ZIP_CMP_MULT = 4
} zip_op_t;
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
extern "C" __global__ void vector_add(const double* x, const double* y, double* z, int len) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < len) {
    z[id] = x[id] + y[id];
  }
}

extern "C" __global__ void vector_sub(const double* x, const double* y, double* z, int len) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < len) {
    z[id] = x[id] - y[id];
  }
}

extern "C" __global__ void vector_mult(const double* x, const double* y, double* z, int len) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < len) {
    z[id] = x[id] * y[id];
  }
}

extern "C" __global__ void vector_div(const double* x, const double* y, double* z, int len) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < len) {
    if(y[id]!=0)
      z[id] = x[id] / y[id];
    else
      z[id] = x[id];
  }
}

extern "C" __global__ void vector_cmp_mult(const double* x, const double* y, double* z, int len) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double r1=x[id*2], r2=y[id*2];
  double i1=x[id*2+1], i2=y[id*2+1];
  if (id < len) {
    z[id*2] = r1*r2 - i1*i2;
    z[id*2+1] = r1*i2 + r2*i1;
  }
}

extern "C" void DASH_ZIP_gpu(double** x, double** y, double** z, int* h_len, zip_op_t* op) {
//    printf("---------------------------------------\n");
//    printf("------------- Zip on GPU --------------\n");
//    printf("---------------------------------------\n");
    const int length = *h_len;
    const int size_in_bytes = length * sizeof(double);
    const int threadsPerBlock = 512;
    const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t err = cudaSuccess;

    double* d_x = NULL;
    double* d_y = NULL;
    double* d_z = NULL;
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
      case ZIP_CMP_MULT:
//	printf("Running vector complex mult kernel\n");
        vector_cmp_mult<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length/2);
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
    const int length = 2048*2048;
    const int size_in_bytes = length * sizeof(double);
    const int threadsPerBlock = 32;
    const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t err = cudaSuccess;

    double* d_x = NULL;
    double* d_y = NULL;
    double* d_z = NULL;
    err = cudaMalloc((void**)&d_x, size_in_bytes);
    err = cudaMalloc((void**)&d_y, size_in_bytes);
    err = cudaMalloc((void**)&d_z, size_in_bytes);

    double *input1, *input2, *output_cpu, *output_gpu;
    input1 = (double *)calloc(length, sizeof(double));
    input2 = (double *)calloc(length, sizeof(double));
    output_cpu = (double *)calloc(length, sizeof(double));
    output_gpu = (double *)calloc(length, sizeof(double));
    for (int i = 0; i < length; i++){
      input1[i]=i;
      input2[i]=i;
    }
    err = cudaMemcpy(d_x, input1, size_in_bytes, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_y, input2, size_in_bytes, cudaMemcpyHostToDevice);
    printf("Testing vector add...\n");
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
    for (int i = 0; i < length; i++){
      output_cpu[i] = input1[i] + input2[i];
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
    printf("Testing vector sub...\n");
    vector_sub<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
    for (int i = 0; i < length; i++){
      output_cpu[i] = input1[i] - input2[i];
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
    printf("Testing vector complex mult...\n");
    vector_cmp_mult<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length/2);
    for (int i = 0; i < length/2; i++){
      output_cpu[2*i] = input1[2*i] * input2[2*i] - input1[2*i+1] * input2[2*i+1];
      output_cpu[2*i+1] = input1[2*i] * input2[2*i+1] + input1[2*i+1] * input2[2*i];
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
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    return 0;
}

