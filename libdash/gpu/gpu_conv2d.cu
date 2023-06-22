#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

#define BLOCK_SIZE 32
__constant__ float mask[1024];

extern "C" __global__ void conv2D(const int16_t* input, int16_t* output, int width, int height, int mask_size, int tile_size/*, const double* __restrict__ mask*/) {
  double result = 0;
  int COL_out = threadIdx.x + blockIdx.x * tile_size; // Col of the thread
  int ROW_out = threadIdx.y + blockIdx.y * tile_size; // Row of the thread

  int COL = COL_out - mask_size / 2;
  int ROW = ROW_out - mask_size / 2;

  // Shared memory for the current tile
  __shared__ int16_t TILE[BLOCK_SIZE][BLOCK_SIZE];

  // 0 pad the shared memory based on the tile coordinates
  if(ROW >= 0 && ROW < height && COL >= 0 && COL < width){
    TILE[threadIdx.y][threadIdx.x] = input[ROW * width + COL];
  }
  else{
    TILE[threadIdx.y][threadIdx.x] = 0;
  }
  // Wait for all threads to complete their initilization of the shared memory space
  __syncthreads();

  // Boundary conditions for threads that takes part in the execution part
  if(threadIdx.y < tile_size && threadIdx.x < tile_size){
    for(int i = 0; i < mask_size; i++){
      for(int j = 0; j < mask_size; ++j){
        result += mask[i * mask_size + j] * TILE[threadIdx.y + i][threadIdx.x + j];
      }
    }
    if(ROW_out < height && COL_out < width){
      output[ROW_out * width + COL_out] = floorf(result);
    }
  }
}

extern "C" void DASH_CONV_2D_gpu(int16_t** input, int *height_in, int *width_in, float **mask_in, int *mask_size_in, int16_t **output) {
//    printf("---------------------------------------\n");
//    printf("----------- CONV-2D on GPU ------------\n");
//    printf("---------------------------------------\n");
  const int width = *width_in;
  const int height = *height_in;
  const int mask_size = *mask_size_in;
  const int tile_size = BLOCK_SIZE - mask_size + 1;

  dim3 grid_dim(((width-1)/tile_size)+1, ((height-1)/tile_size)+1, 1);
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);

  const int size_in_bytes = width * height * sizeof(int16_t);
  const int size_in_bytes_mask = mask_size * mask_size * sizeof(float);
  cudaError_t err = cudaSuccess;

  int16_t* d_input;
  int16_t* d_output;
/*  double* d_mask = NULL;*/
  err = cudaMalloc((void**)&d_input, size_in_bytes);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "1Failed initializations for conv2D on gpu (error code %s)%d\n", cudaGetErrorString(err),size_in_bytes);
  }
  err = cudaMalloc((void**)&d_output, size_in_bytes);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "2Failed initializations for conv2D on gpu (error code %s)\n", cudaGetErrorString(err));
  }
//  err = cudaMalloc((void**)&d_mask, size_in_bytes_mask);

  err = cudaMemcpy(d_input, (*input), size_in_bytes, cudaMemcpyHostToDevice);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "3Failed initializations for conv2D on gpu (error code %s)\n", cudaGetErrorString(err));
  }
  if (mask_size * mask_size > 1024){
    printf("Using more comnstant memory than defined! Change the API call or update the convstant memory definition for convolution!\n");
  }
  err = cudaMemcpyToSymbol(mask, (*mask_in), size_in_bytes_mask);

//  err = cudaMemcpy(d_mask, (*mask_in), size_in_bytes_mask, cudaMemcpyHostToDevice);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "4Failed initializations for conv2D on gpu (error code %s)\n", cudaGetErrorString(err));
  }

  conv2D<<<grid_dim, block_dim>>>(d_input, d_output, width, height, mask_size, tile_size);//, d_mask);

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "5Failed to launch kernel for conv2D on gpu (error code %s)\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
  err = cudaMemcpy((*output), d_output, size_in_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_output); cudaFree(d_input); //cudaFree(d_mask);

  return;
}
