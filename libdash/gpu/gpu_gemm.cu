#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>

#include "dash_types.h"
#include "platform.h"

// Compute C = A * B
__global__ void matrixMultiply(const dash_cmplx_flt_type *A, const dash_cmplx_flt_type *B, dash_cmplx_flt_type *C,
				int numARows, int numAColumns,
				int numBRows, int numBColumns,
			       	int numCRows, int numCColumns) {
  //@@ Insert code to implement basic matrix multiplication for
  //@@ arbitrary size using global memory. 
  int ROW = blockIdx.y*blockDim.y+threadIdx.y; // Calculate Row based on threads position in the block
  int COL = blockIdx.x*blockDim.x+threadIdx.x; // Calculate Col based on threads position in the block
  if( (ROW<numCRows) && (COL<numCColumns) ){ // As long as the thread is in the boundries of C compute the result for sum(col*row)
    dash_cmplx_flt_type partial_p; // Initialize output to be written to the C
    partial_p.re = 0;
    partial_p.im = 0;
    for( int i = 0; i<numAColumns ; i++ ){
      partial_p.re += A[ROW*numAColumns+i].re * B[i*numBColumns+COL].re - A[ROW*numAColumns+i].im * B[i*numBColumns+COL].im; // Compute the A[row][i]*B[i][col] real
      partial_p.im += A[ROW*numAColumns+i].re * B[i*numBColumns+COL].im + A[ROW*numAColumns+i].im * B[i*numBColumns+COL].re; // Compute the A[row][i]*B[i][col] imag
    }
    C[ROW*numCColumns+COL].re = partial_p.re; // Write back the result real
    C[ROW*numCColumns+COL].im = partial_p.im; // Write back the result imag
  }
}


extern "C" void DASH_GEMM_flt_gpu(dash_cmplx_flt_type** A, dash_cmplx_flt_type** B, dash_cmplx_flt_type** C, size_t* A_ROWS, size_t* A_COLS, size_t* B_COLS) {
//    int dev_count;
//    cudaGetDeviceCount(&dev_count);
//    cudaSetDevice(resource_idx%dev_count);
//    printf("---------------------------------------\n");
//    printf("------------- GEMM on GPU --------------\n");
//    printf("---------------------------------------\n");
    const int row_a = *A_ROWS;
    const int col_a = *A_COLS;
    const int col_b = *B_COLS;
    int block_size = 16;
    dim3 grid_dim(((col_b-1)/block_size)+1,((row_a-1)/block_size)+1,1);
    dim3 block_dim(block_size,block_size,1);
    cudaError_t err = cudaSuccess;

    dash_cmplx_flt_type* d_A = NULL;
    dash_cmplx_flt_type* d_B = NULL;
    dash_cmplx_flt_type* d_C = NULL;
    err = cudaMalloc((void**)&d_A, row_a*col_a*sizeof(dash_cmplx_flt_type));
    err = cudaMalloc((void**)&d_B, col_a*col_b*sizeof(dash_cmplx_flt_type));
    err = cudaMalloc((void**)&d_C, row_a*col_b*sizeof(dash_cmplx_flt_type));

    err = cudaMemcpy(d_A, (*A), row_a*col_a*sizeof(dash_cmplx_flt_type), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, (*B), col_a*col_b*sizeof(dash_cmplx_flt_type), cudaMemcpyHostToDevice);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel for zip on gpu (error code %s)\n", cudaGetErrorString(err));
    }
    
//    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    matrixMultiply<<<grid_dim, block_dim>>>(d_A, d_B, d_C,
					row_a, col_a,
					col_a, col_b,
					row_a, col_b);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel for zip on gpu (error code %s)\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    err = cudaMemcpy((*C), d_C, row_a*col_b*sizeof(dash_cmplx_flt_type), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return;
}

