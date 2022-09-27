#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

__global__ void vector_add(const int* x, const int* y, int* z, int len) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < len) {
        z[id] = x[id] + y[id];
    }
}

extern "C" void Vector_Add_GPU(int** x, int** y, int** z, int* h_len) {
    printf("---------------------------------------\n");
    printf("------- Vector Addition on GPU --------\n");
    printf("---------------------------------------\n");
    const int length = *h_len;
    const int size_in_bytes = length * sizeof(int);
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t err = cudaSuccess;

    int* d_x = NULL;
    int* d_y = NULL;
    int* d_z = NULL;
    err = cudaMalloc((void**)&d_x, size_in_bytes);
    err = cudaMalloc((void**)&d_y, size_in_bytes);
    err = cudaMalloc((void**)&d_z, size_in_bytes);

    err = cudaMemcpy(d_x, (*x), size_in_bytes, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_y, (*y), size_in_bytes, cudaMemcpyHostToDevice);

    
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, length);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vector_add kernel (error code %s)\n", cudaGetErrorString(err));
    }

    cudaMemcpy((*z), d_z, size_in_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);

    return;
}