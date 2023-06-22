#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>

#include <cuda_runtime.h>
#include <cufft.h>

#include "dash_types.h"
#include "platform.h"

//#error "GPU FFT module must be updated to adhere to the updated FFT_int/FFT_flt API changes -- see FFT module for an example"

cufftComplex *d_fft256, *d_fft512, *d_fft1024, *d_fft2048;
cufftHandle plan_fft, plan_fft256, plan_fft512, plan_fft1024, plan_fft2048, plan_fft_custom;

extern "C" void __attribute__((constructor)) setup(void) {
    printf("[gpu] Allocating initial memory buffers and baseline FFT plans\n");
    cufftPlan1d(&plan_fft256, 256, CUFFT_C2C, 1);
  
    cufftPlan1d(&plan_fft512, 512, CUFFT_C2C, 1);
 
    cufftPlan1d(&plan_fft1024, 1024, CUFFT_C2C, 1);

    cufftPlan1d(&plan_fft2048, 2048, CUFFT_C2C, 1);

    printf("[gpu] Initial allocation complete!\n");
}
  
extern "C" void __attribute__((destructor)) teardown(void) {
    printf("[gpu] Tearing down GPU memory buffers and FFT plans\n");

// Looks like cuda already frees whatever we have even before coming to this destrcutor
// and c++ compiler removes all global variables by the time we reach here so we get a
// seg_fault when we try to free these memories!

/*
    cudaFree(d_fft256);
    cudaFree(d_fft512);
    cudaFree(d_fft1024);
    cudaFree(d_fft2048);

    cufftDestroy(plan_fft256);
    cufftDestroy(plan_fft512);
    cufftDestroy(plan_fft1024);
    cufftDestroy(plan_fft2048);
*/
    printf("[gpu] Teardown complete!\n");
}

extern "C" void fft_cuda(dash_cmplx_flt_type* input, dash_cmplx_flt_type* output, int64_t fft_size, bool isForwardTransform, cufftHandle plan, cufftComplex *d_data) {
    //printf("===========================================================\n");
    //printf("=============== Executing %ld-Pt %s on GPU ================\n", fft_size, isForwardTransform ? "FFT" : "IFFT");
    //printf("===========================================================\n");

    //double *tmp_d_data = (double*) d_data;
    cudaMemcpy(d_data, input, fft_size * sizeof(input[0]), cudaMemcpyHostToDevice);

    cufftExecC2C(plan, d_data, d_data, isForwardTransform ? CUFFT_FORWARD : CUFFT_INVERSE);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, d_data, fft_size * sizeof(output[0]), cudaMemcpyDeviceToHost);
}

extern "C" void DASH_FFT_flt_gpu(dash_cmplx_flt_type** input, dash_cmplx_flt_type** output, size_t* size, bool* isForwardTransform, uint8_t resource_idx){
  int dev_count;
  cudaGetDeviceCount(&dev_count);
  cudaSetDevice(resource_idx%dev_count);
  //printf("GPU id is %d\n", (resource_idx%dev_count));
  //float* inp = (float*) malloc(2 * (*size) * sizeof(float));
  //float* out = (float*) malloc(2 * (*size) * sizeof(float));

  //for (size_t i = 0; i < (2 * (*size)); i++) {
  //  inp[i] = (float) ((*input)[i]);
  //}
  cufftComplex *d_fft;
  cudaMalloc((void**) &d_fft, sizeof(cufftComplex)*(*size));
  if(*size == 256){
    plan_fft = plan_fft256;
  }
  else if(*size == 512){
    plan_fft = plan_fft512;
  }
  else if(*size == 1024){
    plan_fft = plan_fft1024;
  }
  else  if(*size == 2048){
    plan_fft = plan_fft2048;
  }
  else {
	printf("Running fft with custom length!\n");
  	cufftPlan1d(&plan_fft_custom, (*size), CUFFT_C2C, 1);
	plan_fft = plan_fft_custom;
  }

  fft_cuda(*input, *output, *size, *isForwardTransform, plan_fft, d_fft);

  if(!(*isForwardTransform)){
    for (size_t i = 0; i < (*size); i++) {
      (*output)[i].re = (*output)[i].re / (*size);
      (*output)[i].im = (*output)[i].im / (*size);
    }
  }

  cudaFree(d_fft);
  if(!(*size == 256 || *size == 512 || *size == 1024 || *size == 2048)){
    cufftDestroy(plan_fft_custom);
  }
}

uint32_t _check_fft_result_(dash_cmplx_flt_type *fft_actual, dash_cmplx_flt_type *fft_expected, size_t FFT_DIM) {
    int error_count = 0;
    float diff, c, d;
  
    printf("[fft] Checking actual versus expected output\n");
    for (size_t i = 0; i < FFT_DIM; i++) {
      c = fft_expected[i].re;
      d = fft_actual[i].re;
  
      diff = std::abs(c - d) / c * 100;
  
      if (diff > 0.01) {
        fprintf(stderr, "[fft] [ERROR] Expected = %f, Actual FFT = %f, index = %ld\n", c, d, i);
        error_count++;
      }
    }
    for (size_t i = 0; i < FFT_DIM; i++) {
      c = fft_expected[i].im;
      d = fft_actual[i].im;
  
      diff = std::abs(c - d) / c * 100;
  
      if (diff > 0.01) {
        fprintf(stderr, "[fft] [ERROR] Expected = %f, Actual FFT = %f, index = %ld\n", c, d, i);
        error_count++;
      }
    }
  
    if (error_count == 0) {
      printf("[fft] FFT Passed!\n");
      return 0;
    } else {
      fprintf(stderr, "[fft] FFT Failed!\n");
      return 1;
    }
}
  
void _generate_fft_test_values_(dash_cmplx_flt_type *fft_input, dash_cmplx_flt_type *fft_expected, size_t FFT_SIZE, bool sparse_input) {
  switch (FFT_SIZE) {
  case 256:
  case 512:
  case 2048:
    if (sparse_input) {
      // Let's just do a delta function, the expected output is easy to calculate that way
      fft_input[0].re = 1;
      // As such, the expected output is the constant 1 function
      for (size_t i = 0; i < FFT_SIZE; i++) {
        fft_expected[i].re = 1;
        fft_expected[i].im = 0;
      }
    } else {
      // Otherwise, let's do all 1's as the input
      for (size_t i = 0; i < FFT_SIZE; i++) {
        fft_input[i].re = 1;
        fft_input[i].im = 0;
      }
      // And as such, the output is some form of delta function (height = FFT_SIZE?)
      fft_expected[0].re = FFT_SIZE;
    }
    break;
  default:
    fprintf(stderr, "[fft] Unknown FFT_SIZE specified in _generate_fft_test_values_: %ld\n", FFT_SIZE);
    exit(1);
  }
}
  
void _print_fft_result_(dash_cmplx_flt_type *fft_result, size_t FFT_SIZE, bool isForwardTransform) {
  printf("[fft] Printing received %ld-Pt %s output\n", FFT_SIZE, isForwardTransform ? "FFT" : "IFFT");
  for (size_t i = 0; i < FFT_SIZE; i++) {
    printf("(%f + %fi)", fft_result[i].re, fft_result[i].im);
    if (i < FFT_SIZE - 1) {
      printf(", ");
    }
  }
  printf("\n");
}
 
uint32_t _run_fft_tests_(size_t FFT_SIZE, bool sparse_input, bool exit_on_failure, bool print_intermediate_results) {
  dash_cmplx_flt_type *fft_input, *fft_expected, *fft_actual_cpu, *fft_actual_accel;
  const bool isForwardTransform = true;
  const char* FFT_TYPE = isForwardTransform ? "FFT" : "IFFT";

  uint32_t tests_failed = 0;

  cufftHandle plan;
  cufftComplex *d_fftData;
 
  printf("###################################################\n");
  if (sparse_input) {
    printf("[fft] Testing %ld-Pt %s implementations (sparse)\n", FFT_SIZE, FFT_TYPE);
  } else {
    printf("[fft] Testing %ld-Pt %s implementations (dense) \n", FFT_SIZE, FFT_TYPE);
  }
  printf("###################################################\n");

  if (FFT_SIZE == 256 && isForwardTransform) {
    plan = plan_fft256;
    d_fftData = d_fft256;
  } else if (FFT_SIZE == 512 && isForwardTransform) {
    plan = plan_fft512;
    d_fftData = d_fft512;
  } else if (FFT_SIZE == 2048 && isForwardTransform) {
    plan = plan_fft2048;
    d_fftData = d_fft2048;
  }
  
  fft_input = (dash_cmplx_flt_type *)calloc(FFT_SIZE, sizeof(dash_cmplx_flt_type));
  fft_expected = (dash_cmplx_flt_type *)calloc(FFT_SIZE, sizeof(dash_cmplx_flt_type));
  fft_actual_cpu = (dash_cmplx_flt_type *)calloc(FFT_SIZE, sizeof(dash_cmplx_flt_type));
  fft_actual_accel = (dash_cmplx_flt_type *)calloc(FFT_SIZE, sizeof(dash_cmplx_flt_type));
  
  printf("[fft] Generating input data and expected output\n");
  _generate_fft_test_values_(fft_input, fft_expected, FFT_SIZE, sparse_input);

  printf("[fft] Computing %ld-Pt FFT using the GPU kernel\n", FFT_SIZE);
  fft_cuda(fft_input, fft_actual_accel, FFT_SIZE, isForwardTransform, plan, d_fftData);
  
  if (print_intermediate_results) {
    _print_fft_result_(fft_actual_accel, FFT_SIZE, isForwardTransform);
  }
  
  printf("[fft] Checking the GPU output against the reference expected values\n");
  tests_failed += _check_fft_result_(fft_actual_accel, fft_expected, FFT_SIZE);
  
  if (exit_on_failure && tests_failed > 0) {
    exit(1);
  }
  
  printf("[fft] %ld-Pt FFT tests complete!\n", FFT_SIZE);
  
  free(fft_input);
  free(fft_expected);
  free(fft_actual_cpu);
  free(fft_actual_accel);
  
  return tests_failed;
}
/*
int main() {
  // Stores the cumulative total number of tests that have failed
  uint32_t tests_failed = 0;
  // If true, exit upon the first failed test ("fail fast")
  bool exit_on_failure = true;
  // If true, print full FFT output from each implementation as it runs
  bool print_intermediate_outputs = false;
  
  // Sparse 256-Pt tests
  tests_failed += _run_fft_tests_(256, true, exit_on_failure, print_intermediate_outputs);
  // Dense 256-Pt tests
  tests_failed += _run_fft_tests_(256, false, exit_on_failure, print_intermediate_outputs);
  // Sparse 512-Pt tests
  tests_failed += _run_fft_tests_(512, true, exit_on_failure, print_intermediate_outputs);
  // Dense 512-Pt tests
  tests_failed += _run_fft_tests_(512, false, exit_on_failure, print_intermediate_outputs);
  // Sparse 2048-Pt tests
  tests_failed += _run_fft_tests_(2048, true, exit_on_failure, print_intermediate_outputs);
  // Dense 2048-Pt tests
  tests_failed += _run_fft_tests_(2048, false, exit_on_failure, print_intermediate_outputs);

  printf("[fft] All FFT tests completed - %d tests failed!\n", tests_failed);
  return (tests_failed > 0) ? 1 : 0;
}
*/
